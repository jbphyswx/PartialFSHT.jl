"""
The public matrix-free operator.

`Plan(θ, φ; lmax)` builds a reusable synthesis `𝒴_R : a_{ℓm} ↦ f(Ω_i)` on an arbitrary point
set, with the exact Euclidean adjoint as `plan'`. Both directions are correct at any sky
coverage; see the manual on what an *inversion* from partial data can and cannot recover.
"""

struct Plan{T, LF, BE, VT, NB}
    lmax::Int
    ntrans::Int
    spin::Int
    tol::T
    leaf::LF
    backend::BE
    nufft::NB
    θ::VT
    φ::VT
end

struct AdjointPlan{P}
    parent::P
end

LinearAlgebra.adjoint(p::Plan) = AdjointPlan(p)
LinearAlgebra.adjoint(p::AdjointPlan) = p.parent

function Base.show(io::IO, p::Plan)
    print(io, "PartialFSHT.Plan(lmax=", p.lmax, ", npts=", length(p.θ),
          ", ntrans=", p.ntrans, ", spin=", p.spin,
          ", algorithm=", nameof(typeof(algorithm(p))), ")")
end

"""
    algorithm(plan) -> AbstractAlgorithm

Tag for the leaf the constructor chose.
"""
algorithm(p::Plan) = algorithm(p.leaf)

"""
    ring_work(plan) -> Int

Retained `(ℓ, m, ring)` triples per apply. A counted quantity, so scaling claims can be gated
by assertion rather than by wall-clock.
"""
ring_work(p::Plan) = ring_work(p.leaf)
ring_work(::DirectLeaf) = 0

@inline _resolve_backend(backend::ComputationalBackends.AbstractExecutionBackend) = backend
@inline _resolve_backend(::ComputationalBackends.AbstractAutoBackend) =
    (Threads.nthreads() > 1 && _omt_loaded()) ? ComputationalBackends.ThreadedBackend() :
                                                ComputationalBackends.SerialBackend()

_backend_unavailable(backend, what) = throw(ArgumentError(
    "$(nameof(typeof(backend))) cannot run $what here — the extension providing it is not loaded. " *
    "ThreadedBackend needs `using OhMyThreads`, DistributedBackend `using Distributed`, " *
    "MPIBackend `using MPI`."))

"""
    Plan(θ, φ; lmax, tol=1e-10, spin=0, ntrans=1, method=AutoAlgorithm(),
         backend=SerialBackend(), nufft=AutoNUFFT(), maxboxes=8)

Reusable transform for colatitudes `θ ∈ [0,π]` and longitudes `φ ∈ [0,2π)`.

`method` is an [`AbstractAlgorithm`](@ref); `AutoAlgorithm` picks by counted cost. With
`MPIBackend`, `θ`/`φ` are the **rank-local** points: synthesis writes the local field and the
adjoint `Allreduce`s coefficients, so passing the global point set on every rank would scale the
adjoint by the communicator size.
"""
function Plan(θ::AbstractVector, φ::AbstractVector;
              lmax::Integer,
              tol::Real = 1e-10,
              spin::Integer = 0,
              ntrans::Integer = 1,
              method::AbstractAlgorithm = AutoAlgorithm(),
              backend = ComputationalBackends.SerialBackend(),
              nufft = AutoNUFFT(),
              maxboxes::Integer = 8)
    length(θ) == length(φ) || throw(DimensionMismatch("θ and φ must have the same length"))
    lmax >= 0 || throw(ArgumentError("lmax must be nonnegative"))
    ntrans >= 1 || throw(ArgumentError("ntrans must be ≥ 1"))
    spin == 0 || abs(spin) <= lmax ||
        throw(ArgumentError("(|spin| = $(abs(spin))) > lmax = $lmax"))
    be = _resolve_backend(backend)
    T = float(eltype(θ))
    chosen = select_method(θ, φ, lmax, tol, method, ntrans, Int(spin))
    # Resolve the NUFFT only for leaves that use one: `AutoNUFFT` errors when no fast backend is
    # loaded, and a direct evaluation has no business demanding one.
    nufft_b = (chosen isa StretchAlgorithm || chosen isa MultiBoxAlgorithm) ?
        _resolve_nufft(nufft) : nufft
    leaf = _build_leaf(chosen, θ, φ, Int(lmax), Int(ntrans), Int(spin), T(tol), nufft_b,
                       Int(maxboxes))
    return Plan{T, typeof(leaf), typeof(be), typeof(θ), typeof(nufft_b)}(
        Int(lmax), Int(ntrans), Int(spin), T(tol), leaf, be, nufft_b, θ, φ)
end

_build_leaf(alg::AbstractAlgorithm, args...) = throw(ArgumentError(
    "$(nameof(typeof(alg))) is not available — load its extension, or pass DirectAlgorithm()."))

_build_leaf(::DirectAlgorithm, θ, φ, lmax, ntrans, spin, tol, nufft_b, maxboxes) =
    spin == 0 ? DirectLeaf(θ, φ, lmax, ntrans) :
                SpinDirectLeaf(θ, φ, lmax, ntrans, spin)

_build_leaf(::StretchAlgorithm, θ, φ, lmax, ntrans, spin, tol, nufft_b, maxboxes) =
    StretchLeaf(θ, φ, lmax, ntrans, tol; nufft_backend = nufft_b, spin = spin)

_build_leaf(::MultiBoxAlgorithm, θ, φ, lmax, ntrans, spin, tol, nufft_b, maxboxes) =
    MultiBoxLeaf(θ, φ, lmax, ntrans, tol; nufft_backend = nufft_b, spin = spin,
                 maxboxes = maxboxes)

function LinearAlgebra.mul!(f, plan::Plan, alm)
    _assert_coeffs(alm, plan.lmax, plan.ntrans)
    _assert_field(f, length(plan.θ), plan.ntrans)
    return _mul_plan!(f, plan, alm, plan.backend)
end

function LinearAlgebra.mul!(alm, adj::AdjointPlan, f)
    plan = adj.parent
    _assert_coeffs(alm, plan.lmax, plan.ntrans)
    _assert_field(f, length(plan.θ), plan.ntrans)
    return _mul_adj!(alm, plan, f, plan.backend)
end

_mul_plan!(f, plan::Plan, alm, ::ComputationalBackends.AbstractSerialBackend) =
    _synthesize!(f, alm, plan.leaf, plan)
_mul_adj!(alm, plan::Plan, f, ::ComputationalBackends.AbstractSerialBackend) =
    _adjoint_apply!(alm, f, plan.leaf, plan)

function _mul_plan!(f, plan::Plan, alm, backend::ComputationalBackends.AbstractThreadedBackend)
    _omt_loaded() || _backend_unavailable(backend, "a threaded PartialFSHT apply")
    return _mul_plan_threaded!(f, plan, alm, backend)
end

function _mul_adj!(alm, plan::Plan, f, backend::ComputationalBackends.AbstractThreadedBackend)
    _omt_loaded() || _backend_unavailable(backend, "a threaded PartialFSHT apply")
    return _mul_adj_threaded!(alm, plan, f, backend)
end

_mul_plan!(f, plan::Plan, alm, backend::ComputationalBackends.AbstractMPIBackend) =
    _mul_plan_mpi!(f, plan, alm, backend)
_mul_adj!(alm, plan::Plan, f, backend::ComputationalBackends.AbstractMPIBackend) =
    _mul_adj_mpi!(alm, plan, f, backend)
_mul_plan!(f, plan::Plan, alm, backend::ComputationalBackends.AbstractDistributedBackend) =
    _mul_plan_dist!(f, plan, alm, backend)
_mul_adj!(alm, plan::Plan, f, backend::ComputationalBackends.AbstractDistributedBackend) =
    _mul_adj_dist!(alm, plan, f, backend)

_mul_plan_threaded!(args...) = throw(ArgumentError("load OhMyThreads for ThreadedBackend"))
_mul_adj_threaded!(args...) = throw(ArgumentError("load OhMyThreads for ThreadedBackend"))
_mul_plan_mpi!(args...) = throw(ArgumentError("load MPI for MPIBackend"))
_mul_adj_mpi!(args...) = throw(ArgumentError("load MPI for MPIBackend"))
_mul_plan_dist!(args...) = throw(ArgumentError("load Distributed for DistributedBackend"))
_mul_adj_dist!(args...) = throw(ArgumentError("load Distributed for DistributedBackend"))

_synthesize!(f, alm, leaf::DirectLeaf, ::Plan) = synthesize_direct!(f, alm, leaf)
_adjoint_apply!(alm, f, leaf::DirectLeaf, ::Plan) = adjoint_direct!(alm, f, leaf)
_synthesize!(f, alm, leaf::SpinDirectLeaf, ::Plan) = synthesize_spin_direct!(f, alm, leaf)
_adjoint_apply!(alm, f, leaf::SpinDirectLeaf, ::Plan) = adjoint_spin_direct!(alm, f, leaf)
_synthesize!(f, alm, leaf::StretchLeaf, ::Plan) = synthesize_stretch!(f, alm, leaf)
_adjoint_apply!(alm, f, leaf::StretchLeaf, ::Plan) = adjoint_stretch!(alm, f, leaf)
_synthesize!(f, alm, leaf::MultiBoxLeaf, ::Plan) = synthesize_multibox!(f, alm, leaf)
_adjoint_apply!(alm, f, leaf::MultiBoxLeaf, ::Plan) = adjoint_multibox!(alm, f, leaf)

function Base.:*(plan::Plan, alm)
    f = _field_zeros(plan.θ, eltype(alm), length(plan.θ), plan.ntrans)
    return LinearAlgebra.mul!(f, plan, alm)
end

function Base.:*(adj::AdjointPlan, f)
    plan = adj.parent
    alm = zeros_alm(plan.θ, plan.lmax; ntrans = plan.ntrans)
    return LinearAlgebra.mul!(alm, adj, f)
end

"""
    mul_nodeset(θs, φs, alms, lmax, backend=SerialBackend(); kwargs...) -> Vector

Synthesize independent node sets. Serial and threaded backends build a plan per set in this
process; `DistributedBackend` (after `using Distributed`) builds each plan on a worker so FFT
and NUFFT state is never serialized. This is not MPI point decomposition — use one `Plan` per
rank with that rank's local `(θ, φ)` for that.
"""
function mul_nodeset(θs, φs, alms, lmax,
                     backend::ComputationalBackends.AbstractExecutionBackend =
                         ComputationalBackends.SerialBackend(); kwargs...)
    length(θs) == length(φs) == length(alms) ||
        throw(DimensionMismatch("θs, φs, and alms must have the same length"))
    backend isa ComputationalBackends.AbstractMPIBackend && throw(ArgumentError(
        "mul_nodeset is independent node sets, not MPI point decomposition. " *
        "Build a Plan on each rank from that rank's (θ, φ)."))
    return [Plan(θs[i], φs[i]; lmax = lmax, backend = backend, kwargs...) * alms[i]
            for i in eachindex(θs)]
end
