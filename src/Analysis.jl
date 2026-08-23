# What partial data can actually yield, in the order a user should reach for it:
#
#   1. `recoverability` — what this geometry supports, before any compute is spent.
#   2. `spectrum`       — band-powers with the mask's own coupling accounted for. Well posed at any
#                         sky fraction, and the right answer to "which wavenumbers are doing what".
#   3. `solve!`         — coefficients, with the regularisation named rather than implied.
#
# The distinction that matters: A and A† are exact at any coverage, but A†f is NOT the field's
# coefficients — it is the masked field's, mode-coupled and leakage-biased. Recovering coefficients
# from a region is rank-deficient by construction, so every routine here reports what it did rather
# than returning numbers that look like truth. See the manual under Recoverability.
#
# Everything is bang-first against an explicit workspace: an iterative solve runs hundreds of
# applies, so allocating coefficient arrays per call would dominate the thing it is measuring.

"""
    recoverability(θ, φ, lmax) -> NamedTuple

What a geometry can support, computed from the geometry alone: `fsky`, the Shannon number
`N_S = (lmax+1)² f_sky`, the recoverable *fraction* of coefficient directions, and the spectral
resolution `Δℓ ≈ π/Λ`.

Two facts this is meant to make unmissable. The recoverable fraction is `f_sky` **independently
of `ℓ`** — more wavelengths across the patch improve the spectrum but never the fraction of
individually determined coefficients, because higher `ℓ` brings proportionally more of them. And
the useful output is banded: a 4° patch resolves `Δℓ ≈ 45`, a hemisphere `Δℓ ≈ 1`.
"""
function recoverability(θ::AbstractVector, φ::AbstractVector, lmax::Integer)
    box = bounding_box(θ, φ)
    L = Int(lmax)
    fsky = box_fsky(box)
    Δℓ = spectral_resolution(box)
    return (box = box, fsky = fsky, shannon = shannon_number(L, fsky),
            ncoeff = (L + 1)^2, recoverable_fraction = fsky, Δℓ = Δℓ,
            nbands = Δℓ > 0 && isfinite(Δℓ) ? max(1, Int(floor((L + 1) / Δℓ))) : 1,
            npoints = length(θ))
end

recoverability(plan::Plan) = recoverability(plan.θ, plan.φ, plan.lmax)

# ---------------------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------------------

"""
    AnalysisWorkspace(plan) -> AnalysisWorkspace

Preallocated buffers for [`apply_resolution!`](@ref), [`pseudo_spectrum!`](@ref) and
[`solve!`](@ref). One per plan; reused across calls so an iterative solve allocates nothing
after construction.
"""
struct AnalysisWorkspace{AC, AF, V, BX}
    rhs::AC
    r::AC
    p::AC
    Ap::AC
    alm::AC
    fbuf::AF
    fbuf2::AF
    Cl::V
    priorinv::V
    box::BX
end

# Buffers are allocated in the plan's *own* shapes — 2-D coefficients and a 1-D field when
# `ntrans == 1` — so nothing here ever takes a view. Allocating an oversized buffer and slicing
# it costs a heap-allocated `SubArray` per call, which is exactly the sort of thing that reads as
# "allocation-free except it isn't" in a solver running hundreds of applies.
function AnalysisWorkspace(plan::Plan)
    L = plan.lmax
    B = plan.ntrans
    CT = eltype(zeros_alm(plan.θ, 0))
    T = real(CT)
    mk() = zeros_alm(plan.θ, L; ntrans = B)
    return AnalysisWorkspace(mk(), mk(), mk(), mk(), mk(),
                             _field_zeros(plan.θ, CT, length(plan.θ), B),
                             _field_zeros(plan.θ, CT, length(plan.θ), B),
                             zeros(T, L + 1), zeros(T, L + 1),
                             bounding_box(plan.θ, plan.φ))
end

# ---------------------------------------------------------------------------------------
# Resolution operator
# ---------------------------------------------------------------------------------------

"""
    apply_resolution!(out, plan, alm, w, ws) -> out

Push a coefficient set through the same filter the data saw: `out = A† W A alm`, with `W` the
per-point weight. This is the operator any comparison against partial observations must use on
the model side, so both sides carry identical coupling and nothing needs deconvolving.

Two transforms, expressed entirely in `Y_{ℓm}`, and the operator a third party needs to
reproduce a published comparison — apply it to their own field, then compare.
"""
function apply_resolution!(out, plan::Plan, alm, w::AbstractVector,
                           ws::AnalysisWorkspace = AnalysisWorkspace(plan))
    B = plan.ntrans
    f = ws.fbuf
    LinearAlgebra.mul!(f, plan, alm)
    _weight_field!(f, w, B)
    LinearAlgebra.mul!(out, LinearAlgebra.adjoint(plan), f)
    return out
end

function _weight_field!(f, w::AbstractVector, B::Int)
    length(w) == size(f, 1) ||
        throw(DimensionMismatch("weights have $(length(w)) entries, field has $(size(f,1))"))
    if ndims(f) == 1
        f .*= w
    else
        @inbounds for b in 1:size(f, 2)
            @views f[:, b] .*= w
        end
    end
    return f
end

# ---------------------------------------------------------------------------------------
# Mode coupling and band-powers
# ---------------------------------------------------------------------------------------

"""
    threej000_sq_log(ℓ1, ℓ2, ℓ3) -> Real

`log (ℓ1 ℓ2 ℓ3; 0 0 0)²`, or `-Inf` when the symbol vanishes. Closed form for the all-zero-order
Wigner 3-j symbol, through `loggamma` so nothing overflows:

    (ℓ1 ℓ2 ℓ3; 0 0 0) = (-1)^{J/2} √((J-2ℓ1)!(J-2ℓ2)!(J-2ℓ3)!/(J+1)!)
                        · (J/2)! / ((J/2-ℓ1)!(J/2-ℓ2)!(J/2-ℓ3)!)

with `J = ℓ1+ℓ2+ℓ3`; zero for odd `J` or outside the triangle inequality. Only the square is
needed, so the sign is dropped.
"""
function threej000_sq_log(ℓ1::Integer, ℓ2::Integer, ℓ3::Integer)
    a, b, c = Int(ℓ1), Int(ℓ2), Int(ℓ3)
    J = a + b + c
    isodd(J) && return -Inf
    (abs(a - b) <= c <= a + b) || return -Inf
    h = J ÷ 2
    lg = SpecialFunctions.loggamma
    t = (lg(J - 2a + 1.0) + lg(J - 2b + 1.0) + lg(J - 2c + 1.0) - lg(J + 2.0)) / 2 +
        lg(h + 1.0) - lg(h - a + 1.0) - lg(h - b + 1.0) - lg(h - c + 1.0)
    return 2t
end

"""
    mask_coupling!(M, Wℓ, lmax) -> M

MASTER mode-coupling matrix (Hivon et al., ApJ 567, 2002)

    M_{ℓℓ'} = (2ℓ'+1)/4π Σ_{ℓ''} (2ℓ''+1) W_{ℓ''} (ℓ ℓ' ℓ''; 0 0 0)²

from the mask's *own* power spectrum `Wℓ`. Nothing about the unobserved sky enters — the mask is
chosen, so `Wℓ` is known, which is why a band-power's bias is computable rather than merely
bounded. `O(L³)`, the same scaling every implementation of this kernel has; built once per mask.
"""
function mask_coupling!(M::AbstractMatrix, Wℓ::AbstractVector, lmax::Integer,
                        backend = ComputationalBackends.SerialBackend())
    L = Int(lmax)
    size(M) == (L + 1, L + 1) ||
        throw(DimensionMismatch("M is $(size(M)), expected $((L+1, L+1))"))
    T = eltype(M)
    fill!(M, zero(T))
    _mask_coupling_rows!(M, Wℓ, L, 0:L, backend)
    return M
end

# Rows of `M` are independent, so the threaded override in the OhMyThreads extension only has to
# hand disjoint row ranges to tasks. Everything shared is read-only.
function _mask_coupling_rows!(M::AbstractMatrix, Wℓ::AbstractVector, L::Int, rows, backend)
    _mask_coupling_rows!(M, Wℓ, L, rows)
    return M
end

function _mask_coupling_rows!(M::AbstractMatrix, Wℓ::AbstractVector, L::Int, rows)
    nW = length(Wℓ) - 1
    T = eltype(M)
    # `loggamma` of the same small integers is asked for `O(L³)` times by the closed form, so
    # tabulate it once and let the inner loop be arithmetic — that is what takes a plain
    # lmax = 2048 build from ~285 s down to seconds.
    #
    # The table must cover `log Γ(J+2)` for the largest triple: `ℓ, ℓ' ≤ L` and
    # `ℓ'' ≤ min(ℓ+ℓ', nW)`, so `J ≤ 2L + min(2L, nW)`. Sizing it from `L` alone is short whenever
    # the mask spectrum reaches past `ℓ = L`, and the read is `@inbounds`.
    jmax = 2L + min(2L, max(nW, 0))
    lgt = _loggamma_table(T, jmax + 2)
    @inbounds for ℓ in rows
        for ℓp in 0:L
            acc = zero(T)
            for ℓpp in abs(ℓ - ℓp):min(ℓ + ℓp, nW)
                w = Wℓ[ℓpp + 1]
                iszero(w) && continue
                lg = _threej000_sq_log_tab(lgt, ℓ, ℓp, ℓpp)
                isfinite(lg) || continue
                acc += T(2 * ℓpp + 1) * T(w) * exp(lg)
            end
            M[ℓ + 1, ℓp + 1] = T(2 * ℓp + 1) * acc / (4 * T(π))
        end
    end
    return M
end

"""`log Γ(k+1)` for `k = 0:kmax`, by the exact recurrence `log Γ(k+1) = log Γ(k) + log k`."""
function _loggamma_table(::Type{T}, kmax::Integer) where {T}
    n = Int(kmax)
    v = Vector{T}(undef, n + 1)
    v[1] = zero(T)
    @inbounds for k in 1:n
        v[k + 1] = v[k] + log(T(k))
    end
    return v
end

@inline function _threej000_sq_log_tab(lg::Vector{T}, a::Int, b::Int, c::Int) where {T}
    J = a + b + c
    isodd(J) && return T(-Inf)
    (abs(a - b) <= c <= a + b) || return T(-Inf)
    h = J ÷ 2
    @inbounds t = (lg[J - 2a + 1] + lg[J - 2b + 1] + lg[J - 2c + 1] - lg[J + 2]) / 2 +
                  lg[h + 1] - lg[h - a + 1] - lg[h - b + 1] - lg[h - c + 1]
    return 2t
end

"""
    mask_coupling(Wℓ, lmax, backend=SerialBackend()) -> Matrix

Allocating form of [`mask_coupling!`](@ref). `O(L³)` and depends only on the mask, so for repeated
[`spectrum`](@ref) calls on one region build it once and pass it as `coupling`.
"""
mask_coupling(Wℓ::AbstractVector, lmax::Integer,
              backend = ComputationalBackends.SerialBackend()) =
    mask_coupling!(zeros(float(eltype(Wℓ)), Int(lmax) + 1, Int(lmax) + 1), Wℓ, lmax, backend)

"""
    band_edges(lmax, Δℓ) -> Vector{UnitRange{Int}}

Degree bands of width `Δℓ`, the geometry's own resolution. Binning at this width is what makes
the coupling matrix invertible; finer bands ask for information the region does not carry.
"""
function band_edges(lmax::Integer, Δℓ::Real)
    L = Int(lmax)
    w = max(1, Int(round(Δℓ)))
    bands = UnitRange{Int}[]
    lo = 0
    while lo <= L
        push!(bands, lo:min(lo + w - 1, L))
        lo += w
    end
    return bands
end

"""
    bin_coupling(M, bands) -> Matrix

Bin `M` into band space with a `(2ℓ+1)` within-band weight, giving the square, invertible
`M_{bb'}` that band-power estimation uses.
"""
function bin_coupling(M::AbstractMatrix, bands::Vector{UnitRange{Int}})
    T = eltype(M)
    nb = length(bands)
    Mb = zeros(T, nb, nb)
    @inbounds for (i, bi) in enumerate(bands)
        wi = sum(T(2ℓ + 1) for ℓ in bi)
        for (j, bj) in enumerate(bands)
            acc = zero(T)
            for ℓ in bi, ℓp in bj
                acc += T(2ℓ + 1) * M[ℓ + 1, ℓp + 1]
            end
            Mb[i, j] = acc / wi
        end
    end
    return Mb
end

"""
    pseudo_spectrum!(C, plan, f, w, ws; batch=1) -> C

`C̃_ℓ = (1/(2ℓ+1)) Σ_m |ã_{ℓm}|²` from the windowed adjoint `ã = A†(w·f)`. Biased by
construction — [`spectrum`](@ref) removes the bias with [`mask_coupling!`](@ref).

With `ntrans > 1` the field carries several independent maps and `batch` selects which one's
spectrum is wanted; there is no meaningful single spectrum of a batch, so it is an argument rather
than a silent choice of the first member.
"""
function pseudo_spectrum!(C::AbstractVector, plan::Plan, f, w::AbstractVector,
                          ws::AnalysisWorkspace = AnalysisWorkspace(plan);
                          batch::Integer = 1)
    L = plan.lmax
    B = plan.ntrans
    length(C) == L + 1 || throw(DimensionMismatch("C has $(length(C)) entries, expected $(L+1)"))
    1 <= batch <= B || throw(ArgumentError("batch = $batch outside 1:$B"))
    alm = ws.alm
    fw = ws.fbuf
    copyto!(fw, f)
    _weight_field!(fw, w, B)
    LinearAlgebra.mul!(alm, LinearAlgebra.adjoint(plan), fw)
    T = eltype(C)
    b = Int(batch)
    @inbounds for ℓ in 0:L
        acc = zero(T)
        for m in -ℓ:ℓ
            acc += abs2(_coeff(alm, ℓ, m, b))
        end
        C[ℓ + 1] = acc / T(2ℓ + 1)
    end
    return C
end

"""
    spectrum(plan, f, w, Wℓ; Δℓ=nothing, deconvolve=true, ws=…) -> NamedTuple

Band-powers from data on a region, with the mask coupling accounted for.

Returns `bands`, `bandpowers`, `sigma` — the `√(2/((2ℓ+1) f_sky Δℓ))` scatter, which is the price
of partial sky: **variance, not bias** — the binned coupling `Mb`, and the raw pseudo-spectrum.

`deconvolve = false` leaves the estimate in pseudo space, which is the preferred comparison
direction: push a model spectrum *forward* through `Mb` rather than inverting it. Inversion is a
convenience for plotting in truth space, not a requirement.

`coupling` accepts a `M_{ℓℓ'}` from [`mask_coupling`](@ref). It is `O(L³)` and a function of the
mask alone, so anything looping over realisations or over batch members should build it once and
pass it here rather than have every call rebuild it. `backend` is used for that build.
"""
function spectrum(plan::Plan, f, w::AbstractVector, Wℓ::AbstractVector;
                  Δℓ = nothing, deconvolve::Bool = true, batch::Integer = 1,
                  coupling = nothing, backend = plan.backend,
                  ws::AnalysisWorkspace = AnalysisWorkspace(plan))
    L = plan.lmax
    box = ws.box
    dl = Δℓ === nothing ? spectral_resolution(box) : Δℓ
    bands = band_edges(L, dl)
    Ct = pseudo_spectrum!(ws.Cl, plan, f, w, ws; batch = batch)
    Mfull = coupling === nothing ? mask_coupling(Wℓ, L, backend) : coupling
    size(Mfull) == (L + 1, L + 1) || throw(DimensionMismatch(
        "coupling is $(size(Mfull)), expected $((L + 1, L + 1))"))
    Mb = bin_coupling(Mfull, bands)
    T = eltype(Ct)
    Cb = T[sum(T(2ℓ + 1) * Ct[ℓ + 1] for ℓ in b) / sum(T(2ℓ + 1) for ℓ in b) for b in bands]
    est = deconvolve ? Mb \ Cb : copy(Cb)
    fsky = box_fsky(box)
    σ = Vector{T}(undef, length(bands))
    @inbounds for (i, b) in enumerate(bands)
        ℓc = (first(b) + last(b)) / 2
        nmodes = max(one(T), T(2 * ℓc + 1) * T(length(b)) * T(fsky))
        σ[i] = sqrt(2 / nmodes) * abs(est[i])
    end
    return (bands = bands, bandpowers = est, sigma = σ, Mb = Mb, pseudo = copy(Ct),
            Δℓ = dl, fsky = fsky, deconvolved = deconvolve)
end

# ---------------------------------------------------------------------------------------
# Coefficient inversion, with the regularisation named
# ---------------------------------------------------------------------------------------

"""
    solve!(alm, plan, f; mode=:cg, rtol=1e-6, maxiter=500, prior=nothing, w=nothing, ws=…)
        -> (alm, info)

Coefficients from partial data. `A†A` is rank-deficient by construction, so there is no
regularisation-free answer and `mode` must be chosen, not defaulted past:

- `:cg` — conjugate gradients on the normal equations, stopped at `rtol`. The stopping rule *is*
  the regularisation, so `info.iters` is part of the answer.
- `:wiener` — Tikhonov with a stated `prior` (`C_ℓ`), i.e. the Wiener filter. What a CMB pipeline
  or an assimilation step means by "solve", and the right choice when a background exists.
- `:minnorm` — minimum-norm least squares. Optimal for anything evaluated *inside* the region;
  its field decays outside over the reproducing-kernel scale `~1/L`, so it is the wrong choice
  if the coefficients then feed a global quantity.

`info` carries `shannon`, `Δℓ`, `iters`, `relres` and `mode`, because a coefficient array from
partial data is not interpretable without them. Allocation-free after `ws` is built.
"""
function solve!(alm, plan::Plan, f; mode::Symbol = :cg, rtol::Real = 1e-6,
                maxiter::Integer = 500, prior = nothing, w = nothing,
                ws::AnalysisWorkspace = AnalysisWorkspace(plan))
    mode in (:cg, :wiener, :minnorm) ||
        throw(ArgumentError("mode must be :cg, :wiener or :minnorm, got :$mode"))
    mode === :wiener && prior === nothing &&
        throw(ArgumentError(":wiener needs a prior C_ℓ — that prior is what makes the " *
                            "problem well posed, so it cannot be defaulted."))
    B = plan.ntrans
    _assert_coeffs(alm, plan.lmax, B)
    _assert_field(f, length(plan.θ), B)

    CT = eltype(alm)
    T = real(CT)
    rhs = ws.rhs
    r = ws.r
    p = ws.p
    Ap = ws.Ap
    fbuf = ws.fbuf
    fw = ws.fbuf2

    copyto!(fw, f)
    w !== nothing && _weight_field!(fw, w, B)
    LinearAlgebra.mul!(rhs, LinearAlgebra.adjoint(plan), fw)

    reg = nothing
    if mode === :wiener
        _prior_inverse!(ws.priorinv, prior, plan.lmax)
        reg = ws.priorinv
    end
    tol = mode === :minnorm ? min(T(rtol), T(1e-12)) : T(rtol)

    fill!(alm, zero(CT))
    copyto!(r, rhs)
    copyto!(p, r)
    rsold = LinearAlgebra.dot(r, r)
    rhsn = sqrt(abs(LinearAlgebra.dot(rhs, rhs)))
    box = ws.box
    info0 = (shannon = shannon_number(plan.lmax, box_fsky(box)),
             Δℓ = spectral_resolution(box), mode = mode)
    rhsn == 0 && return alm, merge(info0, (iters = 0, relres = zero(T)))

    iters = 0
    rel = one(T)
    for i in 1:Int(maxiter)
        iters = i
        LinearAlgebra.mul!(fbuf, plan, p)
        w !== nothing && _weight_field!(fbuf, w, B)
        LinearAlgebra.mul!(Ap, LinearAlgebra.adjoint(plan), fbuf)
        reg !== nothing && _add_prior!(Ap, p, reg, plan.lmax, B)
        pAp = LinearAlgebra.dot(p, Ap)
        α = pAp == 0 ? zero(CT) : rsold / pAp
        alm .+= α .* p
        r .-= α .* Ap
        rsnew = LinearAlgebra.dot(r, r)
        rel = sqrt(abs(rsnew)) / rhsn
        rel <= tol && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return alm, merge(info0, (iters = iters, relres = rel))
end

function _prior_inverse!(out::AbstractVector{T}, prior, lmax::Integer) where {T}
    L = Int(lmax)
    length(prior) >= L + 1 ||
        throw(DimensionMismatch("prior must cover ℓ = 0:$L, got $(length(prior)) entries"))
    @inbounds for ℓ in 0:L
        c = T(prior[ℓ + 1])
        out[ℓ + 1] = c > 0 ? one(T) / c : zero(T)
    end
    return out
end

function _add_prior!(Ap, p, reg::AbstractVector{T}, lmax::Integer, B::Int) where {T}
    @inbounds for b in 1:B, ℓ in 0:Int(lmax)
        s = reg[ℓ + 1]
        iszero(s) && continue
        for m in -ℓ:ℓ
            _coeff_add!(Ap, ℓ, m, b, s * _coeff(p, ℓ, m, b))
        end
    end
    return Ap
end

"""
    synthesis_matrix(θ, φ, lmax) -> Matrix

Dense `𝒴_R`, columns in `(ℓ, m)` nested order. A diagnostic for small `lmax` only.

Cost is `O(M L²)`, which is optimal: that is the size of the output. Running one full transform per
column instead — a unit coefficient array through the ring engine, `(L+1)²` times — is `O(M L⁴)`,
measured with a local exponent of 3.96 in `lmax`, so it is worse than the `O(M L³)` of calling
`ylm` per entry. Here the degree recurrence runs *once* per `(point, order)` and every degree it
passes through is written straight into its column.
"""
function synthesis_matrix(θ::AbstractVector, φ::AbstractVector, lmax::Integer)
    L = Int(lmax)
    T = float(promote_type(float(eltype(θ)), float(eltype(φ))))
    CT = Complex{T}
    n = length(θ)
    n == length(φ) || throw(DimensionMismatch("θ and φ must have the same length"))
    θh = _host(θ)
    φh = _host(φ)
    A = zeros(CT, n, (L + 1)^2)
    # Column of (ℓ, m) in the nested order the docstring promises.
    col = (ℓ, m) -> ℓ * ℓ + ℓ + m + 1
    log2c = sectorial_log2c(T, L)
    @inbounds for i in 1:n
        t = T(θh[i])
        p = T(φh[i])
        s, x = sincos(t)
        as = abs(s)
        l2s = as > 0 ? log2(as) : T(-Inf)
        for m in 0:L
            e0 = m == 0 ? log2c[1] : T(m) * l2s + log2c[m + 1]
            isfinite(e0) || break
            E = ceil(e0)
            sgn = iseven(m) ? one(T) : -one(T)
            p2 = sgn * exp2(e0 - E)
            ph = cis(T(m) * p)
            _write_col!(A, i, col, m, m, p2, E, ph, sgn, CT)
            m == L && continue
            p1 = sqrt(T(2m + 3)) * x * p2
            _write_col!(A, i, col, m + 1, m, p1, E, ph, sgn, CT)
            steps = 0
            for ℓ in (m + 2):L
                a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
                b = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                         T(ℓ + m - 1) / T(ℓ + m))
                q = a * x * p1 - b * p2
                p2 = p1
                p1 = q
                steps += 1
                if steps == _RESCALE_STRIDE
                    steps = 0
                    mx = max(abs(p1), abs(p2))
                    if mx > 0 && isfinite(mx)
                        sh = ceil(log2(mx))
                        if sh != 0
                            fs = exp2(-sh)
                            p1 *= fs
                            p2 *= fs
                            E += sh
                        end
                    end
                end
                _write_col!(A, i, col, ℓ, m, p1, E, ph, sgn, CT)
            end
        end
    end
    return A
end

@inline function _write_col!(A, i::Int, col, ℓ::Int, m::Int, P, E, ph, sgn, ::Type{CT}) where {CT}
    v = P * exp2(E)
    @inbounds A[i, col(ℓ, m)] = CT(v) * ph
    # P̄_ℓ^{-m} = (-1)^m P̄_ℓ^m, and the azimuthal factor conjugates.
    m > 0 && (@inbounds A[i, col(ℓ, -m)] = CT(v * sgn) * conj(ph))
    return nothing
end
