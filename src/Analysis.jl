"""
What partial data can actually yield.

Three things live here, in the order a user should reach for them:

1. [`recoverability`](@ref) — what this geometry supports, before any compute is spent.
2. [`spectrum!`](@ref) — band-powers with the mask's own coupling accounted for. Well posed at
   any sky fraction, and the right answer to "which wavenumbers are doing what".
3. [`solve!`](@ref) — coefficients, with the regularisation named rather than implied.

The distinction that matters: `A` and `A†` are exact at any coverage, but `A†f` is *not* the
field's coefficients — it is the masked field's, mode-coupled and leakage-biased. Recovering
coefficients from a region is rank-deficient by construction (only `~N_S = (L+1)² f_sky`
directions are constrained), so every routine here reports what it did rather than returning
numbers that look like truth.

Everything is bang-first against an explicit workspace: an iterative solve runs hundreds of
applies, so allocating five coefficient arrays per call — or a field temporary per matvec —
would dominate the thing it is measuring.
"""

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
function mask_coupling!(M::AbstractMatrix, Wℓ::AbstractVector, lmax::Integer)
    L = Int(lmax)
    size(M) == (L + 1, L + 1) ||
        throw(DimensionMismatch("M is $(size(M)), expected $((L+1, L+1))"))
    nW = length(Wℓ) - 1
    T = eltype(M)
    fill!(M, zero(T))
    @inbounds for ℓ in 0:L, ℓp in 0:L
        acc = zero(T)
        for ℓpp in abs(ℓ - ℓp):min(ℓ + ℓp, nW)
            w = Wℓ[ℓpp + 1]
            iszero(w) && continue
            lg = threej000_sq_log(ℓ, ℓp, ℓpp)
            isfinite(lg) || continue
            acc += T(2 * ℓpp + 1) * T(w) * exp(T(lg))
        end
        M[ℓ + 1, ℓp + 1] = T(2 * ℓp + 1) * acc / (4 * T(π))
    end
    return M
end

"""
    mask_coupling(Wℓ, lmax) -> Matrix

Allocating form of [`mask_coupling!`](@ref).
"""
mask_coupling(Wℓ::AbstractVector, lmax::Integer) =
    mask_coupling!(zeros(float(eltype(Wℓ)), Int(lmax) + 1, Int(lmax) + 1), Wℓ, lmax)

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
    pseudo_spectrum!(C, plan, f, w, ws) -> C

`C̃_ℓ = (1/(2ℓ+1)) Σ_m |ã_{ℓm}|²` from the windowed adjoint `ã = A†(w·f)`. Biased by
construction — [`spectrum`](@ref) removes the bias with [`mask_coupling!`](@ref).
"""
function pseudo_spectrum!(C::AbstractVector, plan::Plan, f, w::AbstractVector,
                          ws::AnalysisWorkspace = AnalysisWorkspace(plan))
    L = plan.lmax
    B = plan.ntrans
    length(C) == L + 1 || throw(DimensionMismatch("C has $(length(C)) entries, expected $(L+1)"))
    alm = ws.alm
    fw = ws.fbuf
    copyto!(fw, f)
    _weight_field!(fw, w, B)
    LinearAlgebra.mul!(alm, LinearAlgebra.adjoint(plan), fw)
    T = eltype(C)
    @inbounds for ℓ in 0:L
        acc = zero(T)
        for m in -ℓ:ℓ
            acc += abs2(_coeff(alm, ℓ, m, 1))
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
"""
function spectrum(plan::Plan, f, w::AbstractVector, Wℓ::AbstractVector;
                  Δℓ = nothing, deconvolve::Bool = true,
                  ws::AnalysisWorkspace = AnalysisWorkspace(plan))
    L = plan.lmax
    box = ws.box
    dl = Δℓ === nothing ? spectral_resolution(box) : Δℓ
    bands = band_edges(L, dl)
    Ct = pseudo_spectrum!(ws.Cl, plan, f, w, ws)
    Mb = bin_coupling(mask_coupling(Wℓ, L), bands)
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

Dense `𝒴_R`, columns in `(ℓ, m)` nested order. A diagnostic for small `lmax` only, but built
through the ring engine rather than by calling `ylm` per entry, so it costs `O(M L²)` instead of
the `O(M L³)` the previous version spent re-running the recurrence for every coefficient.
"""
function synthesis_matrix(θ::AbstractVector, φ::AbstractVector, lmax::Integer)
    L = Int(lmax)
    CT = Complex{float(eltype(θ))}
    n = length(θ)
    leaf = DirectLeaf(θ, φ, L, 1)
    A = Matrix{CT}(undef, n, (L + 1)^2)
    alm = zeros(CT, coeff_size(L)...)
    f = Vector{CT}(undef, n)
    k = 1
    for ℓ in 0:L, m in -ℓ:ℓ
        fill!(alm, zero(CT))
        alm[sph_mode(ℓ, m)] = one(CT)
        synthesize_direct!(f, alm, leaf)
        @views A[:, k] .= f
        k += 1
    end
    return A
end
