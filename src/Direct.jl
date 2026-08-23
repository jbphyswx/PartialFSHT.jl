# Direct evaluation — the correctness oracle. O(M L²) and intentionally so: this is what every fast
# path is gated against, so it must be right at every degree rather than fast.
#
# It reuses the *same* block-scaled recurrence as the production ring engine with polar optimisation
# switched off, so there is exactly one associated-Legendre implementation in the package and the
# oracle cannot drift from the thing it certifies. Treating each sample point as its own ring turns
# the oracle into a ring transform plus a per-point azimuthal phase, so no separate kernel is needed.

"""
    DirectLeaf

Scratch for [`synthesize_direct!`](@ref) / [`adjoint_direct!`](@ref). Points are sorted by
colatitude at construction (the ring engine walks ascending `θ`) and scattered back on output.
"""
struct DirectLeaf{T, CT, EN, A3, VT, VI}
    lmax::Int
    ntrans::Int
    eng::EN
    G::A3
    perm::VI
    θs::VT
    φs::VT
end

function DirectLeaf(θ::AbstractVector, φ::AbstractVector, lmax::Integer, ntrans::Integer)
    T = float(promote_type(float(eltype(θ)), float(eltype(φ))))
    CT = Complex{T}
    L = Int(lmax)
    B = Int(ntrans)
    length(θ) == length(φ) || throw(DimensionMismatch("θ and φ must have the same length"))
    θh = _host(θ)
    φh = _host(φ)
    n = length(θh)
    perm = sortperm(θh)
    θs_h = Vector{T}(undef, n)
    φs_h = Vector{T}(undef, n)
    @inbounds for i in 1:n
        p = perm[i]
        θs_h[i] = T(θh[p])
        φs_h[i] = T(φh[p])
    end

    # ε = 0 asks polar_mmax for no truncation at all: the oracle keeps every order.
    eng = RingLegendre(θs_h, L, zero(T); ntrans = B, CT = CT)

    G = _zeros_like(θ, CT, n, 2L + 1, B)
    θs = _to_like(θ, θs_h)
    φs = _to_like(θ, φs_h)
    return DirectLeaf{T, CT, typeof(eng), typeof(G), typeof(θs), typeof(perm)}(
        L, B, eng, G, perm, θs, φs)
end

algorithm(::DirectLeaf) = DirectAlgorithm()

"""
    synthesize_direct!(f, alm, leaf) -> f

`f_i = Σ_{ℓm} a_{ℓm} Y_{ℓm}(Ω_i)`, exactly. Bang-first.
"""
function synthesize_direct!(f, alm, leaf::DirectLeaf{T, CT}) where {T, CT}
    B = leaf.ntrans
    L = leaf.lmax
    _assert_coeffs(alm, L, B)
    _assert_field(f, length(leaf.θs), B)
    rings_from_alm!(leaf.G, alm, leaf.eng)
    n = length(leaf.θs)
    off = L + 1
    fill!(f, zero(eltype(f)))
    # The azimuthal phase is formed exactly per order rather than as a running product. A running
    # `ph *= cis(φ)` drifts by `O(L·eps)` in both argument and modulus — 1.3e-13 at lmax = 4096 —
    # and this is the reference the fast paths are asserted against at round-off level, so it
    # cannot carry an error that grows with the bandlimit. One `sincos` per (point, order) is ~1%
    # of an `O(M L²)` oracle.
    @inbounds for b in 1:B
        for i in 1:n
            acc = leaf.G[i, off, b]
            φi = leaf.φs[i]
            for m in 1:L
                ph = cis(T(m) * φi)
                acc += leaf.G[i, m + off, b] * ph + leaf.G[i, -m + off, b] * conj(ph)
            end
            _field_set!(f, leaf.perm[i], b, acc)
        end
    end
    return f
end

"""
    adjoint_direct!(alm, f, leaf) -> alm

`(𝒴*f)_{ℓm} = Σ_i conj(Y_{ℓm}(Ω_i)) f_i`, the exact Euclidean adjoint — not a quadrature
inverse.
"""
function adjoint_direct!(alm, f, leaf::DirectLeaf{T, CT}) where {T, CT}
    B = leaf.ntrans
    L = leaf.lmax
    _assert_coeffs(alm, L, B)
    _assert_field(f, length(leaf.θs), B)
    n = length(leaf.θs)
    off = L + 1
    fill!(leaf.G, zero(CT))
    @inbounds for b in 1:B
        for i in 1:n
            fi = _field_get(f, leaf.perm[i], b)
            iszero(fi) && continue
            φi = leaf.φs[i]
            leaf.G[i, off, b] = fi
            for m in 1:L
                ph = cis(T(m) * φi)
                leaf.G[i, m + off, b] = fi * conj(ph)
                leaf.G[i, -m + off, b] = fi * ph
            end
        end
    end
    alm_from_rings!(alm, leaf.G, leaf.eng)
    return alm
end

@inline function _field_get(f, i::Int, b::Int)
    return ndims(f) == 1 ? f[i] : f[i, b]
end

@inline function _field_set!(f, i::Int, b::Int, v)
    if ndims(f) == 1
        f[i] = v
    else
        f[i, b] = v
    end
    return nothing
end
