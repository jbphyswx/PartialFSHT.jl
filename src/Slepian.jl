"""
Slepian / concentration diagnostics. These are not the public representation of
the transform: `Plan` applies `𝒴_R`, and `solve!` already lives in the numerically
well-posed subspace of `𝒴'𝒴`.
"""

"""
    shannon_number(lmax, θstar) -> Real

Shannon number of a polar cap of colatitude `θstar` at bandlimit `lmax`:
`(lmax+1)² (1 - cos θstar) / 2`.
"""
shannon_number(lmax::Integer, θstar::Real) =
    (Int(lmax) + 1)^2 * (1 - cos(θstar)) / 2

"""
    synthesis_matrix(θ, φ, lmax) -> Matrix

Dense `𝒴_R` with columns `Y_{ℓm}` in `(ℓ, m)` nested order, `m = -ℓ:ℓ`. Host
`Array` diagnostic; not used by `Plan`.
"""
function synthesis_matrix(θ::AbstractVector, φ::AbstractVector, lmax::Integer)
    θh, φh = _host(θ), _host(φ)
    T = float(eltype(θ))
    CT = Complex{T}
    n = length(θh)
    cols = (Int(lmax) + 1)^2
    A = Matrix{CT}(undef, n, cols)
    k = 1
    @inbounds for ℓ in 0:Int(lmax), m in -ℓ:ℓ
        for i in 1:n
            A[i, k] = ylm(ℓ, m, θh[i], φh[i])
        end
        k += 1
    end
    return A
end

"""
    weighted_synthesis_matrix(θ, φ, lmax, w) -> Matrix

`W^{1/2} 𝒴_R` with diagonal quadrature weights `w`.
"""
function weighted_synthesis_matrix(θ, φ, lmax, w)
    A = synthesis_matrix(θ, φ, lmax)
    @inbounds for i in axes(A, 1)
        s = sqrt(w[i])
        A[i, :] .*= s
    end
    return A
end

"""
    cap_quadrature_weights(θ, φ, θstar, center) -> Vector

Naive spherical Voronoi-free weights: uniform `Ω_cap / n` on points inside the
cap (diagnostic only).
"""
function cap_quadrature_weights(θ::AbstractVector, φ::AbstractVector, cap::BoundingCap)
    n = length(θ)
    T = typeof(cap.θstar)
    area = 2T(π) * (1 - cos(cap.θstar))
    return fill(area / n, n)
end
