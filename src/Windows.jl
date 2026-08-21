"""
Carron–Reinecke apodization: exponential-of-semicircle (ES) window and reduced bandlimit.

Formulas: Carron & Reinecke, arXiv:2603.17166, eqs. 16–22; ES kernel of
Barnett, Magland & af Klinteberg, SISC 41 (2019).
"""

"""
    es_alpha(ε) -> Real

ES shape parameter `α = |ln ε| / π`.
"""
es_alpha(ε::Real) = abs(log(float(ε))) / π

"""
    stretch_params(θstar, lmax, ε) -> NamedTuple

Apodization width, extended colatitude, and reduced Fourier-sphere bandlimit:

    εapo = √(2 |ln ε| / (θstar L))
    θmax = θstar (1 + εapo)
    Lmax = ⌊ L θstar/π (1 + εapo)² ⌋
"""
function stretch_params(θstar::Real, lmax::Integer, ε::Real)
    T = float(typeof(float(θstar)))
    θs = max(T(θstar), eps(T))
    L = max(Int(lmax), 1)
    εf = T(ε)
    εapo = sqrt(2 * abs(log(εf)) / (θs * T(L)))
    θmax = θs * (1 + εapo)
    θmax = min(θmax, T(π))
    Lmax = max(1, Int(floor(T(L) * θs / T(π) * (1 + εapo)^2)))
    return (εapo = εapo, θmax = θmax, Lmax = Lmax, alpha = es_alpha(εf))
end

"""
    es_bell(z, α)

ES kernel on `z ∈ [-1, 1]`: `exp(απ (√(1-z²) - 1))`, and 0 outside.
"""
function es_bell(z::Real, α::Real)
    T = float(typeof(float(z)))
    az = abs(T(z))
    az >= one(T) && return zero(T)
    return exp(T(α) * T(π) * (sqrt(1 - az * az) - 1))
end

"""
    stretch_window!(w, Θ, θstar, θmax, ε) -> w

ES kernel convolved with a top-hat so that `w ≡ 1` on `Θ ≤ π θstar/θmax` (the
image of the user cap) and `w = 0` at `Θ = π`. `Θ` is the stretched colatitude
in `[0, π]`.
"""
function stretch_window!(w::AbstractVector{T}, Θ::AbstractVector, θstar::Real,
                         θmax::Real, ε::Real) where {T<:AbstractFloat}
    n = length(Θ)
    length(w) == n || throw(DimensionMismatch("w and Θ must have the same length"))
    α = T(es_alpha(ε))
    Θ1 = T(π) * T(θstar) / T(θmax)           # user-cap image
    σ = (T(π) - Θ1) / 2                       # ES half-width so support ends at π
    σ = max(σ, sqrt(eps(T)))
    ΘT = Θ1 + σ                               # top-hat right edge
    invσ = inv(σ)
    # Discrete convolution of Π_[0,ΘT] with a normalized ES kernel of half-width σ,
    # plus even reflection at Θ = 0 so the pole stays in the passband.
    nq = 64
    @inbounds for i in 1:n
        Θi = T(Θ[i])
        acc = zero(T)
        mass = zero(T)
        for q in 1:nq
            # kernel sample z ∈ (-1, 1)
            z = (2q - nq - 1) / T(nq)
            kz = es_bell(z, α)
            t = Θi - z * σ
            mass += kz
            inside = (t >= 0 && t <= ΘT) || (-t >= 0 && -t <= ΘT)
            inside && (acc += kz)
        end
        w[i] = mass == 0 ? zero(T) : acc / mass
    end
    # Force the contract on the user-cap image and at Θ = π.
    @inbounds for i in 1:n
        Θi = T(Θ[i])
        if Θi <= Θ1
            w[i] = one(T)
        elseif Θi >= T(π) - 10 * eps(T)
            w[i] = zero(T)
        end
    end
    return w
end

function stretch_window(Θ::AbstractVector, θstar::Real, θmax::Real, ε::Real)
    w = similar(Θ, float(eltype(Θ)))
    return stretch_window!(w, Θ, θstar, θmax, ε)
end
