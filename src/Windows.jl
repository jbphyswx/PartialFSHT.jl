# Apodisation and the stretched bandlimit.
#
# A degree-lmax field on a colatitude interval is periodised by multiplying with a window that is
# identically one on the data and falls to zero at the period boundary: an exponential-of-semicircle
# bell convolved with a top-hat (Barnett, Magland & af Klinteberg, SISC 41, 2019).
#
# One derived relation fixes the whole construction,
#
#     εapo = √(2β / (lmax·h)),        K = lmax·h(1 + εapo)² / π,       β = |ln ε|
#
# and it is only meaningful for εapo < 1. Past that the padding is wider than the data and the
# construction is refused rather than silently clamped. The derivation is in the manual under
# Theory ("Apodisation and the stretched bandlimit").

# Largest apodisation fraction the construction is used at. `εapo ≥ 1` means the padding is
# wider than the data, at which point the interval balloons and a direct evaluation is the
# right tool. Measured on a 20° cap at lmax = 256: delivered accuracy stays at ≤ 0.1·tol
# across εapo ∈ [0.41, 1.02], so the bound is the derivation's own (padding narrower than the
# data) rather than an accuracy cliff. Overridable per call for calibration sweeps.
const _EPS_APO_MAX = 1.0

"""
    es_beta(ε) -> Real

ES shape parameter `β = |ln ε|`; the kernel's spectral residual is `~e^{-β}`.
"""
es_beta(ε::Real) = abs(log(float(ε)))

"""
    StretchGeometry{T}

Periodisation of a colatitude interval, and the bandlimit it supports.

- `kind` — `:pole` (interval touches a pole; the DFS reflection supplies smoothness there and
  only the far end is apodised), `:interior` (both ends apodised), `:full` (`[0,π]`, DFS at both
  poles, no apodisation)
- `h` — half-extent of the *data* in colatitude
- `period` — padded period in colatitude, `2h(1+εapo)`
- `K` — retained bandlimit on the stretched circle
- `Θ1`, `σ`, `ΘT` — data image, ES half-width, top-hat half-width, in stretched radians
"""
struct StretchGeometry{T<:AbstractFloat}
    kind::Symbol
    lmax::Int
    ε::T
    β::T
    h::T
    θlo::T
    θhi::T
    εapo::T
    period::T
    K::Int
    Θ1::T
    σ::T
    ΘT::T
    invZ::T   # 1/∫ES, hoisted so the window is not renormalised per sample
end

"""
    stretch_geometry(θlo, θhi, lmax, ε) -> StretchGeometry

Periodisation for data occupying `θ ∈ [θlo, θhi]`. Picks `:full`, `:pole` or `:interior` from
the endpoints and applies the derived `εapo`/`K` relation. Throws when `lmax·h ≤ 2|ln ε|`,
i.e. when apodisation would be wider than the data.
"""
function stretch_geometry(θlo::Real, θhi::Real, lmax::Integer, ε::Real;
                          εapo_max::Real = _EPS_APO_MAX)
    g = try_stretch_geometry(θlo, θhi, lmax, ε; εapo_max = εapo_max)
    g === nothing && throw(ArgumentError(
        "stretched transform is degenerate for θ ∈ [$θlo, $θhi] at lmax = $lmax, tol = $ε: every " *
        "periodisation needs εapo > $(εapo_max), i.e. an apodisation comparable to the " *
        "data extent, where this construction stops delivering the requested tolerance. Use a " *
        "direct evaluation for a patch this small at this bandlimit, or raise lmax / loosen tol."))
    return g
end

"""
    try_stretch_geometry(θlo, θhi, lmax, ε; εapo_max) -> StretchGeometry or nothing

[`stretch_geometry`](@ref) as a query: `nothing` where that would throw. The cost model needs to
ask whether a stretched leaf is admissible for a candidate box without treating "not admissible"
as an error, and without a `catch` that would also swallow a genuinely bad argument.
"""
function try_stretch_geometry(θlo::Real, θhi::Real, lmax::Integer, ε::Real;
                              εapo_max::Real = _EPS_APO_MAX)
    T = float(promote_type(typeof(float(θlo)), typeof(float(θhi))))
    L = Int(lmax)
    lo, hi = T(θlo), T(θhi)
    lo <= hi || throw(ArgumentError("θlo = $lo exceeds θhi = $hi"))
    (lo >= 0 && hi <= T(π)) || throw(ArgumentError("colatitudes must lie in [0, π]"))
    β = T(es_beta(ε))
    tiny = sqrt(eps(T))
    # `Z = ∫ES` depends only on β, so it is formed once rather than per surviving candidate.
    Z = es_cdf(β, T(π) / 2)
    invZ = Z > 0 ? one(T) / Z : zero(T)

    if lo <= tiny && hi >= T(π) - tiny
        return StretchGeometry{T}(:full, L, T(ε), β, T(π) / 2, zero(T), T(π),
                                  zero(T), 2 * T(π), L, T(π), zero(T), T(π), one(T))
    end

    # Three admissible periodisations, all of which contain the data:
    #   :pole north — extend the interval to [0, θhi] and reflect through the north pole, so the
    #                 half-extent is θhi and only the far end needs apodising;
    #   :pole south — likewise [θlo, π];
    #   :interior   — apodise both ends, half-extent (θhi-θlo)/2.
    # Pick by bandlimit rather than by a tolerance on "is θlo close enough to zero". A cap
    # sampled *near* a pole has a small interior half-extent and would look degenerate, while
    # its pole treatment is both valid and cheap; a mid-latitude patch is the other way round.
    cands = ((:pole, hi, zero(T), hi),
             (:pole, T(π) - lo, lo, T(π)),
             (:interior, (hi - lo) / 2, lo, hi))
    bestK = typemax(Int)
    best = nothing
    for (kind, h, klo, khi) in cands
        h > 0 || continue
        Lh = T(L) * h
        Lh >= 2 * β / T(εapo_max)^2 || continue
        εapo = sqrt(2 * β / Lh)
        # The padded interval has to stay inside [0, π]. An interior periodisation whose
        # apodisation margin runs past a pole is not merely loose, it is wrong: colatitude
        # stops being a coordinate there and smoothness across the pole needs the DFS
        # reflection, not a window. Letting such a candidate through caps accuracy at ~1e-3
        # no matter what tolerance is asked for.
        pad = h * εapo
        if kind === :interior
            (klo - pad >= 0 && khi + pad <= T(π)) || continue
        else
            h * (1 + εapo) <= T(π) || continue
        end
        K = max(1, Int(ceil(Lh * (1 + εapo)^2 / T(π))))
        if K < bestK
            bestK = K
            Θ1 = T(π) / (1 + εapo)
            σ = (T(π) - Θ1) / 2
            best = StretchGeometry{T}(kind, L, T(ε), β, h, klo, khi, εapo,
                                      2 * h * (1 + εapo), K, Θ1, σ, Θ1 + σ, invZ)
        end
    end
    return best
end

"""
    stretch_geometry_phi(φextent, lmax, ε) -> StretchGeometry

Periodisation of a longitude arc. Longitude is already periodic, so there is no pole case: an
arc narrower than `2π` is apodised from both sides (`:interior`), and a full circle needs no
window at all (`:full`, `K = lmax`).

The same derivation applies verbatim with `h = φextent/2`, because the only inputs were the
half-extent, the frequency of `e^{i·freq·coordinate}`, and the tolerance — none of which care
which angle is being stretched. A narrow arc is exactly where the saving is largest, and it is
why a patch away from a pole is still cheap without a coefficient rotation.
"""
function stretch_geometry_phi(φextent::Real, lmax::Integer, ε::Real)
    T = float(typeof(float(φextent)))
    L = Int(lmax)
    ext = T(φextent)
    β = T(es_beta(ε))
    if ext >= 2 * T(π) - sqrt(eps(T))
        return StretchGeometry{T}(:full, L, T(ε), β, T(π), zero(T), 2 * T(π),
                                  zero(T), 2 * T(π), L, T(π), zero(T), T(π), one(T))
    end
    ext > 0 || throw(ArgumentError("longitude extent must be positive"))
    h = ext / 2
    Lh = T(L) * h
    Lh > 2 * β || throw(ArgumentError(
        "longitude stretch is degenerate here: lmax*φextent/2 = $(Lh) ≤ 2|ln ε| = $(2β). " *
        "Keep the full longitude range for an arc this narrow at this bandlimit."))
    εapo = sqrt(2 * β / Lh)
    period = 2 * h * (1 + εapo)
    period = min(period, 2 * T(π))
    K = max(1, Int(ceil(Lh * (1 + εapo)^2 / T(π))))
    K = min(K, L)
    Θ1 = T(π) / (1 + εapo)
    σ = (T(π) - Θ1) / 2
    Z = es_cdf(β, T(π) / 2)
    return StretchGeometry{T}(:interior, L, T(ε), β, h, zero(T), ext, εapo, period, K,
                              Θ1, σ, Θ1 + σ, Z > 0 ? one(T) / Z : zero(T))
end

# ---------------------------------------------------------------------------------------
# ES cumulative distribution, converged
# ---------------------------------------------------------------------------------------

# 10-point Gauss–Legendre on [-1, 1].
const _GL10_X = (-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                 -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                 0.4333953941292472, 0.6794095682990244, 0.8650633666889845,
                 0.9739065285171717)
const _GL10_W = (0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                 0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                 0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                 0.0666713443086881)

# Target panel width for the composite rule, in the substituted variable.
const _ES_PANEL = 0.05

"""
    es_cdf(β, u) -> T

`∫_{-π/2}^{u} exp(β(cos t - 1)) cos t dt`, the ES kernel's cumulative distribution after the
substitution `z = sin t`. The substitution removes the square-root branch point the kernel has
at `|z| = 1`, leaving an analytic integrand, so composite Gauss–Legendre converges to round-off
instead of the `O(n^{-3/2})` a midpoint rule on `z` would give. (A 64-point midpoint rule on
`z` is off by ~1% — enough to make the window a different function from the one documented.)
"""
function es_cdf(β::T, u::T) where {T<:AbstractFloat}
    a = -T(π) / 2
    u <= a && return zero(T)
    b = min(u, T(π) / 2)
    n = max(1, Int(ceil((b - a) / T(_ES_PANEL))))
    hw = (b - a) / (2 * n)
    acc = zero(T)
    @inbounds for p in 0:(n - 1)
        c = a + (2 * p + 1) * hw
        s = zero(T)
        for q in 1:10
            t = c + hw * T(_GL10_X[q])
            ct = cos(t)
            s += T(_GL10_W[q]) * exp(β * (ct - 1)) * ct
        end
        acc += hw * s
    end
    return acc
end

"""
    stretch_window(geom, Θ) -> T

Window value at stretched colatitude `Θ`. Exactly one where the data lives, exactly zero at the
period boundary, and `C^∞` between: the ES bell convolved with a top-hat, evaluated from
[`es_cdf`](@ref). For `:pole` geometries the distance is measured from the pole (the DFS
reflection makes the window automatically even there); for `:interior` it is measured from the
centre of the period.
"""
function stretch_window(geom::StretchGeometry{T}, Θ::Real) where {T}
    geom.kind === :full && return one(T)
    d = geom.kind === :pole ? abs(T(Θ)) : abs(T(Θ) - T(π))
    d <= geom.Θ1 && return one(T)
    d >= T(π) && return zero(T)
    σ = geom.σ
    σ > 0 || return d <= geom.ΘT ? one(T) : zero(T)
    zlo = clamp((d - geom.ΘT) / σ, -one(T), one(T))
    zhi = clamp((d + geom.ΘT) / σ, -one(T), one(T))
    ulo = asin(zlo)
    uhi = asin(zhi)
    v = (es_cdf(geom.β, uhi) - es_cdf(geom.β, ulo)) * geom.invZ
    return clamp(v, zero(T), one(T))
end

"""
    stretch_window!(w, geom, Θ) -> w

Sample [`stretch_window`](@ref) on a grid of stretched colatitudes.
"""
function stretch_window!(w::AbstractVector, geom::StretchGeometry{T},
                         Θ::AbstractVector) where {T}
    length(w) == length(Θ) ||
        throw(DimensionMismatch("w has $(length(w)) entries, Θ has $(length(Θ))"))
    @inbounds for i in eachindex(Θ)
        w[i] = stretch_window(geom, Θ[i])
    end
    return w
end

"""
    stretch_to_theta(geom, Θ) -> θ

Stretched colatitude back to physical colatitude.
"""
function stretch_to_theta(geom::StretchGeometry{T}, Θ::Real) where {T}
    geom.kind === :full && return T(Θ)
    if geom.kind === :pole
        # Θ ∈ [0, π] maps onto the padded interval measured from whichever pole is touched.
        θmax = geom.h * (1 + geom.εapo)
        t = T(Θ) * θmax / T(π)
        return geom.θlo <= sqrt(eps(T)) ? t : T(π) - t
    end
    lo = geom.θlo - geom.h * geom.εapo
    return lo + geom.period * T(Θ) / (2 * T(π))
end

"""
    theta_to_stretch(geom, θ) -> Θ

Physical colatitude to stretched colatitude; inverse of [`stretch_to_theta`](@ref).
"""
function theta_to_stretch(geom::StretchGeometry{T}, θ::Real) where {T}
    geom.kind === :full && return T(θ)
    if geom.kind === :pole
        θmax = geom.h * (1 + geom.εapo)
        d = geom.θlo <= sqrt(eps(T)) ? T(θ) : T(π) - T(θ)
        return T(π) * d / θmax
    end
    lo = geom.θlo - geom.h * geom.εapo
    return 2 * T(π) * (T(θ) - lo) / geom.period
end

"""
    stretch_rings(geom, nθ) -> Vector

Physical colatitudes of the `nθ` rings the periodisation samples. `:pole` and `:full`
geometries put rings on the half-period `(0, π)` in `Θ` and rely on the DFS reflection for the
other half; `:interior` geometries sample the whole period.
"""
function stretch_rings(geom::StretchGeometry{T}, nθ::Integer) where {T}
    n = Int(nθ)
    n >= 1 || throw(ArgumentError("need at least one ring"))
    if geom.kind === :interior
        Θ = T[2 * T(π) * (i - 1) / n for i in 1:n]
    else
        Θ = T[T(π) * (i - T(0.5)) / n for i in 1:n]
    end
    θ = similar(Θ)
    @inbounds for i in 1:n
        θ[i] = stretch_to_theta(geom, Θ[i])
    end
    return θ, Θ
end
