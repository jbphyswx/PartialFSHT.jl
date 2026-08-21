"""
Polar optimization: associated Legendres on a ring of colatitude `θ` are
exponentially small for `|m| ≳ ℓ sinθ` (Reinecke & Seljebotn, A&A 554 A112, 2013).
"""

"""
    polar_mmax(θ, ℓ, ε) -> Int

Largest order `|m|` that needs to be retained at colatitude `θ` and degree `ℓ`
for relative tolerance `ε`.
"""
function polar_mmax(θ::Real, ℓ::Integer, ε::Real)
    T = float(typeof(float(θ)))
    L = Int(ℓ)
    L <= 0 && return 0
    s = abs(sin(T(θ)))
    pad = T(0.5) * abs(log(T(ε))) / log(T(2)) + T(8)
    return clamp(Int(ceil(L * s + pad)), 0, L)
end

"""
    polar_mmax_lmax(θ, lmax, ε) -> Int

`max_ℓ polar_mmax(θ, ℓ, ε)`, i.e. the m-limit for a degree-`lmax` ring SHT.
"""
polar_mmax_lmax(θ::Real, lmax::Integer, ε::Real) = polar_mmax(θ, lmax, ε)
