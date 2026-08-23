# Conversion between this package's complex orthonormal Condon–Shortley coefficients a_{ℓm} and
# FastTransforms / FastSphericalHarmonics *real* packing c_{ℓm} (NUFSHT's real-field
# specialisation). Derived so that Σ_m a_m Y_ℓm = Σ_m c_m Y_ℓm^FT, with the pairing
#
#     c_0    = a_0
#     c_m    = (-1)^m / √2 (a_m + (-1)^m a_{-m})              (m > 0)
#     c_{-m} = (-1)^m / √2 ⋅ i (a_m - (-1)^m a_{-m})
#
# and inverse
#
#     a_m    = √2/2 ⋅ (-1)^m (c_m - i c_{-m})
#     a_{-m} = √2/2 ⋅ (c_m + i c_{-m})

function complex_to_fasttransforms!(c, a, lmax::Integer)
    B = _ntrans(a)
    fill!(c, zero(eltype(c)))
    invs2 = inv(sqrt(real(eltype(a))(2)))
    @inbounds for b in 1:B, ℓ in 0:Int(lmax)
        _coeff_set!(c, ℓ, 0, b, _coeff(a, ℓ, 0, b))
        for m in 1:ℓ
            am = _coeff(a, ℓ, m, b)
            anm = _coeff(a, ℓ, -m, b)
            sgn = iseven(m) ? 1 : -1
            _coeff_set!(c, ℓ, m, b, sgn * invs2 * (am + sgn * anm))
            _coeff_set!(c, ℓ, -m, b, sgn * invs2 * im * (am - sgn * anm))
        end
    end
    return c
end

function fasttransforms_to_complex!(a, c, lmax::Integer)
    B = _ntrans(c)
    fill!(a, zero(eltype(a)))
    s2h = sqrt(real(eltype(c))(2)) / 2
    @inbounds for b in 1:B, ℓ in 0:Int(lmax)
        _coeff_set!(a, ℓ, 0, b, _coeff(c, ℓ, 0, b))
        for m in 1:ℓ
            cm = _coeff(c, ℓ, m, b)
            cnm = _coeff(c, ℓ, -m, b)
            sgn = iseven(m) ? 1 : -1
            _coeff_set!(a, ℓ, m, b, s2h * sgn * (cm - im * cnm))
            _coeff_set!(a, ℓ, -m, b, s2h * (cm + im * cnm))
        end
    end
    return a
end
