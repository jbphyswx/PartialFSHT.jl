"""
Spin-weighted spherical harmonics by the Wigner-`d` definition

    ₛY_{ℓm}(θ,φ) = √((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}

used as the Direct-path oracle. Stretch / cluster apply the same Wigner-`d` ring SHT
after rotating the bounding cap to the pole; DFS doubling picks up `(-1)^s`.
`NUFSHTAlgorithm` calls NUFSHT's public spin plan (`make_spin_plan`) when that
extension is loaded.
"""

"""
    sYlm(s, ℓ, m, θ, φ) -> Complex

Spin-weighted spherical harmonic `ₛY_{ℓm}(θ,φ)`.
"""
function sYlm(s::Integer, ℓ::Integer, m::Integer, θ::Real, φ::Real)
    abs(s) > ℓ && return zero(Complex{typeof(float(θ))})
    return _Nℓ(ℓ) * wigner_d(ℓ, m, -s, θ) * cis(m * φ)
end

function synthesize_spin_direct!(f, alm, plan::Plan)
    s = plan.spin
    lmax = plan.lmax
    B = plan.ntrans
    θ = plan.θ
    φ = plan.φ
    n = length(θ)
    fill!(f, zero(eltype(f)))
    @inbounds for b in 1:B, i in 1:n
        acc = zero(eltype(f))
        ti = θ[i]; pi = φ[i]
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            acc += _spin_coeff(alm, ℓ, m, lmax, b) * sYlm(s, ℓ, m, ti, pi)
        end
        if ndims(f) == 1
            f[i] = acc
        else
            f[i, b] = acc
        end
    end
    return f
end

function adjoint_spin_direct!(alm, f, plan::Plan)
    s = plan.spin
    lmax = plan.lmax
    B = plan.ntrans
    θ = plan.θ
    φ = plan.φ
    n = length(θ)
    fill!(alm, zero(eltype(alm)))
    @inbounds for b in 1:B, i in 1:n
        fi = ndims(f) == 1 ? f[i] : f[i, b]
        iszero(fi) && continue
        ti = θ[i]; pi = φ[i]
        for ℓ in abs(s):lmax, m in -ℓ:ℓ
            _spin_coeff_add!(alm, ℓ, m, lmax, b, conj(sYlm(s, ℓ, m, ti, pi)) * fi)
        end
    end
    return alm
end
