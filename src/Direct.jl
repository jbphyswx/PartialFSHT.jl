"""
Direct associated-Legendre evaluation of complex orthonormal spherical harmonics
(Condon–Shortley). This is the correctness oracle: `O(M L²)` and intentionally so.

Fully normalized ALFs `P̄_ℓ^m` satisfy

    Y_ℓm(θ,φ) = P̄_ℓ^m(cosθ) exp(imφ),     P̄_ℓ^{-m} = (-1)^m P̄_ℓ^m

and are computed by the Holmes–Featherstone fixed-order, increasing-degree recurrence
(Holmes & Featherstone, J. Geod. 76, 2002) with the Condon–Shortley sectorial sign.
"""

"""
    ylm(ℓ, m, θ, φ) -> Complex

Complex orthonormal spherical harmonic `Y_ℓm(θ,φ)` (Condon–Shortley).
"""
function ylm(ℓ::Integer, m::Integer, θ::Real, φ::Real)
    T = float(typeof(float(θ)))
    P = _alf_value(T, Int(ℓ), Int(m), cos(T(θ)), sin(T(θ)))
    return Complex{T}(P) * cis(T(m) * T(φ))
end

function _alf_value(::Type{T}, ℓ::Int, m::Int, x::T, s::T) where {T<:AbstractFloat}
    am = abs(m)
    am > ℓ && return zero(T)
    pmm = _invsqrt4π(T)
    @inbounds for k in 1:am
        pmm *= -s * sqrt(T(2k + 1) / T(2k))
    end
    P = pmm
    if ℓ > am
        pmmp1 = x * sqrt(T(2am + 3)) * pmm
        if ℓ == am + 1
            P = pmmp1
        else
            plm1 = pmmp1
            plm2 = pmm
            for el in (am + 2):ℓ
                a = sqrt(T(2el - 1) * T(2el + 1) / (T(el - am) * T(el + am)))
                b = sqrt(T(2el + 1) / T(2el - 3) * T(el - am - 1) / T(el - am) *
                         T(el + am - 1) / T(el + am))
                P = a * x * plm1 - b * plm2
                plm2 = plm1
                plm1 = P
            end
        end
    end
    return m < 0 ? ((iseven(am) ? one(T) : -one(T)) * P) : P
end

@inline _invsqrt4π(::Type{T}) where {T} = T(inv(sqrt(4T(π))))

"""
Scratch for [`synthesize_direct!`](@ref) / [`adjoint_direct!`](@ref). Built once per point set.
"""
struct DirectScratch{T<:AbstractFloat, CT<:Complex, V<:AbstractVector{T}, C<:AbstractVector{CT}}
    cosθ::V
    sinθ::V
    eipφ::C
    eimp::C
    Pmm::V
    P0::V
    P1::V
    P2::V
end

function DirectScratch(θ::AbstractVector, φ::AbstractVector)
    n = length(θ)
    length(φ) == n || throw(DimensionMismatch("θ and φ must have the same length"))
    T = float(eltype(θ))
    CT = Complex{T}
    cosθ = similar(θ, T)
    sinθ = similar(θ, T)
    eipφ = similar(θ, CT)
    eimp = similar(θ, CT)
    Pmm = similar(θ, T)
    P0 = similar(θ, T)
    P1 = similar(θ, T)
    P2 = similar(θ, T)
    cosθ .= cos.(T.(θ))
    sinθ .= sin.(T.(θ))
    eipφ .= cis.(T.(φ))
    return DirectScratch{T, CT, typeof(cosθ), typeof(eipφ)}(
        cosθ, sinθ, eipφ, eimp, Pmm, P0, P1, P2)
end

"""
    synthesize_direct!(f, alm, scratch, lmax) -> f

Type-2 synthesis: `f_i = Σ_{ℓm} a_{ℓm} Y_{ℓm}(Ω_i)`. Bang-first.
"""
function synthesize_direct!(f, alm, scratch::DirectScratch, lmax::Integer)
    n = length(scratch.cosθ)
    B = _ntrans(alm)
    _assert_coeffs(alm, lmax, B)
    _assert_field(f, n, B)
    fill!(f, zero(eltype(f)))
    _direct_apply!(f, alm, scratch, Int(lmax), false)
    return f
end

"""
    adjoint_direct!(alm, f, scratch, lmax) -> alm

Euclidean adjoint: `(Y^* f)_{ℓm} = Σ_i conj(Y_{ℓm}(Ω_i)) f_i`. Bang-first.
"""
function adjoint_direct!(alm, f, scratch::DirectScratch, lmax::Integer)
    n = length(scratch.cosθ)
    B = _ntrans(alm)
    _assert_coeffs(alm, lmax, B)
    _assert_field(f, n, B)
    fill!(alm, zero(eltype(alm)))
    _direct_apply!(f, alm, scratch, Int(lmax), true)
    return alm
end

function _direct_apply!(f, alm, scratch::DirectScratch{T}, lmax::Int, adj::Bool) where {T}
    alm_dev = alm
    alm = _coeff_host(alm)
    n = length(scratch.cosθ)
    B = _ntrans(alm)
    x = scratch.cosθ
    s = scratch.sinθ
    eip = scratch.eipφ
    eim = scratch.eimp
    Pmm = scratch.Pmm
    P0 = scratch.P0
    P1 = scratch.P1
    P2 = scratch.P2
    inv4π = _invsqrt4π(T)
    bc = _use_broadcast(x)

    fill!(Pmm, inv4π)
    _accum0!(f, alm, Pmm, n, B, 0, adj)
    if lmax >= 1
        _scale_legendre!(P1, x, Pmm, sqrt(T(3)), bc)
        _accum0!(f, alm, P1, n, B, 1, adj)
        copyto!(P0, Pmm)
        pa, pb, pc = P0, P1, P2
        for ℓ in 2:lmax
            a = sqrt(T(2ℓ - 1) * T(2ℓ + 1)) / T(ℓ)
            b = sqrt(T(2ℓ + 1) / T(2ℓ - 3)) * T(ℓ - 1) / T(ℓ)
            _legendre_step!(pc, x, pb, pa, a, b, bc)
            _accum0!(f, alm, pc, n, B, ℓ, adj)
            pa, pb, pc = pb, pc, pa
        end
    end

    copyto!(eim, eip)
    fill!(Pmm, inv4π)
    @inbounds for m in 1:lmax
        fac = -sqrt(T(2m + 1) / T(2m))
        _scale_inplace!(Pmm, s, fac, bc)
        _accum_pm!(f, alm, Pmm, eim, n, B, m, m, adj)
        if m < lmax
            _scale_legendre!(P1, x, Pmm, sqrt(T(2m + 3)), bc)
            _accum_pm!(f, alm, P1, eim, n, B, m + 1, m, adj)
            copyto!(P0, Pmm)
            pa, pb, pc = P0, P1, P2
            for ℓ in (m + 2):lmax
                a = sqrt(T(2ℓ - 1) * T(2ℓ + 1) / (T(ℓ - m) * T(ℓ + m)))
                b = sqrt(T(2ℓ + 1) / T(2ℓ - 3) * T(ℓ - m - 1) / T(ℓ - m) *
                         T(ℓ + m - 1) / T(ℓ + m))
                _legendre_step!(pc, x, pb, pa, a, b, bc)
                _accum_pm!(f, alm, pc, eim, n, B, ℓ, m, adj)
                pa, pb, pc = pb, pc, pa
            end
        end
        if m < lmax
            _scale_inplace!(eim, eip, one(eltype(eim)), bc)
        end
    end
    if adj && alm_dev !== alm
        copyto!(alm_dev, alm)
    end
    return nothing
end

@inline function _scale_legendre!(out, x, P, c, bc::Bool)
    if bc
        return _scale_legendre_device!(out, x, P, c)
    else
        @inbounds for i in eachindex(out)
            out[i] = c * x[i] * P[i]
        end
        return out
    end
end

@inline function _legendre_step!(out, x, pb, pa, a, b, bc::Bool)
    if bc
        return _legendre_step_device!(out, x, pb, pa, a, b)
    else
        @inbounds for i in eachindex(out)
            out[i] = a * x[i] * pb[i] - b * pa[i]
        end
        return out
    end
end

@inline function _scale_inplace!(P, s, fac, bc::Bool)
    if bc
        return _scale_inplace_device!(P, s, fac)
    else
        @inbounds for i in eachindex(P)
            P[i] *= fac * s[i]
        end
        return P
    end
end

# Device fallback (broadcast). KernelAbstractions overrides these for GPU arrays.
_scale_legendre_device!(out, x, P, c) = (@. out = c * x * P; out)
_legendre_step_device!(out, x, pb, pa, a, b) = (@. out = a * x * pb - b * pa; out)
_scale_inplace_device!(P, s, fac) = (@. P *= fac * s; P)

@inline function _fget(f, i, b)
    return ndims(f) == 1 ? f[i] : f[i, b]
end

@inline function _fadd!(f, i, b, val)
    if ndims(f) == 1
        f[i] += val
    else
        f[i, b] += val
    end
    return nothing
end

function _accum0!(f, alm, P, n::Int, B::Int, ℓ::Int, adj::Bool)
    if _use_broadcast(P)
        return _accum0_bc!(f, alm, P, B, ℓ, adj)
    end
    if adj
        @inbounds for b in 1:B
            acc = zero(eltype(alm))
            for i in 1:n
                acc += P[i] * _fget(f, i, b)
            end
            _coeff_add!(alm, ℓ, 0, b, acc)
        end
    else
        @inbounds for b in 1:B
            a = _coeff(alm, ℓ, 0, b)
            iszero(a) && continue
            for i in 1:n
                _fadd!(f, i, b, a * P[i])
            end
        end
    end
    return nothing
end

function _accum0_bc!(f, alm, P, B::Int, ℓ::Int, adj::Bool)
    if adj
        for b in 1:B
            fb = ndims(f) == 1 ? f : view(f, :, b)
            _coeff_add!(alm, ℓ, 0, b, LinearAlgebra.dot(P, fb))
        end
    else
        for b in 1:B
            a = _coeff(alm, ℓ, 0, b)
            iszero(a) && continue
            if ndims(f) == 1
                @. f += a * P
            else
                fb = view(f, :, b)
                @. fb += a * P
            end
        end
    end
    return nothing
end

function _accum_pm!(f, alm, P, eim, n::Int, B::Int, ℓ::Int, m::Int, adj::Bool)
    sgn = iseven(m) ? one(eltype(P)) : -one(eltype(P))
    if _use_broadcast(P)
        return _accum_pm_bc!(f, alm, P, eim, B, ℓ, m, adj, sgn)
    end
    if adj
        @inbounds for b in 1:B
            accp = zero(eltype(alm))
            accm = zero(eltype(alm))
            for i in 1:n
                fi = _fget(f, i, b)
                Pi = P[i]
                e = eim[i]
                accp += Pi * conj(e) * fi
                accm += sgn * Pi * e * fi
            end
            _coeff_add!(alm, ℓ, m, b, accp)
            _coeff_add!(alm, ℓ, -m, b, accm)
        end
    else
        @inbounds for b in 1:B
            ap = _coeff(alm, ℓ, m, b)
            amv = _coeff(alm, ℓ, -m, b)
            (iszero(ap) & iszero(amv)) && continue
            for i in 1:n
                e = eim[i]
                _fadd!(f, i, b, P[i] * (ap * e + amv * sgn * conj(e)))
            end
        end
    end
    return nothing
end

function _accum_pm_bc!(f, alm, P, eim, B::Int, ℓ::Int, m::Int, adj::Bool, sgn)
    if adj
        for b in 1:B
            fb = ndims(f) == 1 ? f : view(f, :, b)
            _coeff_add!(alm, ℓ, m, b, LinearAlgebra.dot(P .* eim, fb))
            _coeff_add!(alm, ℓ, -m, b, sgn * LinearAlgebra.dot(P .* conj.(eim), fb))
        end
    else
        for b in 1:B
            ap = _coeff(alm, ℓ, m, b)
            amv = _coeff(alm, ℓ, -m, b)
            (iszero(ap) & iszero(amv)) && continue
            if ndims(f) == 1
                @. f += P * (ap * eim + amv * sgn * conj(eim))
            else
                fb = view(f, :, b)
                @. fb += P * (ap * eim + amv * sgn * conj(eim))
            end
        end
    end
    return nothing
end
