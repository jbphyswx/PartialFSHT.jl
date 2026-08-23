"""
Coefficient array layout matching FastSphericalHarmonics / FastTransforms packing.

Complex orthonormal spherical harmonics (Condon–Shortley) are stored in an
`(lmax+1) × (2lmax+1)` array. The coefficient of degree `ℓ` and order `m` sits at
`sph_mode(ℓ, m)`. Unused slots are zero.

Spin-weighted coefficients use a dense `(ℓ+1, m+lmax+1)` layout, matching NUFSHT's
public `spin_coeff_index`.
"""

"""
    coeff_size(lmax) -> Tuple{Int,Int}

Shape of a coefficient array, `(lmax+1, 2lmax+1)`. Only `(lmax+1)²` of those slots are real
modes; the rest are padding the transform never reads, so filling a whole array with random
numbers is fine for comparing fields and wrong for comparing coefficients.
"""
@inline coeff_size(lmax::Integer) = (Int(lmax) + 1, 2Int(lmax) + 1)

"""
    sph_mode(ℓ, m) -> CartesianIndex

Index of the degree-`ℓ`, order-`m` coefficient, in FastSphericalHarmonics packing.
"""
@inline sph_mode(ℓ::Integer, m::Integer) = FastSphericalHarmonics.sph_mode(ℓ, m)

"""
    spin_coeff_index(ℓ, m, lmax) -> CartesianIndex

Index of a spin-weighted coefficient in the dense `(ℓ+1, m+lmax+1)` layout, matching NUFSHT's
public spin packing.
"""
@inline spin_coeff_index(ℓ::Integer, m::Integer, lmax::Integer) =
    CartesianIndex(Int(ℓ) + 1, Int(m) + Int(lmax) + 1)

@inline function _ntrans(A::AbstractArray)
    return ndims(A) >= 3 ? size(A, 3) : 1
end

@inline function _assert_coeffs(alm, lmax::Integer, B::Integer)
    nθ, nφ = coeff_size(lmax)
    if B == 1 && ndims(alm) == 2
        size(alm, 1) == nθ && size(alm, 2) == nφ || throw(DimensionMismatch(
            "coefficient array is $(size(alm)), expected ($nθ, $nφ)"))
    else
        size(alm, 1) == nθ && size(alm, 2) == nφ && size(alm, 3) == B || throw(DimensionMismatch(
            "coefficient array is $(size(alm)), expected ($nθ, $nφ, $B)"))
    end
    return nothing
end

@inline function _assert_field(f, npts::Integer, B::Integer)
    if B == 1 && ndims(f) == 1
        length(f) == npts || throw(DimensionMismatch(
            "field length is $(length(f)), expected $npts"))
    else
        size(f, 1) == npts && size(f, 2) == B || throw(DimensionMismatch(
            "field is $(size(f)), expected ($npts, $B)"))
    end
    return nothing
end

@inline function _coeff(alm, ℓ, m, b)
    idx = sph_mode(ℓ, m)
    return ndims(alm) >= 3 ? alm[idx, b] : alm[idx]
end

@inline function _coeff_set!(alm, ℓ, m, b, val)
    idx = sph_mode(ℓ, m)
    if ndims(alm) >= 3
        alm[idx, b] = val
    else
        alm[idx] = val
    end
    return val
end

@inline function _coeff_add!(alm, ℓ, m, b, val)
    idx = sph_mode(ℓ, m)
    if ndims(alm) >= 3
        alm[idx, b] += val
    else
        alm[idx] += val
    end
    return nothing
end

@inline function _spin_coeff(alm, ℓ, m, lmax, b)
    idx = spin_coeff_index(ℓ, m, lmax)
    return ndims(alm) >= 3 ? alm[idx, b] : alm[idx]
end

@inline function _spin_coeff_set!(alm, ℓ, m, lmax, b, val)
    idx = spin_coeff_index(ℓ, m, lmax)
    if ndims(alm) >= 3
        alm[idx, b] = val
    else
        alm[idx] = val
    end
    return val
end

@inline function _spin_coeff_add!(alm, ℓ, m, lmax, b, val)
    idx = spin_coeff_index(ℓ, m, lmax)
    if ndims(alm) >= 3
        alm[idx, b] += val
    else
        alm[idx] += val
    end
    return nothing
end

# Buffers follow the node array type (host `Array`, `JLArray`, `CuArray`, …).
@inline _zeros_like(ref::AbstractArray, ::Type{S}, dims::Integer...) where {S} =
    fill!(similar(ref, S, dims...), zero(S))
@inline _undef_like(ref::AbstractArray, ::Type{S}, dims::Integer...) where {S} =
    similar(ref, S, dims...)
@inline _to_like(ref::AbstractArray, a::AbstractArray) =
    copyto!(similar(ref, eltype(a), size(a)...), a)
@inline _host(x::Array) = x
@inline _host(x::AbstractArray) = Array(x)

# CPU dense arrays keep the scalar Holmes–Featherstone loops (no extra temps).
# Device arrays (`JLArray`, `CuArray`, …) use broadcasting / host copies of `alm`.
@inline _use_broadcast(::Array) = false
@inline _use_broadcast(::SubArray{T, N, <:Array}) where {T, N} = false
@inline _use_broadcast(::AbstractArray) = true

"""Host copy of a coefficient array for `_coeff` / `_coeff_set!` (scalar indexing)."""
@inline _coeff_host(alm::Array) = alm
@inline _coeff_host(alm::SubArray{T, N, <:Array}) where {T, N} = alm
@inline _coeff_host(alm::AbstractArray) = Array(alm)

"""
    zeros_alm([CT=ComplexF64], lmax; ntrans=1)
    zeros_alm(ref, lmax; ntrans=1)

Zero coefficient array in FastSphericalHarmonics layout. The `ref` method
matches `ref`'s array type so a device node set yields a device coefficient array.
"""
function zeros_alm(::Type{CT}, lmax::Integer; ntrans::Integer = 1) where {CT}
    nθ, nφ = coeff_size(lmax)
    return ntrans == 1 ? zeros(CT, nθ, nφ) : zeros(CT, nθ, nφ, ntrans)
end
zeros_alm(lmax::Integer; ntrans::Integer = 1) = zeros_alm(ComplexF64, lmax; ntrans = ntrans)
function zeros_alm(ref::AbstractArray, lmax::Integer; ntrans::Integer = 1)
    nθ, nφ = coeff_size(lmax)
    E = eltype(ref)
    S = E <: Complex ? E : Complex{float(E)}
    return ntrans == 1 ? _zeros_like(ref, S, nθ, nφ) : _zeros_like(ref, S, nθ, nφ, ntrans)
end

function _field_zeros(ref::AbstractArray, ::Type{S}, npts::Integer, B::Integer) where {S}
    return B == 1 ? _zeros_like(ref, S, npts) : _zeros_like(ref, S, npts, B)
end
