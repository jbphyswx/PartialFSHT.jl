"""
Spherical geometry: unit vectors, bounding polar caps, point clustering, cost-model helpers.
"""

"""
    BoundingCap{T}

Spherical bounding cap of a point set: unit-vector `center`, colatitude radius `θstar`,
and the center's spherical coordinates `(θc, φc)`.
"""
struct BoundingCap{T<:AbstractFloat}
    center::NTuple{3, T}
    θstar::T
    θc::T
    φc::T
end

function _xyz(θ, φ)
    s, c = sincos(θ)
    return (s * cos(φ), s * sin(φ), c)
end

"""
    bounding_cap(θ, φ) -> BoundingCap

Center is the normalized mean of the unit vectors; `θstar` is the maximum angular
distance from that center. A single point yields `θstar = 0`.
"""
function bounding_cap(θ::AbstractVector, φ::AbstractVector)
    n = length(θ)
    n == length(φ) || throw(DimensionMismatch("θ and φ must have the same length"))
    n >= 1 || throw(ArgumentError("need at least one point"))
    # Plan-time reduction is scalar; copy device nodes to host once.
    θh = _host(θ)
    φh = _host(φ)
    T = float(eltype(θ))
    sx = zero(T); sy = zero(T); sz = zero(T)
    @inbounds for i in 1:n
        x, y, z = _xyz(T(θh[i]), T(φh[i]))
        sx += x; sy += y; sz += z
    end
    nrm = sqrt(sx * sx + sy * sy + sz * sz)
    if nrm < eps(T) * T(n)
        # opposite points: fall back to the first sample as center
        x, y, z = _xyz(T(θh[1]), T(φh[1]))
    else
        x, y, z = sx / nrm, sy / nrm, sz / nrm
    end
    θc = acos(clamp(z, -one(T), one(T)))
    φc = atan(y, x)
    φc < 0 && (φc += 2T(π))
    θstar = zero(T)
    @inbounds for i in 1:n
        xi, yi, zi = _xyz(T(θh[i]), T(φh[i]))
        cang = clamp(x * xi + y * yi + z * zi, -one(T), one(T))
        ang = acos(cang)
        θstar = max(θstar, ang)
    end
    θstar = min(θstar, T(π))
    return BoundingCap{T}((x, y, z), θstar, θc, φc)
end

"""
    rotate_to_pole!(θ′, φ′, θ, φ, cap) -> (θ′, φ′)

Rotate points so `cap.center` maps to the north pole: `R = Ry(-θc) Rz(-φc)`.
"""
function rotate_to_pole!(θ′::AbstractVector, φ′::AbstractVector,
                         θ::AbstractVector, φ::AbstractVector, cap::BoundingCap{T}) where {T}
    n = length(θ)
    cβ, sβ = cos(-cap.θc), sin(-cap.θc)
    cγ, sγ = cos(-cap.φc), sin(-cap.φc)
    @inbounds for i in 1:n
        x, y, z = _xyz(T(θ[i]), T(φ[i]))
        # Rz(-φc)
        x1 = cγ * x - sγ * y
        y1 = sγ * x + cγ * y
        z1 = z
        # Ry(-θc)
        x2 = cβ * x1 + sβ * z1
        y2 = y1
        z2 = -sβ * x1 + cβ * z1
        θ′[i] = acos(clamp(z2, -one(T), one(T)))
        p = atan(y2, x2)
        φ′[i] = p < 0 ? p + 2T(π) : p
    end
    return θ′, φ′
end

function rotate_to_pole(θ::AbstractVector, φ::AbstractVector, cap::BoundingCap)
    T = typeof(cap.θstar)
    θ′ = similar(θ, T)
    φ′ = similar(φ, T)
    return rotate_to_pole!(θ′, φ′, θ, φ, cap)
end

"""
    euler_to_pole(cap) -> (α, β, γ)

ZYZ Euler angles of the rotation that sends `cap.center` to the north pole,
matching [`rotate_to_pole!`](@ref): `(0, -θc, -φc)`.
"""
euler_to_pole(cap::BoundingCap) = (zero(cap.θc), -cap.θc, -cap.φc)

@inline function _apply_pole_rotation(x, y, z, θc, φc)
    cγ, sγ = cos(-φc), sin(-φc)
    x1 = cγ * x - sγ * y
    y1 = sγ * x + cγ * y
    cβ, sβ = cos(-θc), sin(-θc)
    return (cβ * x1 + sβ * z, y1, -sβ * x1 + cβ * z)
end

"""
    spin_frame_angle(θ, φ, θ′, φ′, cap) -> χ

Angle from the original phi-hat basis, transported by the pole rotation, to the
spherical phi-hat basis at the rotated coordinates. A spin-`s` field picks up
`exp(i s χ)`.
"""
function spin_frame_angle(θ::Real, φ::Real, θ′::Real, φ′::Real, cap::BoundingCap)
    T = typeof(cap.θc)
    # original φ̂ = (-sinφ, cosφ, 0)
    rx, ry, rz = _apply_pole_rotation(-sin(T(φ)), cos(T(φ)), zero(T), cap.θc, cap.φc)
    s′, c′ = sincos(T(θ′))
    cp, sp = cos(T(φ′)), sin(T(φ′))
    # ê_θ′ = (c′ cp, c′ sp, -s′), ê_φ′ = (-sp, cp, 0)
    y = rx * (c′ * cp) + ry * (c′ * sp) + rz * (-s′)
    x = rx * (-sp) + ry * cp
    return atan(y, x)
end

function spin_basis_phase!(out::AbstractVector, θ, φ, θ′, φ′, cap::BoundingCap, spin::Integer)
    CT = eltype(out)
    n = length(θ)
    s = Int(spin)
    if s == 0
        fill!(out, one(CT))
        return out
    end
    @inbounds for i in 1:n
        χ = spin_frame_angle(θ[i], φ[i], θ′[i], φ′[i], cap)
        out[i] = cis(eltype(χ)(s) * χ)
    end
    return out
end

"""
    cluster_points(θ, φ; k=nothing, target_θstar=nothing, max_k=32, seed=0) -> Vector{Vector{Int}}

Partition point indices into spatially coherent clusters. If `k` is given, run spherical
k-means with that many clusters. Otherwise grow `k` until every cluster's bounding cap
has `θstar ≤ target_θstar` (default `π/6`) or `k == max_k`.
"""
function cluster_points(θ::AbstractVector, φ::AbstractVector;
                        k::Union{Nothing, Integer} = nothing,
                        target_θstar = nothing,
                        max_k::Integer = 32,
                        seed::Integer = 0)
    n = length(θ)
    T = float(eltype(θ))
    θh = _host(θ)
    φh = _host(φ)
    xyz = Array{T}(undef, 3, n)
    @inbounds for i in 1:n
        xyz[1, i], xyz[2, i], xyz[3, i] = _xyz(T(θh[i]), T(φh[i]))
    end
    kwant = k === nothing ? 1 : Int(k)
    θtarg = target_θstar === nothing ? T(π) / 6 : T(target_θstar)
    rng = seed == 0 ? nothing : seed
    kuse = kwant
    labels = Vector{Int}(undef, n)
    while true
        _spherical_kmeans!(labels, xyz, kuse, rng)
        groups = [Int[] for _ in 1:kuse]
        @inbounds for i in 1:n
            push!(groups[labels[i]], i)
        end
        groups = filter!(!isempty, groups)
        if k !== nothing
            return groups
        end
        worst = zero(T)
        for g in groups
            cap = bounding_cap(view(θh, g), view(φh, g))
            worst = max(worst, cap.θstar)
        end
        (worst <= θtarg || kuse >= Int(max_k) || kuse >= n) && return groups
        kuse += 1
    end
end

function _spherical_kmeans!(labels::AbstractVector{Int}, xyz::AbstractMatrix{T}, k::Int, seed) where {T}
    n = size(xyz, 2)
    k = clamp(k, 1, n)
    centers = Array{T}(undef, 3, k)
    # farthest-point init
    fill!(view(centers, :, 1), zero(T))
    @inbounds for j in 1:3
        centers[j, 1] = xyz[j, 1]
    end
    taken = falses(n); taken[1] = true
    d2 = fill(T(Inf), n)
    for c in 2:k
        @inbounds for i in 1:n
            taken[i] && continue
            dx = xyz[1, i] - centers[1, c - 1]
            dy = xyz[2, i] - centers[2, c - 1]
            dz = xyz[3, i] - centers[3, c - 1]
            d2[i] = min(d2[i], dx * dx + dy * dy + dz * dz)
        end
        best = 1; bestd = -one(T)
        @inbounds for i in 1:n
            taken[i] && continue
            if d2[i] > bestd
                bestd = d2[i]; best = i
            end
        end
        taken[best] = true
        @inbounds for j in 1:3
            centers[j, c] = xyz[j, best]
        end
    end
    counts = zeros(Int, k)
    for _iter in 1:25
        changed = false
        fill!(counts, 0)
        @inbounds for i in 1:n
            best = 1; bestd = -T(Inf)
            for c in 1:k
                dotp = xyz[1, i] * centers[1, c] + xyz[2, i] * centers[2, c] + xyz[3, i] * centers[3, c]
                if dotp > bestd
                    bestd = dotp; best = c
                end
            end
            if labels[i] != best
                labels[i] = best
                changed = true
            end
            counts[best] += 1
        end
        fill!(centers, zero(T))
        @inbounds for i in 1:n
            c = labels[i]
            centers[1, c] += xyz[1, i]
            centers[2, c] += xyz[2, i]
            centers[3, c] += xyz[3, i]
        end
        @inbounds for c in 1:k
            nrm = sqrt(centers[1, c]^2 + centers[2, c]^2 + centers[3, c]^2)
            if nrm > 0
                centers[1, c] /= nrm; centers[2, c] /= nrm; centers[3, c] /= nrm
            end
        end
        !changed && break
    end
    return labels
end

"""
    fsky_cap(θstar) -> Real

Fraction of the sphere occupied by a cap of colatitude `θstar`.
"""
fsky_cap(θstar) = (1 - cos(θstar)) / 2
