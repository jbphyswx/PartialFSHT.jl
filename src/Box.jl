"""
Spherical geometry as a colatitude × longitude box.

The delivered code used a bounding *cap* centred on the normalised mean of the unit vectors.
That is neither minimal nor well behaved: for an equatorial band the mean vector is degenerate
and the cap radius approaches `π`, so a band — the geometry a partial-sky transform should
handle best — came out looking like the whole sphere and got split into up to 32 sub-caps.

A colatitude × longitude box is *exactly* minimal and costs `O(n log n)`: the colatitude
extent is a min/max, and the longitude extent is `2π` minus the largest circular gap. It also
matches what the transform actually exploits — one stretched interval per axis — so the
geometry and the algorithm agree instead of the geometry being a proxy.
"""

"""
    SphereBox{T}

Minimal colatitude × longitude box containing a point set.

- `θlo`, `θhi` — colatitude extent
- `φstart`, `φextent` — longitude arc, measured counter-clockwise from `φstart`; `φextent`
  is `2π` when the points wrap the full circle
- `touches_north`, `touches_south` — the colatitude interval reaches a pole, so a longitude
  restriction is meaningless there and the DFS reflection applies
"""
struct SphereBox{T<:AbstractFloat}
    θlo::T
    θhi::T
    φstart::T
    φextent::T
    touches_north::Bool
    touches_south::Bool
end

"""
    bounding_box(θ, φ) -> SphereBox

Exactly minimal box containing the points. `O(n log n)`, dominated by the sort needed for the
largest-circular-gap scan in longitude.

A colatitude interval that reaches either pole forces the full longitude range, since every
longitude is arbitrarily close to the pole; reporting a narrow arc there would be wrong rather
than merely loose.
"""
function bounding_box(θ::AbstractVector, φ::AbstractVector)
    n = length(θ)
    n == length(φ) || throw(DimensionMismatch("θ and φ must have the same length"))
    n >= 1 || throw(ArgumentError("need at least one point"))
    T = float(eltype(θ))
    θh = _host(θ)
    φh = _host(φ)

    θlo = T(Inf)
    θhi = T(-Inf)
    @inbounds for i in 1:n
        t = T(θh[i])
        θlo = min(θlo, t)
        θhi = max(θhi, t)
    end
    θlo = max(θlo, zero(T))
    θhi = min(θhi, T(π))

    # Ring-radius scale below which a colatitude is indistinguishable from the pole for a
    # longitude restriction: use the geometric radius sin(θ) against the box's own θ extent.
    polar_tol = sqrt(eps(T))
    touches_north = θlo <= polar_tol
    touches_south = θhi >= T(π) - polar_tol

    if touches_north || touches_south
        return SphereBox{T}(θlo, θhi, zero(T), 2 * T(π), touches_north, touches_south)
    end

    φstart, φextent = _circular_extent(φh, T)
    return SphereBox{T}(θlo, θhi, φstart, φextent, false, false)
end

"""
Smallest arc containing all longitudes: sort, take the largest gap between consecutive angles
(including the wrap-around gap), and the complement of that gap is the answer.
"""
function _circular_extent(φh::AbstractVector, ::Type{T}) where {T}
    n = length(φh)
    ps = Vector{T}(undef, n)
    @inbounds for i in 1:n
        p = mod(T(φh[i]), 2 * T(π))
        ps[i] = p
    end
    sort!(ps)
    n == 1 && return ps[1], zero(T)
    gap = zero(T)
    at = 1
    @inbounds for i in 1:(n - 1)
        g = ps[i + 1] - ps[i]
        if g > gap
            gap = g
            at = i + 1
        end
    end
    wrap = 2 * T(π) - ps[n] + ps[1]
    if wrap > gap
        gap = wrap
        at = 1
    end
    extent = 2 * T(π) - gap
    extent <= 0 && return ps[1], zero(T)
    extent >= 2 * T(π) && return zero(T), 2 * T(π)
    return ps[at], extent
end

"""
    box_fsky(box) -> Real

Fraction of the sphere the box covers, `(cos θlo - cos θhi)·φextent / 4π`.
"""
function box_fsky(box::SphereBox{T}) where {T}
    return (cos(box.θlo) - cos(box.θhi)) * box.φextent / (4 * T(π))
end

"""
    box_theta_extent(box) -> Real

Colatitude extent of the box; the quantity the stretched transform's ring count is proportional
to, and therefore the one that sets the speedup.
"""
box_theta_extent(box::SphereBox) = box.θhi - box.θlo

"""
    box_phi_extent(box) -> Real

Longitude arc length, `2π` when the points wrap the circle.
"""
box_phi_extent(box::SphereBox) = box.φextent

"""
    box_wraps_phi(box) -> Bool
"""
box_wraps_phi(box::SphereBox{T}) where {T} = box.φextent >= 2 * T(π) - sqrt(eps(T))

"""
    shannon_number(lmax, box) -> Real

`N_S = (lmax+1)² · f_sky`: the number of degrees of freedom of a degree-`lmax` field that a
region of this size can constrain (Simons, Dahlen & Wieczorek, SIAM Rev. 48, 2006). This is
the ceiling on what any inversion from data in `box` can recover, independent of method.
"""
shannon_number(lmax::Integer, box::SphereBox) = (Int(lmax) + 1)^2 * box_fsky(box)

shannon_number(lmax::Integer, fsky::Real) = (Int(lmax) + 1)^2 * fsky

"""
    spectral_resolution(box) -> Real

`Δℓ ≈ π/Λ` with `Λ` the box's larger angular extent: the finest degree separation the geometry
can distinguish, i.e. the width of the mode-coupling kernel a region of this size imposes.
Band-powers finer than this are not resolved no matter how many points are supplied.
"""
function spectral_resolution(box::SphereBox{T}) where {T}
    Λ = max(box_theta_extent(box), box.φextent * sin(clamp((box.θlo + box.θhi) / 2,
                                                           sqrt(eps(T)), T(π) - sqrt(eps(T)))))
    Λ <= 0 && return T(Inf)
    return T(π) / Λ
end

# ---------------------------------------------------------------------------------------
# Partition into several boxes
# ---------------------------------------------------------------------------------------

"""
    partition_boxes(θ, φ, costfn; maxboxes=8) -> Vector{Vector{Int}}

Split the point set into boxes while the predicted cost falls. `costfn(box, npts)` returns a
cost estimate for one box, so the split rule is driven by the calibrated model rather than by
tuned thresholds.

Splits are taken at the *largest gap* in colatitude or longitude, which is the only cut that
can shrink a box's extent without discarding points. A cut is kept only if it lowers total
predicted cost, so a compact patch stays a single box, a band stays a single box (a band has no
interior gap to cut), and genuinely disjoint components separate.
"""
function partition_boxes(θ::AbstractVector, φ::AbstractVector, costfn;
                         maxboxes::Integer = 8)
    n = length(θ)
    θh = _host(θ)
    φh = _host(φ)
    groups = [collect(1:n)]
    boxes = [bounding_box(θh, φh)]
    costs = [costfn(boxes[1], n)]
    while length(groups) < Int(maxboxes)
        best_gain = zero(eltype(costs))
        best_at = 0
        best_split = (Int[], Int[])
        for (gi, g) in enumerate(groups)
            length(g) >= 2 || continue
            sp = _best_gap_split(θh, φh, g)
            sp === nothing && continue
            ga, gb = sp
            (isempty(ga) || isempty(gb)) && continue
            ca = costfn(bounding_box(view(θh, ga), view(φh, ga)), length(ga))
            cb = costfn(bounding_box(view(θh, gb), view(φh, gb)), length(gb))
            gain = costs[gi] - (ca + cb)
            if gain > best_gain
                best_gain = gain
                best_at = gi
                best_split = (ga, gb)
            end
        end
        best_at == 0 && break
        ga, gb = best_split
        deleteat!(groups, best_at)
        deleteat!(boxes, best_at)
        deleteat!(costs, best_at)
        for g in (ga, gb)
            push!(groups, g)
            b = bounding_box(view(θh, g), view(φh, g))
            push!(boxes, b)
            push!(costs, costfn(b, length(g)))
        end
    end
    return groups
end

"""
Split a group at its largest colatitude or longitude gap, whichever gap subtends more area.
Returns `nothing` when the group has no interior gap worth cutting.
"""
function _best_gap_split(θh, φh, g::Vector{Int})
    T = float(eltype(θh))
    ord = sortperm(view(θh, g))
    gapθ = zero(T)
    atθ = 0
    @inbounds for i in 1:(length(ord) - 1)
        d = T(θh[g[ord[i + 1]]]) - T(θh[g[ord[i]]])
        if d > gapθ
            gapθ = d
            atθ = i
        end
    end

    ordφ = sortperm([mod(T(φh[i]), 2 * T(π)) for i in g])
    gapφ = zero(T)
    atφ = 0
    @inbounds for i in 1:(length(ordφ) - 1)
        d = mod(T(φh[g[ordφ[i + 1]]]), 2 * T(π)) - mod(T(φh[g[ordφ[i]]]), 2 * T(π))
        if d > gapφ
            gapφ = d
            atφ = i
        end
    end

    # Compare gaps by the angular width they remove, longitude weighted by ring radius.
    θmid = (T(θh[g[ord[1]]]) + T(θh[g[ord[end]]])) / 2
    wφ = gapφ * sin(clamp(θmid, sqrt(eps(T)), T(π) - sqrt(eps(T))))
    if gapθ >= wφ
        atθ == 0 && return nothing
        a = [g[ord[i]] for i in 1:atθ]
        b = [g[ord[i]] for i in (atθ + 1):length(ord)]
        return (a, b)
    else
        atφ == 0 && return nothing
        a = [g[ordφ[i]] for i in 1:atφ]
        b = [g[ordφ[i]] for i in (atφ + 1):length(ordφ)]
        return (a, b)
    end
end
