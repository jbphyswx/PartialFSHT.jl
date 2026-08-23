# Spherical geometry as a colatitude × longitude box.
#
# A box is exactly minimal, and it matches what the transform actually exploits — one stretched
# interval per axis — so the geometry and the algorithm agree instead of the geometry being a proxy.
# A bounding *cap* is neither: for an equatorial band the mean direction is degenerate and the cap
# radius approaches π, so the geometry a partial-sky transform should handle best looks like the
# whole sphere.
#
# The colatitude extent is a min/max and the longitude extent is 2π minus the largest circular gap.
# Both come from `_max_gap`, which is O(n) and does not sort; see the manual under Implementation
# notes for why that matters to plan construction.

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
    T = float(promote_type(float(eltype(θ)), float(eltype(φ))))
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

    # Colatitude below which a longitude restriction is meaningless, because every longitude is
    # then within `sin θ ≈ θ` of the pole.
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
Smallest arc containing all longitudes: the complement of the largest gap between consecutive
angles, including the wrap-around gap. The gap comes from [`_max_gap`](@ref), so this is `O(n)` and
does not sort — a sort here costs 0.05 s per plan at two million points, for one scalar.
"""
function _circular_extent(φh::AbstractVector, ::Type{T}) where {T}
    n = length(φh)
    ps = Vector{T}(undef, n)
    @inbounds for i in 1:n
        ps[i] = mod(T(φh[i]), 2 * T(π))
    end
    n == 1 && return ps[1], zero(T)
    ws = _GapWorkspace{T}(n)
    φlo, φhi, gap, at = _max_gap(ps, Base.OneTo(n), ws)
    wrap = 2 * T(π) - φhi + φlo
    start = at
    if wrap >= gap
        gap = wrap
        start = φlo
    end
    extent = 2 * T(π) - gap
    extent <= 0 && return φlo, zero(T)
    extent >= 2 * T(π) && return zero(T), 2 * T(π)
    return start, extent
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
partition_boxes(θ::AbstractVector, φ::AbstractVector, costfn; maxboxes::Integer = 8) =
    first(partition_boxes_costed(θ, φ, costfn; maxboxes = maxboxes))

"""
    partition_boxes_costed(θ, φ, costfn; maxboxes=8) -> (groups, boxes, costs)

[`partition_boxes`](@ref) plus the box and predicted cost it already computed for each group. The
planner and the multi-box leaf both need those, and re-deriving a box per group afterwards means
another sort of the point set per group.
"""
function partition_boxes_costed(θ::AbstractVector, φ::AbstractVector, costfn;
                                maxboxes::Integer = 8)
    n = length(θ)
    T = float(promote_type(float(eltype(θ)), float(eltype(φ))))
    θh = _host(θ)
    φh = _host(φ)

    # Nothing here sorts. Both the box and the split site need one quantity — the largest gap in a
    # coordinate — and that is exactly computable in `O(|g|)` by bucketing (see [`_max_gap`](@ref)),
    # so the whole search is `O(maxboxes · n)`. Sorting per candidate instead cost 2.39 s and
    # 198 MB at `M = 2e6`, to return a single box; keeping two maintained orders still cost 1.0 s
    # for the two initial sorts alone.
    θv = Vector{T}(undef, n)
    φv = Vector{T}(undef, n)
    twoπ = 2 * T(π)
    @inbounds for i in 1:n
        θv[i] = T(θh[i])
        φv[i] = mod(T(φh[i]), twoπ)
    end

    ws = _GapWorkspace{T}(n)
    groups = [collect(1:n)]
    boxes = [_group_box(θv, φv, groups[1], ws)]
    costs = [costfn(boxes[1], n)]
    while length(groups) < Int(maxboxes)
        best_gain = zero(eltype(costs))
        best_at = 0
        best_split = nothing
        for (gi, g) in enumerate(groups)
            length(g) >= 2 || continue
            sp = _best_gap_split(θv, φv, g, ws)
            sp === nothing && continue
            ga, gb = sp
            ba = _group_box(θv, φv, ga, ws)
            bb = _group_box(θv, φv, gb, ws)
            gain = costs[gi] - (costfn(ba, length(ga)) + costfn(bb, length(gb)))
            if gain > best_gain
                best_gain = gain
                best_at = gi
                best_split = (ga, gb, ba, bb)
            end
        end
        best_at == 0 && break
        ga, gb, ba, bb = best_split
        deleteat!(groups, best_at)
        deleteat!(boxes, best_at)
        deleteat!(costs, best_at)
        for (g, b) in ((ga, ba), (gb, bb))
            push!(groups, g)
            push!(boxes, b)
            push!(costs, costfn(b, length(g)))
        end
    end
    return groups, boxes, costs
end

"""Bucket scratch for [`_max_gap`](@ref), sized once for the whole point set and reused."""
struct _GapWorkspace{T<:AbstractFloat}
    bmin::Vector{T}
    bmax::Vector{T}
    used::BitVector
end
_GapWorkspace{T}(n::Integer) where {T} =
    _GapWorkspace{T}(Vector{T}(undef, max(1, n)), Vector{T}(undef, max(1, n)), falses(max(1, n)))

"""
    _max_gap(v, g, ws) -> (lo, hi, gap, at)

Extremes of `v` over the indices `g`, its largest gap between consecutive values, and the value at
the gap's upper edge — in `O(|g|)` and without sorting.

Distribute the values into `|g|-1` equal buckets. The largest gap is at least the bucket width, so
it cannot lie *inside* a bucket, and one pass over the buckets in order finds it exactly. This is
the standard maximum-gap argument, and it is what removes the sort from a search whose result is
almost always "do not split".
"""
function _max_gap(v::Vector{T}, g, ws::_GapWorkspace{T}) where {T}
    n = length(g)
    @inbounds lo = hi = v[g[1]]
    @inbounds for i in 2:n
        x = v[g[i]]
        lo = min(lo, x)
        hi = max(hi, x)
    end
    (n < 2 || hi <= lo) && return lo, hi, zero(T), lo
    nb = n - 1
    w = (hi - lo) / nb
    @inbounds for k in 1:nb
        ws.used[k] = false
    end
    @inbounds for i in 1:n
        x = v[g[i]]
        k = clamp(Int(floor((x - lo) / w)) + 1, 1, nb)
        if ws.used[k]
            ws.bmin[k] = min(ws.bmin[k], x)
            ws.bmax[k] = max(ws.bmax[k], x)
        else
            ws.used[k] = true
            ws.bmin[k] = x
            ws.bmax[k] = x
        end
    end
    gap = zero(T)
    at = hi
    prev = lo
    @inbounds for k in 1:nb
        ws.used[k] || continue
        d = ws.bmin[k] - prev
        if d > gap
            gap = d
            at = ws.bmin[k]
        end
        prev = ws.bmax[k]
    end
    return lo, hi, gap, at
end

"""Box of a group in `O(|g|)`: the colatitude ends, and the largest circular gap in longitude."""
function _group_box(θv::Vector{T}, φv::Vector{T}, g::Vector{Int},
                    ws::_GapWorkspace{T}) where {T}
    θlo, θhi, _, _ = _max_gap(θv, g, ws)
    polar_tol = sqrt(eps(T))
    tn = θlo <= polar_tol
    ts = θhi >= T(π) - polar_tol
    (tn || ts) && return SphereBox{T}(θlo, θhi, zero(T), 2 * T(π), tn, ts)
    φlo, φhi, gap, at = _max_gap(φv, g, ws)
    # The wrap-around gap closes the circle; whichever gap is larger is the one excluded from the
    # arc, and the arc then starts at that gap's upper edge.
    wrap = 2 * T(π) - φhi + φlo
    start = at
    if wrap >= gap
        gap = wrap
        start = φlo
    end
    extent = 2 * T(π) - gap
    extent <= 0 && return SphereBox{T}(θlo, θhi, φlo, zero(T), false, false)
    extent >= 2 * T(π) && return SphereBox{T}(θlo, θhi, zero(T), 2 * T(π), false, false)
    return SphereBox{T}(θlo, θhi, start, extent, false, false)
end

"""
Split a group at its largest colatitude or longitude gap, whichever gap subtends more area.
Returns `nothing` when the group has no interior gap worth cutting.
"""
function _best_gap_split(θv::Vector{T}, φv::Vector{T}, g::Vector{Int},
                         ws::_GapWorkspace{T}) where {T}
    θlo, θhi, gapθ, atθ = _max_gap(θv, g, ws)
    _, _, gapφ, atφ = _max_gap(φv, g, ws)
    # Compare gaps by the angular width they remove, longitude weighted by ring radius.
    θmid = (θlo + θhi) / 2
    wφ = gapφ * sin(clamp(θmid, sqrt(eps(T)), T(π) - sqrt(eps(T))))
    if gapθ >= wφ
        gapθ > 0 || return nothing
        return _split_below(θv, g, atθ)
    else
        gapφ > 0 || return nothing
        return _split_below(φv, g, atφ)
    end
end

"""Partition `g` by `v[i] < thresh`, in one pass. `nothing` if either side would be empty."""
function _split_below(v::Vector{T}, g::Vector{Int}, thresh::T) where {T}
    na = 0
    @inbounds for i in g
        v[i] < thresh && (na += 1)
    end
    (na == 0 || na == length(g)) && return nothing
    a = Vector{Int}(undef, na)
    b = Vector{Int}(undef, length(g) - na)
    ia = ib = 0
    @inbounds for i in g
        if v[i] < thresh
            a[ia += 1] = i
        else
            b[ib += 1] = i
        end
    end
    return a, b
end
