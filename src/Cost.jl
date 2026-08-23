"""
Planner cost model.

Costs are *counted* operations — retained `(ℓ,m,ring)` triples, FFT butterflies, NUFFT
spreading writes — scaled by per-unit constants measured once by `benchmark/calibrate.jl`. The
previous model estimated the ring pass as `L²·θ⋆`, which is a factor `L` short of the truth,
and then needed a cascade of tuned overrides (`fsky ≥ 0.3 → Inf`, `clusters ≤ 3 && cover < 0.5`,
`thin_ok`, `steal`) to reach the right answer anyway. Counting instead of guessing removes the
overrides, and makes a complexity claim testable without a stopwatch.
"""

"""
    AbstractAlgorithm

Which apply kernel a [`Plan`](@ref) builds. `mul!` dispatches on the leaf the constructor
stored, not on these tags.
"""
abstract type AbstractAlgorithm end

"""Cost model chooses."""
struct AutoAlgorithm <: AbstractAlgorithm end

"""Scaled associated-Legendre evaluation per point; the oracle, `O(M L²)`."""
struct DirectAlgorithm <: AbstractAlgorithm end

"""Stretched colatitude interval on one box — the workhorse."""
struct StretchAlgorithm <: AbstractAlgorithm end

"""One stretched leaf per box of a partitioned point set."""
struct MultiBoxAlgorithm <: AbstractAlgorithm end

"""Full-sphere leaf via NUFSHT's public plans. Requires `using NUFSHT`."""
struct NUFSHTAlgorithm <: AbstractAlgorithm end

"""
    CostConstants

Per-unit costs in arbitrary consistent units, measured by `benchmark/calibrate.jl` and
overwritten via [`set_cost_constants!`](@ref). Defaults are deliberately crude — a planner
running on defaults should be treated as uncalibrated, not as tuned.

- `ring` — one retained `(ℓ, m, ring)` triple of the ALF pass
- `fft` — one `n log n` butterfly unit
- `spread` — one NUFFT kernel write per point per dimension
- `ft_conn` — one `L² log² L` unit of the FastTransforms connection used by a NUFSHT leaf
"""
mutable struct CostConstants
    ring::Float64
    fft::Float64
    spread::Float64
    ft_conn::Float64
    calibrated::Bool
end

const _COST = CostConstants(1.0, 1.0, 1.0, 4.0, false)

"""
    set_cost_constants!(; ring, fft, spread, ft_conn) -> CostConstants

Install measured per-unit costs. Marks the model calibrated so [`cost_calibrated`](@ref)
can report whether a planner decision rests on measurement.
"""
function set_cost_constants!(; ring = _COST.ring, fft = _COST.fft,
                             spread = _COST.spread, ft_conn = _COST.ft_conn)
    _COST.ring = Float64(ring)
    _COST.fft = Float64(fft)
    _COST.spread = Float64(spread)
    _COST.ft_conn = Float64(ft_conn)
    _COST.calibrated = true
    return _COST
end

"""    cost_calibrated() -> Bool"""
cost_calibrated() = _COST.calibrated

_fftunits(n::Integer) = n <= 1 ? 0.0 : Float64(n) * log2(Float64(n))

"""
    counted_ring_work(θ_rings, lmax, tol) -> Int

Retained `(ℓ, m, ring)` triples for a ring set, without building the engine: the polar-optimised
work count the ALF pass will actually do.
"""
function counted_ring_work(θ_rings::AbstractVector, lmax::Integer, tol::Real)
    L = Int(lmax)
    mlim = [polar_mmax(t, L, tol) for t in θ_rings]
    w = 0
    @inbounds for m in 0:L
        nj = count(>=(m), mlim)
        nj == 0 && continue
        w += nj * (L - m + 1)
    end
    return w
end

"""
    cost_direct(npts, lmax, ntrans) -> Float64
"""
cost_direct(npts::Integer, lmax::Integer, ntrans::Integer) =
    _COST.ring * Float64(npts) * (Int(lmax) + 1)^2 * Int(ntrans) / 2

"""
    cost_stretch(box, npts, lmax, ntrans, tol) -> Float64

Counted cost of one stretched-interval leaf: ALF triples, the longitude and 2-D FFTs, and the
NUFFT spreading. Returns `Inf` when the geometry is degenerate at this bandlimit, which is the
model's way of saying "use a direct evaluation", not a tuned exclusion.
"""
function cost_stretch(box::SphereBox, npts::Integer, lmax::Integer, ntrans::Integer,
                      tol::Real)
    L = Int(lmax)
    B = Int(ntrans)
    geom = try
        stretch_geometry(box.θlo, box.θhi, L, tol)
    catch err
        err isa ArgumentError && return Inf
        rethrow()
    end
    nθ_half = geom.kind === :interior ? next_smooth(2 * geom.K + 1) : geom.K + 1
    nθ = geom.kind === :interior ? nθ_half : next_smooth(2 * nθ_half)
    θr, _ = stretch_rings(geom, geom.kind === :interior ? nθ : nθ ÷ 2)
    rings = counted_ring_work(sort(θr), L, tol)
    mmax = maximum(polar_mmax(t, L, tol) for t in θr)
    nφ = next_smooth(2 * mmax + 1)
    fft = _fftunits(nφ) * length(θr) + _fftunits(nθ * nφ)
    spread = Float64(npts) * (2 * ceil(abs(log10(max(float(tol), 1e-16)))) + 2)^2
    return B * (_COST.ring * rings + _COST.fft * fft + _COST.spread * spread)
end

"""
    cost_nufsht(npts, lmax, ntrans, tol) -> Float64

Counted cost of a full-sphere NUFSHT apply: the FastTransforms connection, the full-torus
oversampled FFT, and spreading over the whole torus. Independent of the patch, which is exactly
why a partial-sky leaf can beat it.
"""
function cost_nufsht(npts::Integer, lmax::Integer, ntrans::Integer, tol::Real)
    L = Int(lmax)
    B = Int(ntrans)
    K = Float64(L + 1)^2
    conn = _COST.ft_conn * K * log2(Float64(L) + 2)^2
    grid = 2 * (2L + 2)
    fft = _fftunits(grid * grid)
    spread = Float64(npts) * (2 * ceil(abs(log10(max(float(tol), 1e-16)))) + 2)^2
    return B * (conn + _COST.fft * fft + _COST.spread * spread)
end

_nufsht_ext_loaded() = Base.get_extension(@__MODULE__, :PartialFSHTNUFSHTExt) !== nothing
_omt_loaded() = Base.get_extension(@__MODULE__, :PartialFSHTOhMyThreadsExt) !== nothing

"""
    select_method(θ, φ, lmax, tol, requested, ntrans, spin) -> AbstractAlgorithm

Pick the cheapest available leaf by counted cost. No thresholds: a compact patch wins on the
stretch leaf because its ring count is small, a band wins because its colatitude interval is
thin, a near-full sky loses to NUFSHT because the counted work converges to NUFSHT's own.

Partitioning is only attempted when a single box leaves a large fraction of the sphere unused,
so an ordinary patch never pays for a partition search it cannot benefit from — the previous
planner ran spherical k-means, up to 32 restarts of 25 iterations, on *every* construction just
to decide.
"""
function select_method(θ, φ, lmax::Integer, tol::Real, requested::AbstractAlgorithm,
                       ntrans::Integer, spin::Integer = 0)
    requested isa AutoAlgorithm || return requested
    n = length(θ)
    L = Int(lmax)
    box = bounding_box(θ, φ)
    cd = cost_direct(n, L, ntrans)
    cs = cost_stretch(box, n, L, ntrans, tol)
    cn = _nufsht_ext_loaded() ? cost_nufsht(n, L, ntrans, tol) : Inf

    best = DirectAlgorithm()
    bestc = cd
    cs < bestc && ((best, bestc) = (StretchAlgorithm(), cs))

    # `partition_boxes` keeps a cut only when counted cost falls, so it is safe to always try:
    # a compact patch and a band both come back as a single box, and the search is O(n log n)
    # per candidate rather than the k-means restarts the previous planner ran.
    costfn = (b, np) -> cost_stretch(b, np, L, ntrans, tol)
    groups = partition_boxes(θ, φ, costfn)
    if length(groups) > 1
        θh, φh = _host(θ), _host(φ)
        cm = sum(cost_stretch(bounding_box(view(θh, g), view(φh, g)), length(g),
                              L, ntrans, tol) for g in groups)
        cm < bestc && ((best, bestc) = (MultiBoxAlgorithm(), cm))
    end

    cn < bestc && (best = NUFSHTAlgorithm())
    return best
end
