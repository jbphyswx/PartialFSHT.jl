# Planner cost model.
#
# Costs are *counted* operations — retained (ℓ,m,ring) triples, FFT butterflies, NUFFT spreading
# writes — scaled by per-unit constants measured once by `benchmark/calibrate.jl`. Counting rather
# than estimating means there are no tuned thresholds, and it makes a complexity claim testable by
# assertion instead of by stopwatch: `counted_ring_work` and `ring_work(engine)` count the same way,
# and a test asserts they agree exactly.

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
overwritten via [`set_cost_constants!`](@ref). Defaults are the measured *ratios* from
`benchmark/calibrate.jl`; [`cost_calibrated`](@ref) reports whether they were measured on this
machine.

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

# Defaults are the *ratios* measured by `benchmark/calibrate.jl` on a 20° cap at tol = 1e-8,
# normalised to the ring unit (1.86 ns/triple there). Only ratios matter, since `select_method`
# compares costs rather than predicting wall-clock, so these are portable in a way absolute
# timings would not be. `cost_calibrated()` still reports `false` until measured on this machine.
const _COST = CostConstants(1.0, 0.394, 0.334, 5.18, false)

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
    counted_ring_work(θ_rings, lmax, tol) -> (triples, mmax)

Retained `(ℓ, m, ring)` triples for a ring set, and the largest order any ring keeps, without
building the engine. One pass of [`polar_mmax`](@ref) feeds both, and the retained ranges come from
the nested-interval walk in [`retained_triples`](@ref) rather than from a `count` per order — the
latter is `O(lmax · nθ)` inside a function the planner calls several times per plan.
"""
function counted_ring_work(θ_rings::AbstractVector, lmax::Integer, tol::Real)
    L = Int(lmax)
    n = length(θ_rings)
    mlim = Vector{Int}(undef, n)
    mx = 0
    @inbounds for j in 1:n
        v = polar_mmax(θ_rings[j], L, tol)
        mlim[j] = v
        mx = max(mx, v)
    end
    return retained_triples(mlim, L), mx
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
    # A degenerate geometry is a modelling answer ("use a direct evaluation"), not an exception, so
    # ask for it as a query. Catching `ArgumentError` here would also swallow a genuine bad
    # argument and silently report the stretch leaf as infinitely expensive.
    geom = try_stretch_geometry(box.θlo, box.θhi, L, tol)
    geom === nothing && return Inf
    # Mirror StretchLeaf's own sizing exactly; costing a grid the leaf would not build makes
    # the model wrong in the one place it is consulted.
    Kθ = geom.K + _K_MARGIN
    if geom.kind === :interior
        nθ = next_smooth(2 * Kθ + 1)
        nθ_half = nθ
    else
        nθ_half = next_smooth(Kθ + 1)
        nθ = 2 * nθ_half
    end
    θr, _ = stretch_rings(geom, nθ_half)
    issorted(θr) || reverse!(θr)
    rings, mmax = counted_ring_work(θr, L, tol)
    nφ = next_smooth_even(2 * mmax + 1)
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

Partitioning is always attempted, because [`partition_boxes`](@ref) keeps a cut only when counted
cost falls and the search costs `O(maxboxes · n)` without sorting — a compact patch and a band both
come back as a single box.
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
    # a compact patch and a band both come back as a single box. The search sorts once and then
    # costs `O(maxboxes · n)`, and it already returns each group's box and cost, so nothing here
    # re-derives a bounding box per group.
    costfn = (b, np) -> cost_stretch(b, np, L, ntrans, tol)
    groups, _, gcosts = partition_boxes_costed(θ, φ, costfn)
    if length(groups) > 1
        cm = sum(gcosts)
        cm < bestc && ((best, bestc) = (MultiBoxAlgorithm(), cm))
    end

    cn < bestc && (best = NUFSHTAlgorithm())
    return best
end
