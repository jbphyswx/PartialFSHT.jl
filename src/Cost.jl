"""
Planner cost model. Units are dimensionless flop estimates, used only to rank methods.

Algorithms are types, not symbols — the same pattern as NUFFT backends. `mul!` dispatches on
the leaf (`DirectScratch`, `StretchLeaf`, …); `select_method` returns one of these tags at
plan construction.
"""

"""
    AbstractAlgorithm

Tag for which apply kernel a [`Plan`](@ref) should build. Pass to `Plan(...; method = ...)`.
`mul!` does not branch on these tags — it dispatches on the leaf type the constructor stored.
"""
abstract type AbstractAlgorithm end

"""Auto: cost model picks Direct, stretch, clustered caps, or NUFSHT."""
struct AutoAlgorithm <: AbstractAlgorithm end

"""Holmes–Featherstone associated-Legendre oracle, `O(M L²)`."""
struct DirectAlgorithm <: AbstractAlgorithm end

"""Carron–Reinecke polar stretch (compact bounding cap)."""
struct StretchAlgorithm <: AbstractAlgorithm end

"""Partition the point set; one stretch leaf per cluster."""
struct ClusterAlgorithm <: AbstractAlgorithm end

"""Full-sphere leaf via NUFSHT public API (scalar or spin). Requires `using NUFSHT`."""
struct NUFSHTAlgorithm <: AbstractAlgorithm end

algorithm(::DirectScratch) = DirectAlgorithm()
algorithm(::StretchLeaf) = StretchAlgorithm()
algorithm(::ClusterLeaf) = ClusterAlgorithm()

function cost_direct(npts::Integer, lmax::Integer, ntrans::Integer)
    return Float64(npts) * (Int(lmax) + 1)^2 * Int(ntrans)
end

function cost_stretch(θstar::Real, npts::Integer, lmax::Integer, ntrans::Integer, ε::Real)
    sp = stretch_params(θstar, lmax, ε)
    L = Int(lmax)
    Lm = sp.Lmax
    M = Int(npts)
    B = Int(ntrans)
    rings = L * L * max(Float64(θstar), 0.05) * B
    nufft = (Lm + 1)^2 * max(log2(Float64(Lm) + 2), 1) * B + M * max(abs(log(Float64(ε))), 1) * B
    mpol = max(polar_mmax(sp.θmax, L, ε), 0)
    wigner = L * L * Float64(max(mpol, 1)) * B
    return rings + nufft + wigner
end

function cost_nufsht(npts::Integer, lmax::Integer, ntrans::Integer, ε::Real)
    L = Int(lmax)
    M = Int(npts)
    B = Int(ntrans)
    return (L + 1)^2 * (log2(Float64(L) + 2)^2) * B + M * max(abs(log(Float64(ε))), 1) * B
end

function _nufsht_ext_loaded()
    return Base.get_extension(@__MODULE__, :PartialFSHTNUFSHTExt) !== nothing
end

function _omt_loaded()
    return Base.get_extension(@__MODULE__, :PartialFSHTOhMyThreadsExt) !== nothing
end

function select_method(θ, φ, lmax::Integer, tol::Real, requested::AbstractAlgorithm,
                       ntrans::Integer, spin::Integer = 0)
    requested isa AutoAlgorithm || return requested
    n = length(θ)
    cap = bounding_cap(θ, φ)
    cd = cost_direct(n, lmax, ntrans)
    cs = cost_stretch(cap.θstar, n, lmax, ntrans, tol)
    hull_fsky = fsky_cap(cap.θstar)
    # One-cap stretch is for compact patches. A near-full bounding hull (bands,
    # islands on a large sphere) is clustered caps or NUFSHT — stretch is both
    # slow and inaccurate when θ* ~ π.
    hull_fsky >= 0.3 && (cs = Inf)
    groups = cluster_points(θ, φ)
    cc = 0.0
    cover = 0.0
    for g in groups
        cg = bounding_cap(view(θ, g), view(φ, g))
        cc += cost_stretch(cg.θstar, length(g), lmax, ntrans, tol)
        cover += fsky_cap(cg.θstar)
    end
    cn = _nufsht_ext_loaded() ? cost_nufsht(n, lmax, ntrans, tol) : Inf
    sp = stretch_params(cap.θstar, lmax, tol)
    mpol = max(polar_mmax(sp.θmax, Int(lmax), tol), abs(Int(spin)))
    # Streamed d(β) is O(L² mpol) but still slower than NUFSHT when the packed
    # cache does not fit (fill+GEMV vs FastTransforms). Keep compact stretch
    # only while thin d is packed.
    thin_ok = _thin_d_cache_nbytes(Int(lmax), mpol, Float64) <= _WIGNER_THIN_D_BYTES
    # A few compact islands: cluster. A many-cluster band still loses to NUFSHT.
    clusters_ok = 1 < length(groups) <= 3 && cover < 0.5
    best = DirectAlgorithm()
    bestc = cd
    if cs < bestc
        best = StretchAlgorithm(); bestc = cs
    end
    if length(groups) > 1 && cc < bestc
        best = ClusterAlgorithm(); bestc = cc
    end
    # Raw L² log² L undercounts NUFSHT vs BLAS. Compact stretch and compact
    # clustered masks keep their leaf; large-sky covering partitions go to NUFSHT.
    if cn < bestc
        steal = true
        if best isa StretchAlgorithm && hull_fsky < 0.3 && thin_ok
            steal = false
        elseif best isa ClusterAlgorithm && clusters_ok
            steal = false
        end
        steal && (best = NUFSHTAlgorithm())
    end
    return best
end
