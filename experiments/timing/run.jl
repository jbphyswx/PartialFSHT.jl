"""
Carron stretch vs full-sphere NUFSHT on a 20° cap (plan performance contract).

Times plan construction and `mul!` with FINUFFT. Direct is only timed where
`O(M L²)` is cheap. Prints as it goes.
"""

using PartialFSHT: PartialFSHT
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra
using Random: Random

const θSTAR = 20 * π / 180
const TOL = 1e-8
const N_APPLY = 3

function _cap_points(n, θstar)
    return θstar .* sqrt.(rand(n)), 2π .* rand(n)
end

function _band_points(n, halfwidth)
    return (π / 2) .+ halfwidth .* (2 .* rand(n) .- 1), 2π .* rand(n)
end

function _min_apply(plan, alm; n = N_APPLY)
    f = plan * alm
    ts = Float64[]
    for _ in 1:n
        push!(ts, @elapsed LinearAlgebra.mul!(f, plan, alm))
    end
    return minimum(ts), f
end

function _row(; lmax, M, method, setup, apply, rel = NaN, extra = "")
    return (lmax = lmax, M = M, method = method, setup_s = setup, apply_s = apply,
            rel = rel, extra = extra)
end

function run_cap!(rows, lmax; time_direct, time_nufsht)
    Random.seed!(0)
    sp = PartialFSHT.stretch_params(θSTAR, lmax, TOL)
    M = max(64, Int(round(4 * (lmax * θSTAR)^2)))
    θ, φ = _cap_points(M, θSTAR)
    cap = PartialFSHT.bounding_cap(θ, φ)
    fsky = (1 - cos(cap.θstar)) / 2
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    println()
    println("=== cap  θ*=20°  L=$lmax  M=$M  Lmax=$(sp.Lmax)  fsky=$(round(fsky; sigdigits=3))  εapo=$(round(sp.εapo; sigdigits=3)) ===")
    flush(stdout)

    fD = nothing
    if time_direct
        tsetup = @elapsed planD = PartialFSHT.Plan(θ, φ; lmax, method = PartialFSHT.DirectAlgorithm())
        tapply, fD = _min_apply(planD, alm)
        rel = 0.0
        println("  Direct   setup=$(round(tsetup; sigdigits=3))s  apply=$(round(tapply; sigdigits=3))s")
        flush(stdout)
        push!(rows, _row(; lmax, M, method = "direct", setup = tsetup, apply = tapply, rel))
    end

    auto = PartialFSHT.select_method(θ, φ, lmax, TOL, PartialFSHT.AutoAlgorithm(), 1, 0)
    println("  Auto     $(nameof(typeof(auto)))")
    flush(stdout)

    tsetup = @elapsed planS = PartialFSHT.Plan(θ, φ; lmax, method = PartialFSHT.StretchAlgorithm(),
                                               nufft = PartialFSHT.FINUFFTBackend(), tol = TOL)
    tapply, fS = _min_apply(planS, alm)
    relS = fD === nothing ? NaN : LinearAlgebra.norm(fS .- fD) / LinearAlgebra.norm(fD)
    println("  Stretch  setup=$(round(tsetup; sigdigits=3))s  apply=$(round(tapply; sigdigits=3))s  vsDirect=$(relS)")
    flush(stdout)
    push!(rows, _row(; lmax, M, method = "stretch", setup = tsetup, apply = tapply, rel = relS,
                     extra = "Lmax=$(planS.leaf.Lmax)"))

    if time_nufsht
        tsetup = @elapsed planN = PartialFSHT.Plan(θ, φ; lmax, method = PartialFSHT.NUFSHTAlgorithm(),
                                                   tol = TOL)
        tapply, fN = _min_apply(planN, alm)
        relN = fD === nothing ? LinearAlgebra.norm(fN .- fS) / LinearAlgebra.norm(fS) :
               LinearAlgebra.norm(fN .- fD) / LinearAlgebra.norm(fD)
        vs = fD === nothing ? "vsStretch" : "vsDirect"
        println("  NUFSHT   setup=$(round(tsetup; sigdigits=3))s  apply=$(round(tapply; sigdigits=3))s  $vs=$(relN)")
        flush(stdout)
        push!(rows, _row(; lmax, M, method = "nufsht", setup = tsetup, apply = tapply, rel = relN))
        if fD === nothing
            println("  stretch/nufsht apply ratio = $(round(rows[end-1].apply_s / tapply; sigdigits=3))  ( <1 means stretch faster )")
        else
            tN = tapply
            tS = rows[end-1].apply_s
            println("  stretch/nufsht apply ratio = $(round(tS / tN; sigdigits=3))")
        end
        flush(stdout)
    end
    return rows
end

function run_band!(rows, lmax)
    Random.seed!(1)
    half = 8 * π / 180
    M = max(64, Int(round(4 * (lmax * 2half)^2)))
    θ, φ = _band_points(M, half)
    cap = PartialFSHT.bounding_cap(θ, φ)
    println()
    println("=== equatorial band ±8°  L=$lmax  M=$M  bounding θ*=$(round(cap.θstar; sigdigits=3)) ===")
    flush(stdout)
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    auto = PartialFSHT.select_method(θ, φ, lmax, TOL, PartialFSHT.AutoAlgorithm(), 1, 0)
    println("  Auto     $(nameof(typeof(auto)))")
    flush(stdout)

    tsetup = @elapsed planC = PartialFSHT.Plan(θ, φ; lmax, method = PartialFSHT.ClusterAlgorithm(),
                                               nufft = PartialFSHT.FINUFFTBackend(), tol = TOL)
    tC, fC = _min_apply(planC, alm)
    ng = length(planC.leaf.groups)
    println("  cluster          setup=$(round(tsetup; sigdigits=3))s  apply=$(round(tC; sigdigits=3))s  k=$ng")
    flush(stdout)
    push!(rows, _row(; lmax, M, method = "band-cluster", setup = tsetup, apply = tC,
                     extra = "k=$ng"))

    tsetup = @elapsed planN = PartialFSHT.Plan(θ, φ; lmax, method = PartialFSHT.NUFSHTAlgorithm(),
                                               tol = TOL)
    tN, fN = _min_apply(planN, alm)
    rel = LinearAlgebra.norm(fC .- fN) / LinearAlgebra.norm(fN)
    println("  NUFSHT           setup=$(round(tsetup; sigdigits=3))s  apply=$(round(tN; sigdigits=3))s  vsCluster=$rel")
    println("  cluster/nufsht apply ratio = $(round(tC / tN; sigdigits=3))  ( <1 means cluster faster )")
    flush(stdout)
    push!(rows, _row(; lmax, M, method = "band-nufsht", setup = tsetup, apply = tN, rel))
    return rows
end

function main()
    println("PartialFSHT timing  (FINUFFT, tol=$TOL, nthreads=$(Threads.nthreads()))")
    println("cap: θ*=20°, M ≈ 4(L θ*)²")
    flush(stdout)
    rows = Any[]
    run_cap!(rows, 64; time_direct = true, time_nufsht = true)
    run_cap!(rows, 128; time_direct = true, time_nufsht = true)
    run_cap!(rows, 256; time_direct = false, time_nufsht = true)
    run_cap!(rows, 512; time_direct = false, time_nufsht = true)
    run_band!(rows, 256)
    run_cap!(rows, 1024; time_direct = false, time_nufsht = true)
    run_cap!(rows, 2048; time_direct = false, time_nufsht = true)
    println()
    println("done.")
    return rows
end

main()
