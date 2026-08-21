Test.@testset "NUFSHT full-sphere leaf" begin
    Test.@test PartialFSHT._ext_loaded(:PartialFSHTNUFSHTExt)
    Random.seed!(14)
    lmax = 6
    pts = FastSphericalHarmonics.sph_points(lmax + 1)
    θ = vec([θ for θ in pts[1], φ in pts[2]])
    φ = vec([φ for θ in pts[1], φ in pts[2]])
    planN = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.NUFSHTAlgorithm(),
                             tol = 1e-10)
    planD = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    Test.@test PartialFSHT.algorithm(planN) isa PartialFSHT.NUFSHTAlgorithm

    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fN = planN * alm
    fD = planD * alm
    Test.@test LinearAlgebra.norm(fN .- fD) / LinearAlgebra.norm(fD) < 1e-8

    fR = randn(ComplexF64, length(θ))
    Test.@test _rel_adjoint(planN, alm, fR) < 1e-10

    Test.@testset "spin-1 vs Direct" begin
        s = 1
        planNs = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = s,
                                  method = PartialFSHT.NUFSHTAlgorithm(), tol = 1e-10)
        planDs = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = s,
                                  method = PartialFSHT.DirectAlgorithm())
        almS = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
        fNs = planNs * almS
        fDs = planDs * almS
        Test.@test LinearAlgebra.norm(fNs .- fDs) / LinearAlgebra.norm(fDs) < 1e-8
        fRs = randn(ComplexF64, length(θ))
        Test.@test _rel_adjoint(planNs, almS, fRs) < 1e-10

        alm10 = PartialFSHT.zeros_alm(lmax)
        alm10[PartialFSHT.spin_coeff_index(1, 0, lmax)] = 1
        ref = PartialFSHT.sYlm.(1, 1, 0, θ, φ)
        Test.@test LinearAlgebra.norm((planNs * alm10) .- ref) / LinearAlgebra.norm(ref) < 1e-8
    end

    Test.@testset "ntrans batch" begin
        B = 2
        planB = PartialFSHT.Plan(θ, φ; lmax = lmax, ntrans = B,
                                 method = PartialFSHT.NUFSHTAlgorithm(), tol = 1e-10)
        almB = randn(ComplexF64, PartialFSHT.coeff_size(lmax)..., B)
        fB = planB * almB
        for b in 1:B
            fb = planN * almB[:, :, b]
            Test.@test LinearAlgebra.norm(fB[:, b] .- fb) / LinearAlgebra.norm(fb) < 1e-8
        end
    end
end

Test.@testset "Auto: compact cap uses stretch while thin d fits" begin
    Random.seed!(0)
    θstar = 20 * π / 180
    tol = 1e-8
    for (L, expect) in ((128, PartialFSHT.StretchAlgorithm),
                        (512, PartialFSHT.StretchAlgorithm),
                        (1024, PartialFSHT.NUFSHTAlgorithm))
        n = 4000
        θc = θstar .* sqrt.(rand(n))
        φc = 2π .* rand(n)
        chosen = PartialFSHT.select_method(θc, φc, L, tol, PartialFSHT.AutoAlgorithm(), 1, 0)
        Test.@test chosen isa expect
    end
end

Test.@testset "Auto: equatorial band uses NUFSHT; two islands use cluster" begin
    Random.seed!(2)
    half = 8 * π / 180
    n = 4000
    θb = (π / 2) .+ half .* (2 .* rand(n) .- 1)
    φb = 2π .* rand(n)
    chosen = PartialFSHT.select_method(θb, φb, 64, 1e-8, PartialFSHT.AutoAlgorithm(), 1, 0)
    Test.@test chosen isa PartialFSHT.NUFSHTAlgorithm

    n1 = 1500
    θi = vcat(0.2 .* sqrt.(rand(n1)), π .- 0.2 .* sqrt.(rand(n1)))
    φi = 2π .* rand(2n1)
    choseni = PartialFSHT.select_method(θi, φi, 64, 1e-8, PartialFSHT.AutoAlgorithm(), 1, 0)
    Test.@test choseni isa PartialFSHT.ClusterAlgorithm

    pts = FastSphericalHarmonics.sph_points(9)
    θf = vec([θ for θ in pts[1], φ in pts[2]])
    φf = vec([φ for θ in pts[1], φ in pts[2]])
    chosenf = PartialFSHT.select_method(θf, φf, 8, 1e-8, PartialFSHT.AutoAlgorithm(), 1, 0)
    Test.@test chosenf isa PartialFSHT.NUFSHTAlgorithm
end
