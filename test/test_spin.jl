Test.@testset "spin-1 Direct" begin
    Random.seed!(8)
    lmax = 4
    n = 20
    θ = 0.4 .* rand(n) .+ 0.2
    φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = 1, method = PartialFSHT.DirectAlgorithm())
    alm = PartialFSHT.zeros_alm(lmax)
    alm[PartialFSHT.spin_coeff_index(1, 0, lmax)] = 1
    f = plan * alm
    ref = PartialFSHT.sYlm.(1, 1, 0, θ, φ)
    Test.@test maximum(abs.(f .- ref)) < 1e-10

    almR = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(plan, almR, fR) < 1e-11
end

Test.@testset "spin-1 stretch vs Direct" begin
    Random.seed!(5)
    lmax = 8
    n = 50
    θ = 0.2 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    planD = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = 1, method = PartialFSHT.DirectAlgorithm())
    planS = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = 1, method = PartialFSHT.StretchAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
    Test.@test PartialFSHT.algorithm(planS) isa PartialFSHT.StretchAlgorithm
    Test.@test planS.leaf.spin == 1

    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fD = planD * alm
    fS = planS * alm
    rel = LinearAlgebra.norm(fD .- fS) / LinearAlgebra.norm(fD)
    Test.@test rel < 5e-3

    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(planS, alm, fR) < 1e-10
end
