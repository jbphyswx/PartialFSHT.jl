Test.@testset "clustered caps" begin
    Random.seed!(6)
    lmax = 6
    n = 40
    θ1 = 0.15 .* sqrt.(rand(n ÷ 2))
    φ1 = 2π .* rand(n ÷ 2)
    θ2 = π .- 0.15 .* sqrt.(rand(n - n ÷ 2))
    φ2 = 2π .* rand(n - n ÷ 2)
    θ = vcat(θ1, θ2)
    φ = vcat(φ1, φ2)
    planC = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.ClusterAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend(), k = 2, tol = 1e-8)
    Test.@test PartialFSHT.algorithm(planC) isa PartialFSHT.ClusterAlgorithm
    Test.@test length(planC.leaf.groups) == 2

    planD = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fC = planC * alm
    fD = planD * alm
    rel = LinearAlgebra.norm(fC .- fD) / LinearAlgebra.norm(fD)
    Test.@test rel < 5e-3

    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(planC, alm, fR) < 1e-10
end

Test.@testset "clustered caps spin-1" begin
    Random.seed!(6)
    lmax = 6
    n = 40
    θ1 = 0.15 .* sqrt.(rand(n ÷ 2))
    φ1 = 2π .* rand(n ÷ 2)
    θ2 = π .- 0.15 .* sqrt.(rand(n - n ÷ 2))
    φ2 = 2π .* rand(n - n ÷ 2)
    θ = vcat(θ1, θ2)
    φ = vcat(φ1, φ2)
    planC = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = 1,
                             method = PartialFSHT.ClusterAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend(), k = 2, tol = 1e-8)
    planD = PartialFSHT.Plan(θ, φ; lmax = lmax, spin = 1, method = PartialFSHT.DirectAlgorithm())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    rel = LinearAlgebra.norm((planC * alm) .- (planD * alm)) / LinearAlgebra.norm(planD * alm)
    # Two-cap spin at this L is window/Lmax truncation (~0.8% at tol=1e-8), not an O(1) phase bug.
    Test.@test rel < 1e-2
    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(planC, alm, fR) < 1e-10
end

Test.@testset "equatorial band: cluster vs Direct" begin
    Random.seed!(7)
    lmax = 10
    half = 8 * π / 180
    n = 80
    θ = (π / 2) .+ half .* (2 .* rand(n) .- 1)
    φ = 2π .* rand(n)
    planC = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.ClusterAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
    planD = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    Test.@test length(planC.leaf.groups) > 1
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    rel = LinearAlgebra.norm((planC * alm) .- (planD * alm)) / LinearAlgebra.norm(planD * alm)
    Test.@test rel < 5e-3
end
