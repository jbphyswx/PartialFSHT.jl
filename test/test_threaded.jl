Test.@testset "OhMyThreads cluster apply" begin
    Random.seed!(6)
    lmax = 6
    n = 40
    θ1 = 0.15 .* sqrt.(rand(n ÷ 2))
    φ1 = 2π .* rand(n ÷ 2)
    θ2 = π .- 0.15 .* sqrt.(rand(n - n ÷ 2))
    φ2 = 2π .* rand(n - n ÷ 2)
    θ = vcat(θ1, θ2)
    φ = vcat(φ1, φ2)
    kwargs = (lmax = lmax, method = PartialFSHT.ClusterAlgorithm(),
              nufft = PartialFSHT.DirectSumBackend(), k = 2, tol = 1e-8)
    planS = PartialFSHT.Plan(θ, φ; kwargs...,
                             backend = ComputationalBackends.SerialBackend())
    planT = PartialFSHT.Plan(θ, φ; kwargs...,
                             backend = ComputationalBackends.ThreadedBackend())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    Test.@test LinearAlgebra.norm((planT * alm) .- (planS * alm)) < 1e-12
    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(planT, alm, fR) < 1e-10
end

Test.@testset "mul_nodeset serial" begin
    Random.seed!(3)
    lmax = 3
    θs = [0.2 .* rand(8), 0.3 .* rand(9) .+ 0.4]
    φs = [2π .* rand(length(θs[1])), 2π .* rand(length(θs[2]))]
    alms = [randn(ComplexF64, PartialFSHT.coeff_size(lmax)...) for _ in 1:2]
    fs = PartialFSHT.mul_nodeset(θs, φs, alms, lmax)
    for i in 1:2
        plan = PartialFSHT.Plan(θs[i], φs[i]; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
        Test.@test LinearAlgebra.norm(fs[i] .- (plan * alms[i])) < 1e-12
    end
end
