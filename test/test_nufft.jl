Test.@testset "NUFFT backends vs DirectSum" begin
    Random.seed!(13)
    lmax = 6
    n = 24
    θ = 0.18 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    kwargs = (lmax = lmax, method = PartialFSHT.StretchAlgorithm(), tol = 1e-6)
    plan0 = PartialFSHT.Plan(θ, φ; kwargs..., nufft = PartialFSHT.DirectSumBackend())
    f0 = plan0 * alm

    Test.@testset "FINUFFT" begin
        planF = PartialFSHT.Plan(θ, φ; kwargs..., nufft = PartialFSHT.FINUFFTBackend())
        fF = planF * alm
        Test.@test LinearAlgebra.norm(fF .- f0) / LinearAlgebra.norm(f0) < 1e-4
    end

    Test.@testset "NonuniformFFTs" begin
        planN = PartialFSHT.Plan(θ, φ; kwargs..., nufft = PartialFSHT.NonuniformFFTsBackend())
        fN = planN * alm
        Test.@test LinearAlgebra.norm(fN .- f0) / LinearAlgebra.norm(f0) < 5e-3
    end
end
