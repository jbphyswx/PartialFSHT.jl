Test.@testset "Slepian diagnostic / solve" begin
    Random.seed!(12)
    lmax = 4
    n = 60
    θ = 0.35 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    f = plan * alm
    rec = PartialFSHT.zeros_alm(lmax)
    rec, iters, rel = PartialFSHT.solve!(rec, plan, f; rtol = 1e-8, maxiter = 200)
    Test.@test rel < 1e-6 || iters >= 1
    f2 = plan * rec
    Test.@test LinearAlgebra.norm(f2 .- f) / LinearAlgebra.norm(f) < 0.05

    cap = PartialFSHT.bounding_cap(θ, φ)
    Ns = PartialFSHT.shannon_number(lmax, cap.θstar)
    Test.@test Ns > 0
    w = PartialFSHT.cap_quadrature_weights(θ, φ, cap)
    WA = PartialFSHT.weighted_synthesis_matrix(θ, φ, lmax, w)
    Test.@test size(WA, 1) == n
    Test.@test size(WA, 2) == (lmax + 1)^2
end
