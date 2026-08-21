function _rel_adjoint(plan, alm, f)
    Ax = plan * alm
    Aty = plan' * f
    lhs = LinearAlgebra.dot(Ax, f)
    rhs = LinearAlgebra.dot(alm, Aty)
    return abs(lhs - rhs) / max(abs(lhs), eps(Float64))
end

Test.@testset "Direct operator" begin
    Random.seed!(3)
    lmax = 6
    n = 40
    θ = 0.4 .* rand(n)
    φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    Test.@test PartialFSHT.algorithm(plan) isa PartialFSHT.DirectAlgorithm

    alm = PartialFSHT.zeros_alm(lmax)
    alm[PartialFSHT.sph_mode(0, 0)] = 1
    f = plan * alm
    Test.@test all(isapprox.(f, PartialFSHT.ylm.(0, 0, θ, φ); atol = 1e-14))

    alm[PartialFSHT.sph_mode(1, 0)] = 2 + 0im
    f = similar(f)
    LinearAlgebra.mul!(f, plan, alm)
    ref = PartialFSHT.ylm.(0, 0, θ, φ) .+ 2 .* PartialFSHT.ylm.(1, 0, θ, φ)
    Test.@test maximum(abs.(f .- ref)) < 1e-13

    almR = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(plan, almR, fR) < 1e-12

    Test.@testset "ntrans batch" begin
        B = 3
        planB = PartialFSHT.Plan(θ, φ; lmax = lmax, ntrans = B, method = PartialFSHT.DirectAlgorithm())
        almB = randn(ComplexF64, PartialFSHT.coeff_size(lmax)..., B)
        fB = planB * almB
        for b in 1:B
            fb = plan * almB[:, :, b]
            Test.@test maximum(abs.(fB[:, b] .- fb)) < 1e-13
        end
    end
end
