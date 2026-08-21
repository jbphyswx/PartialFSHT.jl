Test.@testset "Carron stretch vs Direct" begin
    Random.seed!(5)
    lmax = 8
    n = 50
    θ = 0.2 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    planD = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend())
    planS = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.StretchAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
    Test.@test PartialFSHT.algorithm(planS) isa PartialFSHT.StretchAlgorithm

    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    fD = planD * alm
    fS = planS * alm
    rel = LinearAlgebra.norm(fD .- fS) / LinearAlgebra.norm(fD)
    Test.@test rel < 5e-3

    fR = randn(ComplexF64, n)
    Test.@test _rel_adjoint(planS, alm, fR) < 1e-10

    Test.@testset "window is 1 on the user cap" begin
        leaf = planS.leaf
        Θ1 = π * leaf.cap.θstar / leaf.θmax
        Test.@test maximum(leaf.Θ_pts) <= Θ1 + 1e-12
        oncap = leaf.Θ_rings .<= Θ1
        Test.@test all(isapprox.(leaf.w[oncap], 1; atol = 1e-14))
    end

    Test.@testset "ntrans batch" begin
        B = 3
        planB = PartialFSHT.Plan(θ, φ; lmax = lmax, ntrans = B,
                                 method = PartialFSHT.StretchAlgorithm(),
                                 nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
        almB = randn(ComplexF64, PartialFSHT.coeff_size(lmax)..., B)
        fB = planB * almB
        for b in 1:B
            fb = planS * almB[:, :, b]
            Test.@test maximum(abs.(fB[:, b] .- fb)) < 1e-12
        end
    end
end

Test.@testset "ring φ DFT via FFT matches brute sum" begin
    Random.seed!(21)
    lmax = 6
    nθ, nφ, B = 5, 9, 2  # nφ < 2lmax+1 aliases
    G = randn(ComplexF64, nθ, 2lmax + 1, B)
    F = zeros(ComplexF64, nθ, nφ, B)
    PartialFSHT.fourier_to_rings!(F, G, lmax)
    brute = zeros(ComplexF64, nθ, nφ, B)
    off = lmax + 1
    for b in 1:B, j in 1:nθ, k in 1:nφ
        φ = 2π * (k - 1) / nφ
        acc = 0.0im
        for m in -lmax:lmax
            acc += G[j, m + off, b] * cis(m * φ)
        end
        brute[j, k, b] = acc
    end
    Test.@test F ≈ brute atol = 1e-12

    G2 = zeros(ComplexF64, nθ, 2lmax + 1, B)
    PartialFSHT.rings_to_fourier!(G2, F, lmax)
    Gadj = zeros(ComplexF64, nθ, 2lmax + 1, B)
    for b in 1:B, j in 1:nθ, m in -lmax:lmax
        acc = 0.0im
        for k in 1:nφ
            φ = 2π * (k - 1) / nφ
            acc += F[j, k, b] * cis(-m * φ)
        end
        Gadj[j, m + off, b] = acc
    end
    Test.@test G2 ≈ Gadj atol = 1e-12
end
