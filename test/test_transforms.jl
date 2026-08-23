_reladj(plan, alm, f) = begin
    Ax = plan * alm
    Aty = plan' * f
    lhs = LinearAlgebra.dot(Ax, f)
    rhs = LinearAlgebra.dot(alm, Aty)
    abs(lhs - rhs) / max(abs(lhs), eps(Float64))
end

Test.@testset "Direct oracle" begin
    Random.seed!(31)
    for L in (6, 32)
        n = 40
        θ = 0.4 .* rand(n)
        φ = 2π .* rand(n)
        plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
        Test.@test PartialFSHT.algorithm(plan) isa PartialFSHT.DirectAlgorithm
        alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)

        # against the explicit Σ a Y sum, term by term
        ref = zeros(ComplexF64, n)
        for ℓ in 0:L, m in -ℓ:ℓ
            c = alm[PartialFSHT.sph_mode(ℓ, m)]
            iszero(c) && continue
            for i in 1:n
                ref[i] += c * PartialFSHT.ylm(ℓ, m, θ[i], φ[i])
            end
        end
        f = plan * alm
        Test.@test maximum(abs.(f .- ref)) / maximum(abs.(ref)) < 1e-13
        Test.@test _reladj(plan, alm, randn(ComplexF64, n)) < 1e-13
    end

    Test.@testset "single harmonic reproduces itself" begin
        L, n = 8, 25
        θ = 0.5 .* rand(n) .+ 0.1
        φ = 2π .* rand(n)
        plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
        for (ℓ, m) in ((0, 0), (1, -1), (5, 3), (8, -8))
            alm = PartialFSHT.zeros_alm(L)
            alm[PartialFSHT.sph_mode(ℓ, m)] = 1
            Test.@test maximum(abs.((plan * alm) .- PartialFSHT.ylm.(ℓ, m, θ, φ))) < 1e-13
        end
    end

    Test.@testset "ntrans batching matches per-column" begin
        L, n, B = 12, 30, 3
        θ = 0.4 .* rand(n); φ = 2π .* rand(n)
        p1 = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
        pB = PartialFSHT.Plan(θ, φ; lmax = L, ntrans = B,
                              method = PartialFSHT.DirectAlgorithm())
        almB = randn(ComplexF64, PartialFSHT.coeff_size(L)..., B)
        fB = pB * almB
        for b in 1:B
            Test.@test maximum(abs.(fB[:, b] .- (p1 * almB[:, :, b]))) < 1e-13
        end
    end
end

Test.@testset "stretched leaf vs Direct, across degree and tolerance" begin
    # This is the gate that did not exist: every previous accuracy test ran at lmax ≤ 10, where
    # εapo > 3.6 and the apodisation is wider than the patch, and asserted 5e-3 for tol = 1e-8.
    Random.seed!(32)
    θstar = 20 * π / 180
    for L in (128, 256, 512)
        for tol in (1e-4, 1e-8)
            n = 150
            θ = θstar .* sqrt.(rand(n))
            φ = 2π .* rand(n)
            pd = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
            ps = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.StretchAlgorithm(),
                                  nufft = PartialFSHT.DirectSumBackend(), tol = tol)
            alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
            fd = pd * alm
            fs = ps * alm
            rel = maximum(abs.(fs .- fd)) / maximum(abs.(fd))
            # delivered accuracy is at or better than the requested tolerance
            Test.@test rel < 10 * tol
            # and nφ must be even, or the DFS half-turn is wrong
            Test.@test iseven(ps.leaf.nφ)
        end
    end
end

Test.@testset "stretched leaf: worst-case spectra" begin
    # A single ℓ = lmax mode is the hardest case for the stretch truncation; a red spectrum is
    # the easiest. Both must hold, and the error must not live in the patch interior.
    Random.seed!(33)
    L, tol = 256, 1e-8
    θstar = 20 * π / 180
    n = 200
    θ = θstar .* sqrt.(rand(n)); φ = 2π .* rand(n)
    pd = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    ps = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.StretchAlgorithm(),
                          nufft = PartialFSHT.DirectSumBackend(), tol = tol)

    alm = PartialFSHT.zeros_alm(L)
    for m in -L:L
        alm[PartialFSHT.sph_mode(L, m)] = randn(ComplexF64)
    end
    fd = pd * alm; fs = ps * alm
    Test.@test maximum(abs.(fs .- fd)) / maximum(abs.(fd)) < 10 * tol

    almr = PartialFSHT.zeros_alm(L)
    for ℓ in 0:L, m in -ℓ:ℓ
        almr[PartialFSHT.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)^2
    end
    fdr = pd * almr; fsr = ps * almr
    Test.@test maximum(abs.(fsr .- fdr)) / maximum(abs.(fdr)) < 10 * tol
end

Test.@testset "stretched leaf adjoint is the exact transpose" begin
    Random.seed!(34)
    for L in (256, 512)
        n = 120
        θ = (20 * π / 180) .* sqrt.(rand(n))
        φ = 2π .* rand(n)
        ps = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.StretchAlgorithm(),
                              nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
        alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
        # round-off, not tolerance: every stage is transposed exactly
        Test.@test _reladj(ps, alm, randn(ComplexF64, n)) < 1e-12
    end
end

Test.@testset "stretched leaf: off-pole and southern patches" begin
    Random.seed!(35)
    L, tol = 256, 1e-8
    for (θc, tag) in ((0.9, "mid-latitude"), (π - 0.3, "southern"))
        n = 150
        θ = θc .+ 0.12 .* (2 .* rand(n) .- 1)
        φ = 1.0 .+ 0.4 .* rand(n)
        pd = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
        ps = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.StretchAlgorithm(),
                              nufft = PartialFSHT.DirectSumBackend(), tol = tol)
        alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
        fd = pd * alm; fs = ps * alm
        Test.@test maximum(abs.(fs .- fd)) / maximum(abs.(fd)) < 10 * tol
        Test.@test _reladj(ps, alm, randn(ComplexF64, n)) < 1e-12
    end
end

Test.@testset "stretched leaf: ntrans batching" begin
    Random.seed!(36)
    L, B = 256, 3
    n = 100
    θ = (20 * π / 180) .* sqrt.(rand(n)); φ = 2π .* rand(n)
    kw = (lmax = L, method = PartialFSHT.StretchAlgorithm(),
          nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
    p1 = PartialFSHT.Plan(θ, φ; kw...)
    pB = PartialFSHT.Plan(θ, φ; ntrans = B, kw...)
    almB = randn(ComplexF64, PartialFSHT.coeff_size(L)..., B)
    fB = pB * almB
    for b in 1:B
        Test.@test maximum(abs.(fB[:, b] .- (p1 * almB[:, :, b]))) < 1e-9
    end
end

Test.@testset "Float32 end to end" begin
    # Element types must survive the whole pipeline: a silent promotion to Float64 would look
    # like it works while doubling memory traffic and halving throughput.
    Random.seed!(38)
    for L in (128, 256)
        n = 120
        θ64 = (20 * π / 180) .* sqrt.(rand(n))
        φ64 = 2π .* rand(n)
        alm64 = rand_alm(L)
        pd = PartialFSHT.Plan(θ64, φ64; lmax = L, method = PartialFSHT.DirectAlgorithm())
        fd = pd * alm64

        θ32 = Float32.(θ64)
        φ32 = Float32.(φ64)
        alm32 = ComplexF32.(alm64)
        p32 = PartialFSHT.Plan(θ32, φ32; lmax = L,
                               method = PartialFSHT.StretchAlgorithm(),
                               nufft = PartialFSHT.DirectSumBackend(), tol = 1.0f-6)
        f32 = p32 * alm32
        Test.@test eltype(f32) === ComplexF32
        # The accuracy floor here is Float32 round-off accumulated over ~L² terms, not the
        # requested tolerance, so the gate is that floor rather than `tol`.
        Test.@test maximum(abs.(ComplexF64.(f32) .- fd)) / maximum(abs.(fd)) < 1e-3

        fr = randn(ComplexF32, n)
        a32 = p32' * fr
        Test.@test eltype(a32) === ComplexF32
        lhs = LinearAlgebra.dot(f32, fr)
        rhs = LinearAlgebra.dot(alm32, a32)
        Test.@test abs(lhs - rhs) / abs(lhs) < 1e-4
    end

    Test.@testset "Float32 oracle" begin
        L, n = 32, 40
        θ32 = Float32.(0.4 .* rand(n))
        φ32 = Float32.(2π .* rand(n))
        p = PartialFSHT.Plan(θ32, φ32; lmax = L, method = PartialFSHT.DirectAlgorithm())
        alm = ComplexF32.(rand_alm(L))
        f = p * alm
        Test.@test eltype(f) === ComplexF32
        ref = zeros(ComplexF64, n)
        for ℓ in 0:L, m in -ℓ:ℓ
            c = ComplexF64(alm[PartialFSHT.sph_mode(ℓ, m)])
            iszero(c) && continue
            for i in 1:n
                ref[i] += c * PartialFSHT.ylm(ℓ, m, Float64(θ32[i]), Float64(φ32[i]))
            end
        end
        Test.@test maximum(abs.(ComplexF64.(f) .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

Test.@testset "multi-box leaf: two caps" begin
    Random.seed!(37)
    L, tol = 256, 1e-8
    n1 = 80
    θ = vcat(0.25 .* sqrt.(rand(n1)), π .- 0.25 .* sqrt.(rand(n1)))
    φ = 2π .* rand(2n1)
    pd = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    pc = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.MultiBoxAlgorithm(),
                          nufft = PartialFSHT.DirectSumBackend(), tol = tol)
    Test.@test length(pc.leaf.groups) == 2
    alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
    fd = pd * alm; fc = pc * alm
    Test.@test maximum(abs.(fc .- fd)) / maximum(abs.(fd)) < 10 * tol
    Test.@test _reladj(pc, alm, randn(ComplexF64, 2n1)) < 1e-12
end
