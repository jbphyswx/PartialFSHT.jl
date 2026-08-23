Test.@testset "Wigner-d" begin
    β = 0.37
    Test.@testset "closed forms at ℓ = 1, 2" begin
        Test.@test PartialFSHT.wigner_d(1, 0, 0, β) ≈ cos(β)
        Test.@test PartialFSHT.wigner_d(1, 1, 0, β) ≈ -sin(β) / sqrt(2)
        Test.@test PartialFSHT.wigner_d(1, 0, 1, β) ≈ sin(β) / sqrt(2)
        Test.@test PartialFSHT.wigner_d(1, 1, 1, β) ≈ (1 + cos(β)) / 2
        Test.@test PartialFSHT.wigner_d(1, 1, -1, β) ≈ (1 - cos(β)) / 2
        Test.@test PartialFSHT.wigner_d(1, -1, 0, β) ≈ sin(β) / sqrt(2)
        Test.@test PartialFSHT.wigner_d(1, 0, -1, β) ≈ -sin(β) / sqrt(2)
        # m = n = 0 is the Legendre polynomial; the degree recurrence divides by ℓ, so ℓ₀ = 0
        # needs its own seed and got zero before that was handled.
        Test.@test PartialFSHT.wigner_d(2, 0, 0, β) ≈ (3cos(β)^2 - 1) / 2
        Test.@test PartialFSHT.wigner_d(3, 0, 0, β) ≈ (5cos(β)^3 - 3cos(β)) / 2
    end

    Test.@testset "endpoints" begin
        Test.@test PartialFSHT.wigner_d(5, 2, 2, 0.0) ≈ 1
        Test.@test PartialFSHT.wigner_d(5, 2, 3, 0.0) ≈ 0
        Test.@test abs(PartialFSHT.wigner_d(4, 2, -2, Float64(π))) ≈ 1
    end

    Test.@testset "high degree vs BigFloat, and bounded by 1" begin
        for (ℓ, m, n, b) in ((512, 100, -2, 0.8), (2048, 700, -1, 0.5), (4096, 30, -2, 1.1),
                             (1024, 0, 0, 2.0))
            g = PartialFSHT.wigner_d(Float64, ℓ, m, n, b)
            r = PartialFSHT.wigner_d(BigFloat, ℓ, m, n, b)
            Test.@test abs(g) <= 1 + 1e-10
            if abs(r) > 1e-300
                Test.@test Float64(abs(g - r) / abs(r)) < 1e-10
            end
        end
    end

    Test.@testset "symmetry d^ℓ_{mn} = (-1)^{m-n} d^ℓ_{nm}" begin
        for (ℓ, m, n) in ((7, 3, -2), (40, 11, 5), (200, 60, -7))
            a = PartialFSHT.wigner_d(ℓ, m, n, 0.9)
            b = PartialFSHT.wigner_d(ℓ, n, m, 0.9)
            Test.@test a ≈ (iseven(m - n) ? 1 : -1) * b atol = 1e-12
        end
    end
end

Test.@testset "spin-weighted harmonics" begin
    Test.@testset "spin 0 reduces to the scalar harmonic" begin
        θ, φ = 0.8, 2.1
        for (ℓ, m) in ((0, 0), (3, 2), (12, -5))
            Test.@test PartialFSHT.sYlm(0, ℓ, m, θ, φ) ≈ PartialFSHT.ylm(ℓ, m, θ, φ) atol = 1e-13
        end
    end

    Test.@testset "spin Direct oracle" begin
        Random.seed!(41)
        L, s, n = 8, 1, 25
        θ = 0.4 .* rand(n) .+ 0.2
        φ = 2π .* rand(n)
        plan = PartialFSHT.Plan(θ, φ; lmax = L, spin = s,
                                method = PartialFSHT.DirectAlgorithm())
        alm = PartialFSHT.zeros_alm(L)
        alm[PartialFSHT.spin_coeff_index(4, 2, L)] = 1
        Test.@test maximum(abs.((plan * alm) .- PartialFSHT.sYlm.(s, 4, 2, θ, φ))) < 1e-12
        Test.@test _reladj(plan, randn(ComplexF64, PartialFSHT.coeff_size(L)...),
                           randn(ComplexF64, n)) < 1e-12
    end

    Test.@testset "spin stretch vs spin Direct" begin
        Random.seed!(42)
        L, tol = 256, 1e-8
        for s in (1, 2, -1)
            n = 120
            θ = (20 * π / 180) .* sqrt.(rand(n))
            φ = 2π .* rand(n)
            pd = PartialFSHT.Plan(θ, φ; lmax = L, spin = s,
                                  method = PartialFSHT.DirectAlgorithm())
            ps = PartialFSHT.Plan(θ, φ; lmax = L, spin = s,
                                  method = PartialFSHT.StretchAlgorithm(),
                                  nufft = PartialFSHT.DirectSumBackend(), tol = tol)
            alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
            fd = pd * alm
            fs = ps * alm
            Test.@test maximum(abs.(fs .- fd)) / maximum(abs.(fd)) < 10 * tol
            Test.@test _reladj(ps, alm, randn(ComplexF64, n)) < 1e-12
        end
    end
end
