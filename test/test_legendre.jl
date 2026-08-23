Test.@testset "scaled associated Legendre" begin
    Test.@testset "vs BigFloat at high degree" begin
        # The case the raw recurrence got 100% wrong: m near the turning point on the outer
        # ring of a compact cap, where sin^m θ underflows Float64 long before it stops
        # mattering. At lmax = 4096 this held 67% of that ring's Legendre energy.
        worst = 0.0
        for L in (256, 1024, 2048, 4096, 8192)
            g = PartialFSHT.stretch_geometry(0.0, 20 * π / 180, L, 1e-8)
            θmax = g.h * (1 + g.εapo)
            m = PartialFSHT.polar_mmax(θmax, L, 1e-8)
            got = PartialFSHT.alf(Float64, L, m, θmax)
            ref = PartialFSHT.alf(BigFloat, L, m, θmax)
            Test.@test ref != 0                       # the value is *not* negligible
            Test.@test got != 0                       # and Float64 must not flush it away
            worst = max(worst, Float64(abs(got - ref) / abs(ref)))
        end
        Test.@test worst < 1e-11

        for (L, m, θ) in ((4096, 1000, 0.4), (4096, 3000, 1.2), (8192, 4000, 0.9),
                          (2048, 0, 1.5), (513, 513, 1.0), (100, 7, 3.0))
            got = PartialFSHT.alf(Float64, L, m, θ)
            ref = PartialFSHT.alf(BigFloat, L, m, θ)
            if abs(ref) >= big(floatmin(Float64))
                Test.@test Float64(abs(got - ref) / abs(ref)) < 1e-11
            end
        end
    end

    Test.@testset "closed-form Ylm" begin
        θ, φ = 0.7, 1.1
        i4π = 1 / sqrt(4π)
        Test.@test PartialFSHT.ylm(0, 0, 0.3, 1.2) ≈ i4π
        Test.@test PartialFSHT.ylm(1, 0, θ, φ) ≈ sqrt(3 / 4π) * cos(θ)
        Test.@test PartialFSHT.ylm(1, 1, θ, φ) ≈ -sqrt(3 / 8π) * sin(θ) * cis(φ)
        Test.@test PartialFSHT.ylm(1, -1, θ, φ) ≈ sqrt(3 / 8π) * sin(θ) * cis(-φ)
        Test.@test PartialFSHT.ylm(2, 2, θ, φ) ≈ sqrt(15 / 32π) * sin(θ)^2 * cis(2φ)
        # P̄_ℓ^{-m} = (-1)^m P̄_ℓ^m
        for (ℓ, m) in ((3, 2), (7, 5), (40, 13))
            Test.@test PartialFSHT.alf(Float64, ℓ, -m, θ) ≈
                       (iseven(m) ? 1 : -1) * PartialFSHT.alf(Float64, ℓ, m, θ)
        end
    end

    Test.@testset "Float32 tracks Float64" begin
        for (ℓ, m, θ) in ((64, 20, 0.5), (256, 100, 0.8))
            a32 = PartialFSHT.alf(Float32, ℓ, m, Float32(θ))
            a64 = PartialFSHT.alf(Float64, ℓ, m, θ)
            Test.@test abs(a32 - a64) <= 1e-4 * max(abs(a64), 1e-30)
        end
    end

    Test.@testset "polar optimisation truncates only what is negligible" begin
        # The retained order limit must actually bound the discarded content. The previous
        # criterion used a constant pad, which under-estimates the Airy transition width
        # (25 vs ~85 orders at ℓ = 4096) and so silently over-truncated.
        for (L, θ, ε) in ((512, 0.4, 1e-8), (1024, 0.6, 1e-8), (2048, 0.35, 1e-6))
            mm = PartialFSHT.polar_mmax(θ, L, ε)
            mm >= L && continue
            peak = maximum(abs(PartialFSHT.alf(Float64, L, m, θ)) for m in 0:mm)
            dropped = maximum(abs(PartialFSHT.alf(Float64, L, m, θ))
                              for m in (mm + 1):min(L, mm + 40))
            Test.@test dropped <= ε * peak
        end
        # monotone in θ on [0, π/2], as the contiguous-range argument requires
        ms = [PartialFSHT.polar_mmax(θ, 512, 1e-8) for θ in range(1e-4, π / 2; length = 40)]
        Test.@test issorted(ms)
    end

    Test.@testset "ring engine forward is the transpose of its adjoint" begin
        Random.seed!(11)
        for L in (16, 64)
            θr = sort(0.05 .+ 0.9 .* rand(37))
            eng = PartialFSHT.RingLegendre(θr, L, 1e-12)
            G = zeros(ComplexF64, length(θr), 2L + 1, 1)
            alm = randn(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
            PartialFSHT.rings_from_alm!(G, alm, eng)
            Gr = randn(ComplexF64, length(θr), 2L + 1, 1)
            almr = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
            PartialFSHT.alm_from_rings!(almr, Gr, eng)
            # ⟨R a, g⟩ = ⟨a, Rᵀ g⟩ to round-off, not to a quadrature tolerance
            lhs = LinearAlgebra.dot(G, Gr)
            rhs = LinearAlgebra.dot(alm, almr)
            Test.@test abs(lhs - rhs) <= 1e-11 * abs(lhs)
        end
    end

    Test.@testset "ring engine matches per-point evaluation" begin
        L = 48
        θr = sort([0.02, 0.11, 0.37, 0.8, 1.4, 2.7])
        eng = PartialFSHT.RingLegendre(θr, L, 1e-13)
        alm = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        alm[PartialFSHT.sph_mode(30, 7), 1] = 1.0
        G = zeros(ComplexF64, length(θr), 2L + 1, 1)
        PartialFSHT.rings_from_alm!(G, alm, eng)
        for (j, θ) in enumerate(θr)
            Test.@test G[j, 7 + L + 1, 1] ≈ PartialFSHT.alf(Float64, 30, 7, θ) atol = 1e-13
        end
    end

    Test.@testset "ring work is counted, and polar optimisation reduces it" begin
        L = 512
        θr = collect(range(1e-3, 0.5; length = 120))
        w_opt = PartialFSHT.counted_ring_work(θr, L, 1e-8)
        w_all = length(θr) * sum(L - m + 1 for m in 0:L)
        Test.@test w_opt < w_all
        Test.@test w_opt > 0
    end
end
