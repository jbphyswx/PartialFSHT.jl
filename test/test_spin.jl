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

Test.@testset "spin ring engine against wigner_d" begin
    # The spin *fast* path is otherwise gated only against the spin *oracle*, and both share
    # `SpinRings` — so a fault in the shared ring engine would cancel out of that comparison.
    # `wigner_d` is the independent reference: a single-value, block-scaled evaluation.
    #
    # Errors are measured against the ring *maximum*, not against each ring's own value. Polar
    # optimisation deliberately drops orders that are evanescent on a ring, and there they are
    # small in absolute terms while a self-relative error is meaningless — the dropped entries here
    # sit at 1e-180 and below relative to the peak.
    Test.@testset "wide colatitude span, large orders" begin
        nθ = 200
        θ = collect(range(0.01, 0.6; length = nθ))
        for L in (128, 256, 512)
            eng = PartialFSHT.SpinRings(θ, L, 2, 0.0)
            for m in (L ÷ 4, L ÷ 2, 3L ÷ 4, L - 1)
                alm = zeros(ComplexF64, L + 1, 2L + 1)
                alm[PartialFSHT.spin_coeff_index(L, m, L)] = 1
                G = zeros(ComplexF64, nθ, 2L + 1, 1)
                PartialFSHT.spin_rings_from_alm!(G, alm, eng)
                refs = [PartialFSHT._Nℓ(Float64, L) *
                        PartialFSHT.wigner_d(Float64, L, m, -2, θ[j]) for j in 1:nθ]
                peak = maximum(abs, refs)
                Test.@test peak > 0
                worst = 0.0
                worst_dropped = 0.0
                for j in 1:nθ
                    got = real(G[j, m + L + 1, 1])
                    if got == 0 && refs[j] != 0
                        worst_dropped = max(worst_dropped, abs(refs[j]) / peak)
                    else
                        worst = max(worst, abs(got - refs[j]) / peak)
                    end
                end
                Test.@test worst < 1e-12
                # Anything truncated must have been negligible, so truncation cannot hide an error.
                Test.@test worst_dropped < 1e-30
            end
        end
    end

    Test.@testset "chunking bounds the shared exponent's span" begin
        # Both engines carry one power-of-two exponent per ring chunk, and both must split a chunk
        # before its spread can flush the smallest ring. The scalar engine bounds
        # `mmax · |Δlog₂ sinθ|`; the spin seed's span is `2ℓ₀ · |Δlog₂ cos(θ/2) or sin(θ/2)|`, so a
        # wide colatitude range at high degree must produce more than one chunk here too.
        θwide = collect(range(0.01, 1.4; length = 300))
        for L in (256, 1024)
            Test.@test length(PartialFSHT.RingLegendre(θwide, L, 0.0).chunks) > 1
            Test.@test length(PartialFSHT.SpinRings(θwide, L, 2, 0.0).chunks) > 1
        end
    end

    Test.@testset "poles are exact without an endpoint special case" begin
        # The tabulated seed reaches θ = 0 and θ = π through `p == 0` guards rather than branches,
        # so those rings must come out exactly right.
        L, s = 6, 2
        θ = [0.0, 0.3, Float64(π) / 2, Float64(π) - 0.3, Float64(π)]
        eng = PartialFSHT.SpinRings(θ, L, s, 0.0)
        for m in (-3, 0, 2, 5)
            alm = zeros(ComplexF64, L + 1, 2L + 1)
            alm[PartialFSHT.spin_coeff_index(5, m, L)] = 1
            G = zeros(ComplexF64, length(θ), 2L + 1, 1)
            PartialFSHT.spin_rings_from_alm!(G, alm, eng)
            for j in eachindex(θ)
                ref = PartialFSHT._Nℓ(Float64, 5) * PartialFSHT.wigner_d(Float64, 5, m, -s, θ[j])
                Test.@test real(G[j, m + L + 1, 1]) ≈ ref atol = 1e-13
            end
        end
    end

    Test.@testset "reflection is absorbed by the engine, not by the coefficients" begin
        # `reflect = true` means the physical colatitudes are π .- θ. It must reproduce
        # ₛY_{ℓm}(π-θ) = (-1)^{ℓ+m} ₋ₛY_{ℓm}(θ) with the spin weight left alone.
        L, s = 8, 1
        θasc = collect(range(0.2, 1.1; length = 9))
        eng = PartialFSHT.SpinRings(θasc, L, s, 0.0; reflect = true)
        for (ℓ, m) in ((3, 2), (7, -5), (8, 0))
            alm = zeros(ComplexF64, L + 1, 2L + 1)
            alm[PartialFSHT.spin_coeff_index(ℓ, m, L)] = 1
            G = zeros(ComplexF64, length(θasc), 2L + 1, 1)
            PartialFSHT.spin_rings_from_alm!(G, alm, eng)
            for j in eachindex(θasc)
                ref = PartialFSHT._Nℓ(Float64, ℓ) *
                      PartialFSHT.wigner_d(Float64, ℓ, m, -s, Float64(π) - θasc[j])
                Test.@test real(G[j, m + L + 1, 1]) ≈ ref atol = 1e-12
            end
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
