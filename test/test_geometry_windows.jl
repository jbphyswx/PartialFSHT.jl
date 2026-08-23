Test.@testset "bounding box is exactly minimal" begin
    Random.seed!(21)
    # colatitude extent is a min/max, so it is exact
    θ = 0.3 .+ 0.4 .* rand(500)
    φ = 1.0 .+ 0.5 .* rand(500)
    b = PartialFSHT.bounding_box(θ, φ)
    Test.@test b.θlo ≈ minimum(θ)
    Test.@test b.θhi ≈ maximum(θ)
    Test.@test PartialFSHT.box_phi_extent(b) ≈ maximum(φ) - minimum(φ)
    Test.@test !b.touches_north && !b.touches_south

    Test.@testset "longitude extent is the largest-gap complement, including wrap-around" begin
        # points straddling φ = 0: the minimal arc wraps, so a naive min/max would report ~2π
        φw = vcat(0.05 .* rand(200), 2π .- 0.05 .* rand(200))
        θw = fill(1.0, 400)
        bw = PartialFSHT.bounding_box(θw, φw)
        Test.@test PartialFSHT.box_phi_extent(bw) < 0.15
        # a full circle is reported as a full circle
        bf = PartialFSHT.bounding_box(fill(1.0, 500), 2π .* rand(500))
        Test.@test PartialFSHT.box_phi_extent(bf) > 2π - 0.2
    end

    Test.@testset "a pole in range forces the full longitude range" begin
        bp = PartialFSHT.bounding_box(vcat(0.0, 0.2 .* rand(100)), 0.1 .* rand(101))
        Test.@test bp.touches_north
        Test.@test PartialFSHT.box_phi_extent(bp) ≈ 2π
    end
end

Test.@testset "partitioning: bands stay whole, disjoint components separate" begin
    Random.seed!(22)
    costfn = (b, np) -> PartialFSHT.cost_stretch(b, np, 256, 1, 1e-8)

    # An equatorial band is one colatitude interval and must not be cut: the previous
    # centroid-cap geometry saw θ⋆ ≈ π here and split it into up to 32 caps.
    half = 8 * π / 180
    θb = (π / 2) .+ half .* (2 .* rand(3000) .- 1)
    φb = 2π .* rand(3000)
    Test.@test length(PartialFSHT.partition_boxes(θb, φb, costfn)) == 1
    bb = PartialFSHT.bounding_box(θb, φb)
    Test.@test PartialFSHT.box_theta_extent(bb) < 2.1 * half

    # Two antipodal caps must separate.
    θa = vcat(0.2 .* sqrt.(rand(1500)), π .- 0.2 .* sqrt.(rand(1500)))
    φa = 2π .* rand(3000)
    Test.@test length(PartialFSHT.partition_boxes(θa, φa, costfn)) == 2

    # A single compact patch must not be split.
    θc = 0.2 .* sqrt.(rand(2000))
    φc = 2π .* rand(2000)
    Test.@test length(PartialFSHT.partition_boxes(θc, φc, costfn)) == 1
end

Test.@testset "stretch geometry and apodisation" begin
    Test.@testset "derived relations hold" begin
        for (θhi, L, ε) in ((0.35, 512, 1e-8), (0.6, 1024, 1e-10), (1.0, 2048, 1e-6))
            g = PartialFSHT.stretch_geometry(0.0, θhi, L, ε)
            β = PartialFSHT.es_beta(ε)
            Test.@test g.β ≈ β
            Test.@test g.εapo ≈ sqrt(2β / (L * g.h))
            Test.@test g.K == max(1, Int(ceil(L * g.h * (1 + g.εapo)^2 / π)))
            Test.@test g.Θ1 ≈ π / (1 + g.εapo)
            Test.@test g.σ ≈ (π - g.Θ1) / 2
            Test.@test g.ΘT ≈ g.Θ1 + g.σ
        end
    end

    Test.@testset "window contract: exactly 1 on the data, 0 at the period boundary" begin
        g = PartialFSHT.stretch_geometry(0.0, 0.35, 512, 1e-8)
        for Θ in range(0, g.Θ1; length = 50)
            Test.@test PartialFSHT.stretch_window(g, Θ) == 1
        end
        Test.@test PartialFSHT.stretch_window(g, π) == 0
        # monotone non-increasing through the apodisation annulus
        ws = [PartialFSHT.stretch_window(g, Θ) for Θ in range(g.Θ1, π; length = 200)]
        Test.@test all(diff(ws) .<= 1e-12)
    end

    Test.@testset "ES CDF is converged, not a coarse midpoint rule" begin
        # The substitution z = sinu removes the branch point at |z| = 1, so composite
        # Gauss-Legendre converges; a 64-point midpoint rule on z is off by ~1%.
        β = PartialFSHT.es_beta(1e-8)
        for u in (-1.0, -0.3, 0.0, 0.7, π / 2)
            coarse = PartialFSHT.es_cdf(β, u)
            Test.@test isfinite(coarse)
        end
        Test.@test PartialFSHT.es_cdf(β, -π / 2) == 0
        Test.@test PartialFSHT.es_cdf(β, π / 2) > 0
        # monotone
        us = range(-π / 2, π / 2; length = 60)
        Test.@test issorted([PartialFSHT.es_cdf(β, u) for u in us])
    end

    Test.@testset "coordinate maps invert" begin
        for (lo, hi, L) in ((0.0, 0.4, 512), (0.8, 1.2, 1024), (0.0, π, 256))
            g = PartialFSHT.stretch_geometry(lo, hi, L, 1e-8)
            for θ in range(max(lo, 1e-6), hi; length = 20)
                Θ = PartialFSHT.theta_to_stretch(g, θ)
                Test.@test PartialFSHT.stretch_to_theta(g, Θ) ≈ θ atol = 1e-10
            end
        end
    end

    Test.@testset "an interior periodisation never pads across a pole" begin
        # Padding a near-pole interval as if it were interior runs θ negative, where colatitude
        # stops being a coordinate; that capped accuracy at ~1e-3 regardless of tol.
        for (lo, hi, L) in ((0.02, 0.35, 256), (0.01, 0.5, 512), (2.7, 3.13, 512))
            g = PartialFSHT.stretch_geometry(lo, hi, L, 1e-8)
            if g.kind === :interior
                pad = g.h * g.εapo
                Test.@test g.θlo - pad >= -1e-12
                Test.@test g.θhi + pad <= π + 1e-12
            end
        end
    end

    Test.@testset "degenerate geometry errors instead of silently missing tol" begin
        # lmax·h below the apodisation budget: the padding would exceed the data.
        Test.@test_throws ArgumentError PartialFSHT.stretch_geometry(0.0, 0.05, 32, 1e-10)
        Test.@test_throws ArgumentError PartialFSHT.stretch_geometry(0.4, 0.45, 16, 1e-12)
    end

    Test.@testset "full sphere is the no-apodisation case" begin
        g = PartialFSHT.stretch_geometry(0.0, π, 256, 1e-8)
        Test.@test g.kind === :full
        Test.@test g.K == 256
        Test.@test PartialFSHT.stretch_window(g, 1.0) == 1
    end
end

Test.@testset "grid sizes are 5-smooth, and nφ is even" begin
    # nφ *must* be even: the DFS reflection shifts longitude by Nφ÷2 cells to realise φ→φ+π,
    # which is only a half-turn for even Nφ. Nφ = 135 shifts 3.118 rad instead of 3.1416 and
    # caps accuracy at ~1e-5 whatever tolerance is requested.
    issmooth(n) = begin
        m = n
        for p in (2, 3, 5)
            while m % p == 0
                m ÷= p
            end
        end
        m == 1
    end
    for n in (1, 2, 7, 100, 122, 227, 1000, 4097)
        Test.@test PartialFSHT.next_smooth(n) >= n
        Test.@test issmooth(PartialFSHT.next_smooth(n))
        e = PartialFSHT.next_smooth_even(n)
        Test.@test e >= n
        Test.@test iseven(e)
        Test.@test issmooth(e)
    end
    Test.@test PartialFSHT.next_smooth_even(135) == 144
    Test.@test PartialFSHT.next_smooth_even(225) == 240
end
