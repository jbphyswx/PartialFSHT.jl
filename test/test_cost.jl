Test.@testset "cost model counts rather than guesses" begin
    Test.@testset "ring work grows with degree and shrinks with a thinner interval" begin
        θr = collect(range(1e-3, 0.4; length = 100))
        w = [first(PartialFSHT.counted_ring_work(θr, L, 1e-8)) for L in (128, 256, 512)]
        Test.@test issorted(w)
        wide, mwide = PartialFSHT.counted_ring_work(collect(range(1e-3, 1.5; length = 100)),
                                                    512, 1e-8)
        narrow, mnarrow = PartialFSHT.counted_ring_work(collect(range(1e-3, 0.2; length = 100)),
                                                        512, 1e-8)
        Test.@test narrow < wide
        # The largest retained order comes back from the same pass, and a thinner interval keeps
        # fewer orders, so the planner cannot size a grid from a count it did not make.
        Test.@test mnarrow < mwide
    end

    Test.@testset "counting matches the engine it predicts" begin
        # The cost model and the ring engine must agree exactly, or a planner decision is made
        # about work that never happens.
        for (L, hi) in ((128, 0.3), (512, 0.5), (512, 1.4))
            θr = collect(range(1e-3, hi; length = 90))
            counted, mmax = PartialFSHT.counted_ring_work(θr, L, 1e-8)
            eng = PartialFSHT.RingLegendre(θr, L, 1e-8)
            Test.@test counted == PartialFSHT.ring_work(eng)
            Test.@test mmax == eng.mmax
        end
    end

    Test.@testset "a degenerate geometry costs Inf rather than throwing" begin
        b = PartialFSHT.bounding_box(0.01 .* rand(20), 2π .* rand(20))
        Test.@test PartialFSHT.cost_stretch(b, 20, 8, 1, 1e-12) == Inf
    end

    Test.@testset "stretch beats a full-sphere transform on a compact cap" begin
        Random.seed!(71)
        n = 4000
        θ = (20 * π / 180) .* sqrt.(rand(n))
        φ = 2π .* rand(n)
        b = PartialFSHT.bounding_box(θ, φ)
        for L in (512, 1024, 2048)
            cs = PartialFSHT.cost_stretch(b, n, L, 1, 1e-8)
            cn = PartialFSHT.cost_nufsht(n, L, 1, 1e-8)
            Test.@test cs < cn
        end
    end

    Test.@testset "select_method: compact cap takes the stretch leaf" begin
        Random.seed!(72)
        n = 4000
        θ = (20 * π / 180) .* sqrt.(rand(n))
        φ = 2π .* rand(n)
        for L in (256, 512, 1024, 2048)
            chosen = PartialFSHT.select_method(θ, φ, L, 1e-8, PartialFSHT.AutoAlgorithm(), 1, 0)
            Test.@test chosen isa PartialFSHT.StretchAlgorithm
        end
    end

    Test.@testset "select_method: tiny problems take the oracle" begin
        Random.seed!(73)
        θ = 0.3 .* rand(12); φ = 2π .* rand(12)
        Test.@test PartialFSHT.select_method(θ, φ, 4, 1e-8,
                                             PartialFSHT.AutoAlgorithm(), 1, 0) isa
                   PartialFSHT.DirectAlgorithm
    end

    Test.@testset "select_method: two antipodal caps take the multi-box leaf" begin
        Random.seed!(74)
        n1 = 2000
        θ = vcat(0.2 .* sqrt.(rand(n1)), π .- 0.2 .* sqrt.(rand(n1)))
        φ = 2π .* rand(2n1)
        chosen = PartialFSHT.select_method(θ, φ, 512, 1e-8, PartialFSHT.AutoAlgorithm(), 1, 0)
        Test.@test chosen isa PartialFSHT.MultiBoxAlgorithm
    end

    Test.@testset "cost constants are installable and reported" begin
        old = (PartialFSHT._COST.ring, PartialFSHT._COST.fft,
               PartialFSHT._COST.spread, PartialFSHT._COST.ft_conn)
        try
            PartialFSHT.set_cost_constants!(ring = 2.0)
            Test.@test PartialFSHT.cost_calibrated()
            Test.@test PartialFSHT._COST.ring == 2.0
        finally
            PartialFSHT.set_cost_constants!(ring = old[1], fft = old[2],
                                            spread = old[3], ft_conn = old[4])
            PartialFSHT._COST.calibrated = false
        end
    end
end
