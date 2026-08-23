# Allocation gates on *every* leaf, against an explicit SerialBackend. The previous suite gated
# only the Direct path, so the leaf that actually ships was never checked.
_alloc2(f!, args...) = (f!(args...); f!(args...);
                        min(@allocated(f!(args...)), @allocated(f!(args...))))

Test.@testset "zero-allocation applies" begin
    Random.seed!(61)
    n = 200
    θ = (20 * π / 180) .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    be = ComputationalBackends.SerialBackend()

    Test.@testset "Direct" begin
        for L in (16, 64)
            p = PartialFSHT.Plan(θ, φ; lmax = L, backend = be,
                                 method = PartialFSHT.DirectAlgorithm())
            alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
            f = zeros(ComplexF64, n)
            Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q, x), f, p, alm) == 0
            Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q', x), alm, p, f) == 0
        end
    end

    Test.@testset "Stretch" begin
        for L in (256, 512)
            p = PartialFSHT.Plan(θ, φ; lmax = L, backend = be, tol = 1e-8,
                                 method = PartialFSHT.StretchAlgorithm(),
                                 nufft = PartialFSHT.DirectSumBackend())
            alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
            f = zeros(ComplexF64, n)
            Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q, x), f, p, alm) == 0
            Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q', x), alm, p, f) == 0
        end
    end

    Test.@testset "Stretch, spin" begin
        L = 256
        p = PartialFSHT.Plan(θ, φ; lmax = L, spin = 2, backend = be, tol = 1e-8,
                             method = PartialFSHT.StretchAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend())
        alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
        f = zeros(ComplexF64, n)
        Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q, x), f, p, alm) == 0
        Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q', x), alm, p, f) == 0
    end

    Test.@testset "MultiBox" begin
        L = 256
        n1 = 80
        θ2 = vcat(0.25 .* sqrt.(rand(n1)), π .- 0.25 .* sqrt.(rand(n1)))
        φ2 = 2π .* rand(2n1)
        p = PartialFSHT.Plan(θ2, φ2; lmax = L, backend = be, tol = 1e-8,
                             method = PartialFSHT.MultiBoxAlgorithm(),
                             nufft = PartialFSHT.DirectSumBackend())
        alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
        f = zeros(ComplexF64, 2n1)
        Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q, x), f, p, alm) == 0
        Test.@test _alloc2((y, q, x) -> LinearAlgebra.mul!(y, q', x), alm, p, f) == 0
    end

    Test.@testset "solve! against a preallocated workspace" begin
        L = 16
        p = PartialFSHT.Plan(θ, φ; lmax = L, backend = be,
                             method = PartialFSHT.DirectAlgorithm())
        alm = PartialFSHT.zeros_alm(L)
        f = randn(ComplexF64, n)
        ws = PartialFSHT.AnalysisWorkspace(p)
        PartialFSHT.solve!(alm, p, f; maxiter = 10, ws = ws)
        # the iteration itself must not allocate; a few hundred bytes of return plumbing is
        # per-call, not per-iteration, so gate on growth with iteration count
        a10 = @allocated PartialFSHT.solve!(alm, p, f; maxiter = 10, ws = ws)
        a80 = @allocated PartialFSHT.solve!(alm, p, f; maxiter = 80, ws = ws)
        Test.@test a80 <= a10 + 64
    end
end
