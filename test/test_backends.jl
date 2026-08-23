Test.@testset "NUFFT backends agree to tolerance" begin
    # The gate the previous suite missed: it accepted 5e-3 disagreement between two NUFFT
    # libraries at tol = 1e-6, which is a mode-ordering/sign bug rather than a tolerance.
    Random.seed!(81)
    L = 256
    n = 300
    θ = (20 * π / 180) .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
    for tol in (1e-6, 1e-10)
        kw = (lmax = L, method = PartialFSHT.StretchAlgorithm(), tol = tol)
        f0 = PartialFSHT.Plan(θ, φ; kw..., nufft = PartialFSHT.DirectSumBackend()) * alm
        for be in (PartialFSHT.FINUFFTBackend(), PartialFSHT.NonuniformFFTsBackend())
            f = PartialFSHT.Plan(θ, φ; kw..., nufft = be) * alm
            rel = maximum(abs.(f .- f0)) / maximum(abs.(f0))
            Test.@test rel < 200 * tol
        end
    end
end

Test.@testset "AutoNUFFT resolves to a fast backend when one is loaded" begin
    Test.@test PartialFSHT._ext_loaded(:PartialFSHTFINUFFTExt) ||
               PartialFSHT._ext_loaded(:PartialFSHTNonuniformFFTsExt)
    Test.@test !(PartialFSHT._resolve_nufft(PartialFSHT.AutoNUFFT()) isa
                 PartialFSHT.DirectSumBackend)
end

Test.@testset "NUFSHT full-sphere leaf" begin
    Test.@test PartialFSHT._ext_loaded(:PartialFSHTNUFSHTExt)
    Random.seed!(82)
    L = 12
    pts = FastSphericalHarmonics.sph_points(L + 1)
    θ = vec([t for t in pts[1], p in pts[2]])
    φ = vec([p for t in pts[1], p in pts[2]])
    pn = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.NUFSHTAlgorithm(), tol = 1e-10)
    pd = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    Test.@test PartialFSHT.algorithm(pn) isa PartialFSHT.NUFSHTAlgorithm
    alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
    fn = pn * alm
    fd = pd * alm
    Test.@test LinearAlgebra.norm(fn .- fd) / LinearAlgebra.norm(fd) < 1e-8
    Test.@test _reladj(pn, alm, randn(ComplexF64, length(θ))) < 1e-9
end

Test.@testset "threaded backend matches serial exactly enough" begin
    Test.@test PartialFSHT._omt_loaded()
    Random.seed!(83)
    L = 256
    n1 = 60
    θ = vcat(0.22 .* sqrt.(rand(n1)), π .- 0.22 .* sqrt.(rand(n1)))
    φ = 2π .* rand(2n1)
    kw = (lmax = L, method = PartialFSHT.MultiBoxAlgorithm(),
          nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
    ps = PartialFSHT.Plan(θ, φ; kw..., backend = ComputationalBackends.SerialBackend())
    pt = PartialFSHT.Plan(θ, φ; kw..., backend = ComputationalBackends.ThreadedBackend())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
    fs = ps * alm
    ft = pt * alm
    # same arithmetic, possibly different summation order
    Test.@test maximum(abs.(ft .- fs)) <= 1e-12 * maximum(abs.(fs))
    Test.@test _reladj(pt, alm, randn(ComplexF64, 2n1)) < 1e-12
end

Test.@testset "mul_nodeset over independent node sets" begin
    Random.seed!(84)
    L = 8
    θs = [0.3 .* rand(20), 0.4 .* rand(25) .+ 0.3]
    φs = [2π .* rand(20), 2π .* rand(25)]
    alms = [randn(ComplexF64, PartialFSHT.coeff_size(L)...) for _ in 1:2]
    fs = PartialFSHT.mul_nodeset(θs, φs, alms, L; method = PartialFSHT.DirectAlgorithm())
    for i in 1:2
        p = PartialFSHT.Plan(θs[i], φs[i]; lmax = L,
                             method = PartialFSHT.DirectAlgorithm())
        Test.@test LinearAlgebra.norm(fs[i] .- (p * alms[i])) < 1e-12
    end
    Test.@test_throws ArgumentError PartialFSHT.mul_nodeset(
        θs, φs, alms, L, ComputationalBackends.MPIBackend())
end

Test.@testset "device-generic DFS on JLArray" begin
    Test.@test JLArrays.JLArray <: GPUArraysCore.AbstractGPUArray
    Random.seed!(85)
    Nθ, Nφ, B = 7, 12, 2
    F = randn(ComplexF64, Nθ, Nφ, B)
    for s in (0, 1)
        host = zeros(ComplexF64, 2Nθ, Nφ, B)
        PartialFSHT.dfs_double!(host, F, s)
        dev = JLArrays.JLArray(zeros(ComplexF64, 2Nθ, Nφ, B))
        PartialFSHT.dfs_double!(dev, JLArrays.JLArray(F), s)
        Test.@test Array(dev) ≈ host
        fold_h = zeros(ComplexF64, Nθ, Nφ, B)
        PartialFSHT.dfs_fold!(fold_h, host, s)
        fold_d = JLArrays.JLArray(zeros(ComplexF64, Nθ, Nφ, B))
        PartialFSHT.dfs_fold!(fold_d, dev, s)
        Test.@test Array(fold_d) ≈ fold_h
    end
end

Test.@testset "DFS doubling is the transpose of folding" begin
    Random.seed!(86)
    Nθ, Nφ, B = 6, 10, 1
    for s in (0, 1)
        F = randn(ComplexF64, Nθ, Nφ, B)
        Fd = zeros(ComplexF64, 2Nθ, Nφ, B)
        PartialFSHT.dfs_double!(Fd, F, s)
        G = randn(ComplexF64, 2Nθ, Nφ, B)
        Gf = zeros(ComplexF64, Nθ, Nφ, B)
        PartialFSHT.dfs_fold!(Gf, G, s)
        Test.@test abs(LinearAlgebra.dot(Fd, G) - LinearAlgebra.dot(F, Gf)) <
                   1e-12 * abs(LinearAlgebra.dot(Fd, G))
    end
end
