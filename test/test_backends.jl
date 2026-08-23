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

Test.@testset "device-generic ring pass on JLArray" begin
    # The forward kernel is race-free — one thread per (ring, order), each owning its output — so
    # it must agree with the host *exactly*. The adjoint scatters through atomic adds, so its
    # summation order is not fixed and it can only agree to round-off.
    Random.seed!(89)
    for L in (32, 128, 257)
        θh = sort(0.02 .+ 0.6 .* rand(41))
        almh = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        for ℓ in 0:L, m in -ℓ:ℓ
            almh[PartialFSHT.sph_mode(ℓ, m), 1] = randn(ComplexF64)
        end

        engh = PartialFSHT.RingLegendre(θh, L, 1e-8; ntrans = 1)
        Gh = zeros(ComplexF64, length(θh), 2L + 1, 1)
        PartialFSHT.rings_from_alm!(Gh, almh, engh)

        engd = PartialFSHT.RingLegendre(JLArrays.JLArray(θh), L, 1e-8; ntrans = 1)
        Gd = JLArrays.JLArray(zeros(ComplexF64, length(θh), 2L + 1, 1))
        PartialFSHT.rings_from_alm!(Gd, JLArrays.JLArray(almh), engd)
        Test.@test Array(Gd) == Gh

        ah = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        PartialFSHT.alm_from_rings!(ah, Gh, engh)
        ad = JLArrays.JLArray(zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1))
        PartialFSHT.alm_from_rings!(ad, JLArrays.JLArray(Gh), engd)
        Test.@test maximum(abs.(Array(ad) .- ah)) <= 1e-12 * maximum(abs.(ah))

        # adjoint identity through the device engine alone
        Gr = randn(ComplexF64, length(θh), 2L + 1, 1)
        ar = JLArrays.JLArray(zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1))
        PartialFSHT.alm_from_rings!(ar, JLArrays.JLArray(Gr), engd)
        lhs = LinearAlgebra.dot(Gh, Gr)
        rhs = LinearAlgebra.dot(almh, Array(ar))
        Test.@test abs(lhs - rhs) <= 1e-11 * abs(lhs)
    end
end

Test.@testset "device-generic spin ring pass on JLArray" begin
    # Same contract as the scalar kernel: the forward direction is race-free and must match the
    # host exactly, the adjoint scatters atomically and matches to round-off. The seed comes from
    # the engine's tables, so this also gates those being transferred to the device.
    Random.seed!(90)
    for (L, s) in ((32, 2), (128, 1), (128, -2))
        θh = sort(0.05 .+ 0.5 .* rand(37))
        almh = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        for ℓ in 0:L, m in -ℓ:ℓ
            almh[PartialFSHT.spin_coeff_index(ℓ, m, L), 1] = randn(ComplexF64)
        end

        engh = PartialFSHT.SpinRings(θh, L, s, 1e-8; ntrans = 1)
        Gh = zeros(ComplexF64, length(θh), 2L + 1, 1)
        PartialFSHT.spin_rings_from_alm!(Gh, almh, engh)

        engd = PartialFSHT.SpinRings(JLArrays.JLArray(θh), L, s, 1e-8; ntrans = 1)
        Gd = JLArrays.JLArray(zeros(ComplexF64, length(θh), 2L + 1, 1))
        PartialFSHT.spin_rings_from_alm!(Gd, JLArrays.JLArray(almh), engd)
        Test.@test Array(Gd) == Gh

        ah = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        PartialFSHT.spin_alm_from_rings!(ah, Gh, engh)
        ad = JLArrays.JLArray(zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1))
        PartialFSHT.spin_alm_from_rings!(ad, JLArrays.JLArray(Gh), engd)
        Test.@test maximum(abs.(Array(ad) .- ah)) <= 1e-12 * maximum(abs.(ah))

        Gr = randn(ComplexF64, length(θh), 2L + 1, 1)
        ar = JLArrays.JLArray(zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1))
        PartialFSHT.spin_alm_from_rings!(ar, JLArrays.JLArray(Gr), engd)
        lhs = LinearAlgebra.dot(Gh, Gr)
        rhs = LinearAlgebra.dot(almh, Array(ar))
        Test.@test abs(lhs - rhs) <= 1e-11 * abs(lhs)
    end
end

Test.@testset "a host engine with device buffers is refused, not run" begin
    # An engine built from a host ring vector has host-resident tables. Handing it a device grid
    # is a launch failure on real hardware and would silently work on a CPU-backed KA backend, so
    # it is checked rather than left to the launch.
    L = 32
    θh = sort(0.05 .+ 0.4 .* rand(21))
    engh = PartialFSHT.RingLegendre(θh, L, 1e-8; ntrans = 1)
    almd = JLArrays.JLArray(zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1))
    Gd = JLArrays.JLArray(zeros(ComplexF64, length(θh), 2L + 1, 1))
    Test.@test_throws ArgumentError PartialFSHT.rings_from_alm!(Gd, almd, engh)

    sengh = PartialFSHT.SpinRings(θh, L, 2, 1e-8; ntrans = 1)
    Test.@test_throws ArgumentError PartialFSHT.spin_rings_from_alm!(Gd, almd, sengh)
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

Test.@testset "threaded ring pass equals serial bit-for-bit" begin
    # Orders are independent and each is summed in the same order regardless of which task runs
    # it, so the threaded walk must agree *exactly*, not to a tolerance. Anything else means a
    # shared scratch buffer or a reordered reduction.
    Random.seed!(87)
    be_s = ComputationalBackends.SerialBackend()
    be_t = ComputationalBackends.ThreadedBackend()
    for L in (64, 257)
        θr = sort(0.01 .+ 0.5 .* rand(53))
        alm = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        for ℓ in 0:L, m in -ℓ:ℓ
            alm[PartialFSHT.sph_mode(ℓ, m), 1] = randn(ComplexF64)
        end

        eng = PartialFSHT.RingLegendre(θr, L, 1e-8; ntrans = 1)
        Gs = zeros(ComplexF64, length(θr), 2L + 1, 1)
        Gt = similar(Gs)
        PartialFSHT.rings_from_alm!(Gs, alm, eng, be_s)
        PartialFSHT.rings_from_alm!(Gt, alm, eng, be_t)
        Test.@test Gs == Gt

        as = zeros(ComplexF64, PartialFSHT.coeff_size(L)..., 1)
        at = similar(as)
        PartialFSHT.alm_from_rings!(as, Gs, eng, be_s)
        PartialFSHT.alm_from_rings!(at, Gs, eng, be_t)
        Test.@test as == at

        for s in (1, -2)
            seng = PartialFSHT.SpinRings(θr, L, s, 1e-8; ntrans = 1)
            Hs = zeros(ComplexF64, length(θr), 2L + 1, 1)
            Ht = similar(Hs)
            PartialFSHT.spin_rings_from_alm!(Hs, alm, seng, be_s)
            PartialFSHT.spin_rings_from_alm!(Ht, alm, seng, be_t)
            Test.@test Hs == Ht
        end
    end
end

Test.@testset "threaded stretch apply equals serial" begin
    Random.seed!(88)
    L = 512
    n = 400
    θ = (20 * π / 180) .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    kw = (lmax = L, method = PartialFSHT.StretchAlgorithm(),
          nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
    ps = PartialFSHT.Plan(θ, φ; kw..., backend = ComputationalBackends.SerialBackend())
    pt = PartialFSHT.Plan(θ, φ; kw..., backend = ComputationalBackends.ThreadedBackend())
    alm = rand_alm(L)
    Test.@test maximum(abs.((pt * alm) .- (ps * alm))) == 0
    fr = randn(ComplexF64, n)
    Test.@test maximum(abs.((pt' * fr) .- (ps' * fr))) == 0
    # thread count actually available, so a green suite at -t1 is not mistaken for coverage
    Test.@test PartialFSHT.nslots(ps.leaf.eng) >= 1
    if Threads.nthreads() > 1
        Test.@test PartialFSHT.nslots(pt.leaf.eng) > 1
    end
end
