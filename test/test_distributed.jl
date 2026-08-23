Test.@testset "Distributed node-set farming" begin
    Test.@test PartialFSHT._ext_loaded(:PartialFSHTDistributedExt)

    # A prebuilt plan holds FFT and NUFFT state that does not serialise, so applying one across
    # workers must fail loudly rather than silently rebuild or corrupt it.
    Random.seed!(91)
    L = 8
    θ = 0.3 .* rand(30)
    φ = 2π .* rand(30)
    plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm(),
                            backend = ComputationalBackends.DistributedBackend())
    alm = rand_alm(L)
    f = zeros(ComplexF64, 30)
    Test.@test_throws ArgumentError LinearAlgebra.mul!(f, plan, alm)
    Test.@test_throws ArgumentError LinearAlgebra.mul!(alm, plan', f)

    # `mul_nodeset` is the supported path: each worker builds its own plan from its node set.
    θs = [0.25 .* rand(24), 0.4 .* rand(31) .+ 0.2, 0.15 .* rand(18)]
    φs = [2π .* rand(length(t)) for t in θs]
    alms = [rand_alm(L) for _ in 1:3]
    ref = [PartialFSHT.Plan(θs[i], φs[i]; lmax = L,
                            method = PartialFSHT.DirectAlgorithm()) * alms[i] for i in 1:3]

    # With no workers the extension falls back to serial, which must give the same answer.
    got0 = PartialFSHT.mul_nodeset(θs, φs, alms, L,
                                   ComputationalBackends.DistributedBackend();
                                   method = PartialFSHT.DirectAlgorithm())
    for i in 1:3
        Test.@test maximum(abs.(got0[i] .- ref[i])) < 1e-12
    end

    added = Int[]
    try
        added = Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
        Test.@test length(added) == 2
        Distributed.@everywhere added begin
            using PartialFSHT: PartialFSHT
            using Distributed: Distributed
        end
        got = PartialFSHT.mul_nodeset(θs, φs, alms, L,
                                      ComputationalBackends.DistributedBackend();
                                      method = PartialFSHT.DirectAlgorithm())
        Test.@test length(got) == 3
        for i in 1:3
            Test.@test maximum(abs.(got[i] .- ref[i])) < 1e-12
        end
    finally
        isempty(added) || Distributed.rmprocs(added)
    end
end
