Test.@testset "rank laboratory (small L)" begin
    Random.seed!(11)
    lmax = 8
    θstar = 20 * π / 180
    n = 80
    θ = θstar .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    A = PartialFSHT.synthesis_matrix(θ, φ, lmax)
    s = LinearAlgebra.svdvals(A)
    cap = PartialFSHT.bounding_cap(θ, φ)
    Ns = PartialFSHT.shannon_number(lmax, cap.θstar)
    r = PartialFSHT.numerical_rank(s, 1e-3)
    Test.@test r <= size(A, 2)
    Test.@test Ns > 1

    ranks = PartialFSHT.per_degree_ranks(θ, φ, lmax, 1e-3)
    Test.@test length(ranks) == lmax + 1
    Test.@test ranks[1] == 1
    Test.@test ranks[end] <= 2lmax + 1

    P = PartialFSHT.alf_block(θ, 2, lmax)
    R = PartialFSHT.complementary_block_ranks(P, 4, 1e-3)
    Test.@test size(R) == (4, 4)
end
