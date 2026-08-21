# Rank experiments A–D at small L (CPU). Prints tables; does not write figures.
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using Random: Random

function cap_samples(n, θstar)
    θ = θstar .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    return θ, φ
end

function experiment_A(; lmax = 16, θstar = 20π / 180, n = 200)
    Random.seed!(1)
    θ, φ = cap_samples(n, θstar)
    A = PartialFSHT.synthesis_matrix(θ, φ, lmax)
    cap = PartialFSHT.bounding_cap(θ, φ)
    w = PartialFSHT.cap_quadrature_weights(θ, φ, cap)
    WA = PartialFSHT.weighted_synthesis_matrix(θ, φ, lmax, w)
    sA = LinearAlgebra.svdvals(A)
    sW = LinearAlgebra.svdvals(WA)
    Ns = PartialFSHT.shannon_number(lmax, cap.θstar)
    println("A  L=$lmax θ*=$(θstar) Ns=$(round(Ns; digits=1)) rank(A,1e-3)=$(PartialFSHT.numerical_rank(sA, 1e-3)) rank(W^{1/2}A,1e-3)=$(PartialFSHT.numerical_rank(sW, 1e-3))")
end

function experiment_B(; lmax = 16, θstar = 20π / 180, n = 200)
    Random.seed!(2)
    θ, φ = cap_samples(n, θstar)
    ranks = PartialFSHT.per_degree_ranks(θ, φ, lmax, 1e-3)
    println("B  per-degree ranks: ", ranks)
end

function experiment_C(; lmax = 32, θstar = 20π / 180, n = 64, m = 2)
    Random.seed!(3)
    θ = range(0, θstar; length = n) |> collect
    P = PartialFSHT.alf_block(θ, m, lmax)
    R = PartialFSHT.complementary_block_ranks(P, 4, 1e-4)
    println("C  ALF complementary block ranks (m=$m):")
    show(stdout, "text/plain", R)
    println()
    println("C  stretched FFT already saturates this rank; no butterfly leaf")
end

function experiment_D(; lmax = 12, n = 200)
    Random.seed!(4)
    masks = (
        cap = (0.3 .* sqrt.(rand(n)), 2π .* rand(n)),
        band = (π / 2 .+ 0.05 .* (rand(n) .- 0.5), 2π .* rand(n)),
        two_cap = (vcat(0.2 .* sqrt.(rand(n ÷ 2)), π .- 0.2 .* sqrt.(rand(n - n ÷ 2))), 2π .* rand(n)),
        random10 = begin
            nt, nφ = 36, 72
            keep = rand(nt * nφ) .< 0.1
            θg = [π * (i + 0.5) / nt for i in 0:nt-1, j in 0:nφ-1]
            φg = [2π * j / nφ for i in 0:nt-1, j in 0:nφ-1]
            (θg[keep], φg[keep])
        end,
        ocean = begin
            θo = acos.(1 .- 2 .* rand(n))
            φo = 2π .* rand(n)
            keep = (θo .> 0.6) .& .!((φo .> 1.2) .& (φo .< 2.4) .& (θo .< 2.2))
            (θo[keep], φo[keep])
        end,
    )
    for (name, (θ, φ)) in pairs(masks)
        cap = PartialFSHT.bounding_cap(θ, φ)
        groups = PartialFSHT.cluster_points(θ, φ; max_k = 8)
        clustered = sum(PartialFSHT.bounding_cap(view(θ, g), view(φ, g)).θstar^2 for g in groups)
        println("D  $name  bounding θ*=$(round(cap.θstar; digits=3))  k=$(length(groups))  Σθc²=$(round(clustered; digits=3))")
    end
end

function main()
    experiment_A()
    experiment_B()
    experiment_C()
    experiment_D()
end

main()
