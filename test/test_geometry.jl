Test.@testset "bounding cap and clustering" begin
    θ = fill(0.2, 8)
    φ = range(0, 2π - 0.1; length = 8) |> collect
    cap = PartialFSHT.bounding_cap(θ, φ)
    Test.@test cap.θstar < 0.35
    Test.@test cap.θc < 0.3

    θ′, φ′ = PartialFSHT.rotate_to_pole(θ, φ, cap)
    Test.@test maximum(θ′) <= cap.θstar + 1e-12

    groups = PartialFSHT.cluster_points(θ, φ; k = 2)
    Test.@test length(groups) == 2
    Test.@test sort(vcat(groups...)) == 1:8
end
