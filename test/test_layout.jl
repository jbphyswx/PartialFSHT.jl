Test.@testset "coefficient layout and buffer types" begin
    lmax = 4
    Test.@test PartialFSHT.coeff_size(lmax) == (5, 9)
    Test.@test PartialFSHT.sph_mode(2, 0) == FastSphericalHarmonics.sph_mode(2, 0)
    alm = PartialFSHT.zeros_alm(lmax)
    Test.@test size(alm) == (5, 9)
    Test.@test eltype(alm) == ComplexF64

    θ = rand(11)
    φ = 2π .* rand(11)
    scratch = PartialFSHT.DirectScratch(θ, φ)
    Test.@test typeof(scratch.cosθ) == typeof(similar(θ, Float64))
    Test.@test typeof(scratch.eipφ) == typeof(similar(θ, ComplexF64))

    θv = view(θ, 1:5)
    φv = view(φ, 1:5)
    scratchv = PartialFSHT.DirectScratch(θv, φv)
    Test.@test scratchv.cosθ isa Vector{Float64}

    almθ = PartialFSHT.zeros_alm(θ, 3)
    Test.@test eltype(almθ) <: Complex
    Test.@test typeof(almθ) <: Array
end
