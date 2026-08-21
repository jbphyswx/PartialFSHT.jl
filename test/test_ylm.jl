Test.@testset "closed-form Yℓm (Condon–Shortley orthonormal)" begin
    invsqrt4π = 1 / sqrt(4π)
    Test.@test PartialFSHT.ylm(0, 0, 0.3, 1.2) ≈ invsqrt4π
    θ, φ = 0.7, 1.1
    Test.@test PartialFSHT.ylm(1, 0, θ, φ) ≈ sqrt(3 / 4π) * cos(θ)
    Test.@test PartialFSHT.ylm(1, 1, θ, φ) ≈ -sqrt(3 / 8π) * sin(θ) * cis(φ)
    Test.@test PartialFSHT.ylm(1, -1, θ, φ) ≈ sqrt(3 / 8π) * sin(θ) * cis(-φ)
    Test.@test PartialFSHT.ylm(2, 2, θ, φ) ≈ sqrt(15 / 32π) * sin(θ)^2 * cis(2φ)
end
