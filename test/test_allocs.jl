_a_mul(f, p, a) = (LinearAlgebra.mul!(f, p, a); @allocated LinearAlgebra.mul!(f, p, a); @allocated LinearAlgebra.mul!(f, p, a))
_a_adj(a, p, f) = (LinearAlgebra.mul!(a, p', f); @allocated LinearAlgebra.mul!(a, p', f); @allocated LinearAlgebra.mul!(a, p', f))

Test.@testset "zero-allocation Direct apply" begin
    Random.seed!(9)
    lmax = 6
    n = 32
    θ = 0.3 .* rand(n)
    φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = lmax, method = PartialFSHT.DirectAlgorithm())
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    f = zeros(ComplexF64, n)
    LinearAlgebra.mul!(f, plan, alm)
    LinearAlgebra.mul!(alm, plan', f)
    if Threads.nthreads() == 1
        Test.@test _a_mul(f, plan, alm) == 0
        Test.@test _a_adj(alm, plan, f) == 0
    end
end
