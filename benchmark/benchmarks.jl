using BenchmarkTools: BenchmarkTools
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using Random: Random

function _cap(n, θstar)
    Random.seed!(0)
    return θstar .* sqrt.(rand(n)), 2π .* rand(n)
end

function suite()
    s = BenchmarkTools.BenchmarkGroup()
    lmax = 32
    θ, φ = _cap(400, 25 * π / 180)
    alm = randn(ComplexF64, PartialFSHT.coeff_size(lmax)...)
    for (name, meth) in (("direct", PartialFSHT.DirectAlgorithm()),
                         ("stretch", PartialFSHT.StretchAlgorithm()))
        plan = PartialFSHT.Plan(θ, φ; lmax = lmax, method = meth,
                                nufft = PartialFSHT.DirectSumBackend(), tol = 1e-8)
        f = zeros(ComplexF64, length(θ))
        s[name] = BenchmarkTools.@benchmarkable LinearAlgebra.mul!($f, $plan, $alm)
    end
    return s
end
