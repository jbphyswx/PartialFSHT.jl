using BenchmarkTools: BenchmarkTools
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using Random: Random

"""
Apply-time suite. Wall-clock contracts live here, never in `test/` — see
`benchmark/calibrate.jl` for the measured speedup table and the cost-model constants.
"""
function suite()
    s = BenchmarkTools.BenchmarkGroup()
    Random.seed!(0)
    θstar = 20 * π / 180
    for L in (256, 512)
        n = 4 * round(Int, (L * θstar)^2)
        θ = θstar .* sqrt.(rand(n))
        φ = 2π .* rand(n)
        alm = PartialFSHT.zeros_alm(L)
        for ℓ in 0:L, m in -ℓ:ℓ
            alm[PartialFSHT.sph_mode(ℓ, m)] = randn(ComplexF64)
        end
        f = zeros(ComplexF64, n)
        plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.StretchAlgorithm(),
                                tol = 1e-8)
        s["stretch/synth/L=$L"] = BenchmarkTools.@benchmarkable LinearAlgebra.mul!($f, $plan, $alm)
        s["stretch/adj/L=$L"] = BenchmarkTools.@benchmarkable LinearAlgebra.mul!($alm, $plan', $f)
    end
    return s
end
