using Test: Test
using Random: Random
using LinearAlgebra: LinearAlgebra
using FastSphericalHarmonics: FastSphericalHarmonics
using PartialFSHT: PartialFSHT
using ComputationalBackends: ComputationalBackends
using Aqua: Aqua
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using KernelAbstractions: KernelAbstractions
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArrays
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using Distributed: Distributed

"""
Random coefficients in the *valid* modes only. `coeff_size(lmax)` is `(lmax+1)×(2lmax+1)` but
only `(lmax+1)²` of those slots are real modes in the `sph_mode` layout; the rest are padding the
transform never reads. Filling them with `randn(coeff_size(L)...)` is harmless when comparing
fields, and wrong when comparing coefficients — an unrecoverable padding slot then dominates a
`maximum`.
"""
function rand_alm(lmax::Integer; ntrans::Integer = 1)
    a = ntrans == 1 ? PartialFSHT.zeros_alm(lmax) :
        PartialFSHT.zeros_alm(ComplexF64, lmax; ntrans = ntrans)
    for b in 1:ntrans, ℓ in 0:Int(lmax), m in -ℓ:ℓ
        idx = PartialFSHT.sph_mode(ℓ, m)
        if ntrans == 1
            a[idx] = randn(ComplexF64)
        else
            a[idx, b] = randn(ComplexF64)
        end
    end
    return a
end

"""Max relative difference over the valid modes only."""
function alm_reldiff(a, b, lmax::Integer)
    num = 0.0
    den = 0.0
    for ℓ in 0:Int(lmax), m in -ℓ:ℓ
        idx = PartialFSHT.sph_mode(ℓ, m)
        num = max(num, abs(a[idx] - b[idx]))
        den = max(den, abs(b[idx]))
    end
    return num / max(den, eps())
end

Test.@testset "PartialFSHT.jl" begin
    include("test_legendre.jl")
    include("test_geometry_windows.jl")
    include("test_transforms.jl")
    include("test_spin.jl")
    include("test_analysis.jl")
    include("test_cost.jl")
    include("test_backends.jl")
    include("test_distributed.jl")
    include("test_allocs.jl")

    Test.@testset "Aqua quality checks" begin
        Aqua.test_all(PartialFSHT; ambiguities = false, deps_compat = (check_extras = false,))
    end
end
