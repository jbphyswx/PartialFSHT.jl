using Test: Test
using Random: Random
using LinearAlgebra: LinearAlgebra
using FastSphericalHarmonics: FastSphericalHarmonics
using PartialFSHT: PartialFSHT
using Aqua: Aqua
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using KernelAbstractions: KernelAbstractions
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArrays
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads
using ComputationalBackends: ComputationalBackends

Test.@testset "PartialFSHT.jl" begin
    include("test_ylm.jl")
    include("test_layout.jl")
    include("test_direct.jl")
    include("test_geometry.jl")
    include("test_wigner.jl")
    include("test_stretch.jl")
    include("test_cluster.jl")
    include("test_spin.jl")
    include("test_rank.jl")
    include("test_slepian.jl")
    include("test_allocs.jl")
    include("test_nufft.jl")
    include("test_ka.jl")
    include("test_nufsht.jl")
    include("test_threaded.jl")

    Test.@testset "Aqua quality checks" begin
        Aqua.test_all(PartialFSHT; ambiguities = false, deps_compat = (check_extras = false,))
    end
end
