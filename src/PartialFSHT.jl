"""
    PartialFSHT

Matrix-free partial-sphere spherical harmonic transforms: the operator
`𝒴_R: a_{ℓm} ↦ f(Ω_i)` on an arbitrary region, with Euclidean adjoint `plan'`.

Coefficients are complex orthonormal Condon–Shortley harmonics stored in
FastSphericalHarmonics `sph_mode` layout. Direct associated-Legendre evaluation is
the correctness oracle. Compact patches use Carron–Reinecke polar stretch;
irregular masks use clustered caps. Near-full sky may delegate to NUFSHT's public
API when that extension is loaded.
"""
module PartialFSHT

using ComputationalBackends: ComputationalBackends
using AbstractFFTs: AbstractFFTs
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra
using SpecialFunctions: SpecialFunctions

include("Layout.jl")
include("Direct.jl")
include("Geometry.jl")
include("Windows.jl")
include("PolarOpt.jl")
include("Wigner.jl")
include("DFS.jl")
include("NUFFT.jl")
include("Rings.jl")
include("StretchPlan.jl")
include("ClusterPlan.jl")
include("Cost.jl")
include("Conversion.jl")
include("Plan.jl")
include("Spin.jl")
include("Slepian.jl")
include("Rank.jl")

export Plan, zeros_alm, sph_mode, coeff_size, spin_coeff_index, ylm, sYlm
export AbstractAlgorithm, AutoAlgorithm, DirectAlgorithm, StretchAlgorithm, ClusterAlgorithm, NUFSHTAlgorithm
export DirectSumBackend, FINUFFTBackend, NonuniformFFTsBackend, AutoNUFFT
export BoundingCap, bounding_cap, stretch_params, polar_mmax, shannon_number, solve!
export algorithm

end
