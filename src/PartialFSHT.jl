"""
    PartialFSHT

Partial-sky spherical harmonic transforms: the operator `𝒴_R : a_{ℓm} ↦ f(Ω_i)` on an arbitrary
point set, with its exact Euclidean adjoint `plan'`.

Coefficients are complex orthonormal Condon–Shortley harmonics in FastSphericalHarmonics
`sph_mode` layout. The fast path is a stretched colatitude interval — the field on the few rings
that meet the data, apodised, periodised and interpolated with one small NUFFT (Carron &
Reinecke, arXiv:2603.17166) — with no coefficient rotation anywhere in the apply. Direct
evaluation is the oracle and shares the same block-scaled Legendre recurrence.

Synthesis and the adjoint are exact at any sky coverage. *Inversion* from a region is not:
only `~N_S = (lmax+1)² f_sky` directions are constrained, so [`solve!`](@ref) names its
regularisation and [`recoverability`](@ref) reports what the geometry supports before any work
is spent. For "which wavenumbers carry what", [`spectrum`](@ref) is the well-posed answer.
"""
module PartialFSHT

using ComputationalBackends: ComputationalBackends
using AbstractFFTs: AbstractFFTs
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
using LinearAlgebra: LinearAlgebra
using SpecialFunctions: SpecialFunctions

include("Layout.jl")
include("Legendre.jl")
include("Box.jl")
include("Windows.jl")
include("DFS.jl")
include("NUFFT.jl")
# Spin before Stretch: the stretched leaf dispatches on `SpinRings` in a method signature.
include("Spin.jl")
include("Stretch.jl")
include("Cost.jl")
include("Direct.jl")
include("MultiBox.jl")
include("Conversion.jl")
include("Plan.jl")
include("Analysis.jl")

export Plan, zeros_alm, coeff_size, sph_mode, spin_coeff_index
export ylm, sYlm, alf, wigner_d, polar_mmax
export AbstractAlgorithm, AutoAlgorithm, DirectAlgorithm, StretchAlgorithm,
       MultiBoxAlgorithm, NUFSHTAlgorithm, algorithm
export DirectSumBackend, FINUFFTBackend, NonuniformFFTsBackend, AutoNUFFT
export SphereBox, bounding_box, box_fsky, box_theta_extent, box_phi_extent,
       shannon_number, spectral_resolution, partition_boxes
export StretchGeometry, stretch_geometry, stretch_geometry_phi, stretch_window, next_smooth, next_smooth_even
export recoverability, apply_resolution!, spectrum, pseudo_spectrum!, mask_coupling,
       mask_coupling!, bin_coupling, band_edges, solve!, synthesis_matrix,
       AnalysisWorkspace, threej000_sq_log
export ring_work, mul_nodeset, set_cost_constants!, cost_calibrated

end
