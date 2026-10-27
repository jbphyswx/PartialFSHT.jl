```@meta
CurrentModule = PartialFSHT
```

# API

## The operator

```@docs
Plan
solve!
spectrum
recoverability
apply_resolution!
```

## Coefficients

```@docs
zeros_alm
coeff_size
sph_mode
spin_coeff_index
```

## Direct evaluation

```@docs
ylm
sYlm
alf
wigner_d
polar_mmax
```

## Algorithms and cost

```@docs
AbstractAlgorithm
AutoAlgorithm
DirectAlgorithm
StretchAlgorithm
MultiBoxAlgorithm
NUFSHTAlgorithm
algorithm
ring_work
set_cost_constants!
cost_calibrated
```

## Geometry and periodisation

```@docs
SphereBox
bounding_box
box_fsky
box_theta_extent
box_phi_extent
partition_boxes
shannon_number
spectral_resolution
StretchGeometry
stretch_geometry
stretch_geometry_phi
stretch_window
PartialFSHT.es_beta
PartialFSHT.es_cdf
next_smooth
next_smooth_even
```

## Spectral estimation

```@docs
pseudo_spectrum!
mask_coupling
mask_coupling!
bin_coupling
band_edges
threej000_sq_log
AnalysisWorkspace
synthesis_matrix
```

## Backends

```@docs
DirectSumBackend
FINUFFTBackend
NonuniformFFTsBackend
AutoNUFFT
mul_nodeset
```
