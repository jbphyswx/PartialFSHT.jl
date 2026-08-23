# PartialFSHT.jl

Partial-sky spherical harmonic transforms: ``\mathcal{Y}_R a = f|_R`` on an arbitrary point set,
with the exact Euclidean adjoint `plan'`.

```@contents
Pages = ["theory.md", "recoverability.md", "api.md"]
Depth = 2
```

## Quick start

```julia
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra

plan = PartialFSHT.Plan(θ, φ; lmax = 2048, tol = 1e-8)
LinearAlgebra.mul!(f, plan, alm)      # synthesis
LinearAlgebra.mul!(alm, plan', f)     # exact adjoint
```

The public surface is four calls — `mul!` in both directions, [`solve!`](@ref) and
[`spectrum`](@ref) — all in spherical harmonics, all returning plain arrays. Everything else in
this manual is implementation.

```@docs
PartialFSHT.PartialFSHT
```
