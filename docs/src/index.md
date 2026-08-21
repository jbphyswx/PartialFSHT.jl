# PartialFSHT.jl

Matrix-free partial-sphere spherical harmonic transforms: \(\mathcal{Y}_R a = f|_R\)
with Euclidean adjoint `plan'`.

```@contents
Pages = ["theory.md", "ranks.md", "api.md"]
Depth = 2
```

## Quick start

```julia
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra

plan = PartialFSHT.Plan(θ, φ; lmax = 32)
LinearAlgebra.mul!(f, plan, alm)
LinearAlgebra.mul!(alm, plan', f)
```

```@docs
PartialFSHT.PartialFSHT
```
