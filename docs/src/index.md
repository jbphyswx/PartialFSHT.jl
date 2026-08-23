# PartialFSHT.jl

Partial-sky spherical harmonic transforms: the operator

```math
(\mathcal{Y}_R a)_i = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\Omega_i)
```

on an arbitrary point set, with its exact Euclidean adjoint.

```julia
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using FINUFFT: FINUFFT

plan = PartialFSHT.Plan(θ, φ; lmax = 2048, tol = 1e-8)
LinearAlgebra.mul!(f, plan, alm)      # synthesis
LinearAlgebra.mul!(alm, plan', f)     # exact adjoint
```

Evaluate the degree-`lmax` field on the few rings that meet the data's colatitude interval,
apodise, periodise, interpolate with one small NUFFT — Carron & Reinecke's construction
([arXiv:2603.17166](https://arxiv.org/abs/2603.17166)) with **no coefficient rotation**, because a
per-apply Wigner rotation costs about eight times the ring work it saves.

![geometry](assets/geometry.png)

## Start here

- **[Guide](guide.md)** — install, first transform, which call you actually want, and the
  parallelism and allocation knobs.
- **[Theory](theory.md)** — the derivations: scaled Legendre recurrences, the apodisation window,
  periodisation, spin, and the two front ends that were measured and rejected.
- **[Recoverability](recoverability.md)** — what partial data can and cannot give you. Read this
  before treating `plan' * f` as coefficients.
- **[Implementation notes](internals.md)** — the parts where a correct algorithm can still be
  built wrongly, with the measurements.
- **[API](api.md)** — every exported name.

Runnable examples live in
[`examples/`](https://github.com/jbphyswx/PartialFSHT.jl/tree/main/examples); each asserts the
property it demonstrates.

## The public surface is small

```julia
mul!(f, plan, alm)          # fast partial-geometry synthesis
mul!(alm, plan', f)         # exact Euclidean adjoint
solve!(alm, plan, f; mode)  # coefficients, with the regularisation named
spectrum(plan, f, w, Wℓ)    # band-powers, coupling handled internally
```

Everything returns plain arrays of spherical harmonic coefficients or `ℓ`-indexed band-powers. No
Slepian labels, no wrapper types, nothing to carry around alongside the numbers.

## Two things worth knowing immediately

**Synthesis and the adjoint are exact at any sky coverage.** There is no support condition, because
synthesis is a function *evaluation* — the coefficients already define the field everywhere.

**Inversion is not.** Only `~N_S = (lmax+1)² · f_sky` coefficient directions are constrained, and
that fraction is `f_sky` independently of `ℓ`. So `plan' * f` is *not* "the coefficients of my
data" — it is the exact adjoint, and therefore the *masked* field's coefficients. Band-powers via
[`spectrum`](@ref) are the well-posed product; see [Recoverability](recoverability.md).

```@docs
PartialFSHT.PartialFSHT
```
