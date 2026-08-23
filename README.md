# PartialFSHT.jl

Partial-sky spherical harmonic transforms: the operator

```math
(\mathcal{Y}_R a)_i = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\Omega_i)
```

on an arbitrary point set, with its exact Euclidean adjoint. Coefficients are complex
orthonormal Condon–Shortley harmonics in FastSphericalHarmonics `sph_mode` layout.

```julia
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra

plan = PartialFSHT.Plan(θ, φ; lmax = 2048, tol = 1e-8)
LinearAlgebra.mul!(f, plan, alm)      # synthesis
LinearAlgebra.mul!(alm, plan', f)     # exact adjoint
```

## Performance

Synthesis on a 20° polar cap with `M ≈ 4(L·θ⋆)²` points, `tol = 1e-8`, against a full-sphere
`NUFSHT.nusht_type2!` on the *same* points. Measured by `benchmark/calibrate.jl`; the machine
was under other load, so both columns are pessimistic.

| `lmax` | `M` | PartialFSHT | NUFSHT | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 3.2e4 | 14.2 ms | 55.8 ms | 3.9× |
| 512 | 1.3e5 | 40.3 ms | 218 ms | 5.4× |
| 1024 | 5.1e5 | 208 ms | 1.22 s | 5.9× |
| 2048 | 2.0e6 | 1.01 s | 6.54 s | 6.5× |

Plan construction is 5.3 s versus 40 s at `lmax = 2048`. Applies allocate nothing after
construction, in both directions and for every leaf.

## Accuracy

Delivered accuracy is at or below the requested tolerance, verified against direct
associated-Legendre evaluation:

| `lmax` | `tol = 1e-4` | `tol = 1e-8` |
| ---: | ---: | ---: |
| 128 | 4.1e-6 | 3.5e-10 |
| 256 | 4.1e-6 | 4.2e-10 |
| 512 | 2.8e-6 | 3.0e-10 |
| 1024 | 1.6e-6 | 2.9e-10 |

The adjoint is the forward pipeline transposed stage by stage, so
`⟨𝒴a, f⟩ = ⟨a, 𝒴*f⟩` holds to round-off (2e-15) rather than to the transform tolerance. The
direct path is stable at every degree — `alf` agrees with `BigFloat` to 2e-12 at `ℓ = 8192`,
including orders near the turning point where a raw recurrence underflows to zero.

## Algorithm

```
alm ──ring ALF (polar-optimised, block-scaled)──▶ G[ring, m]
    ──apodise in θ──▶ periodise (DFS at a pole, two-sided otherwise)
    ──2-D FFT──▶ reduced Fourier torus ──type-2 NUFFT──▶ f(Ωᵢ)
```

Evaluate the degree-`lmax` field on the few rings that meet the data's colatitude interval,
apodise, periodise, interpolate. This is Carron & Reinecke's construction
([arXiv:2603.17166](https://arxiv.org/abs/2603.17166)) with **no coefficient rotation** — they
do not rotate either; they assume a frame with the patch at a pole. A per-apply Wigner rotation
costs about eight times the ring work it saves, which is enough to turn the method into a net
loss.

Geometry is a minimal colatitude × longitude box, which is exactly minimal and `O(n log n)`,
and matches what the transform exploits. An equatorial band is one box.

| method | when |
| --- | --- |
| `DirectAlgorithm` | scaled associated Legendre per point; the oracle, `O(M L²)` |
| `StretchAlgorithm` | one stretched colatitude interval — the workhorse |
| `MultiBoxAlgorithm` | one leaf per disjoint component of the point set |
| `NUFSHTAlgorithm` | full-sphere leaf via NUFSHT's public plans (needs `using NUFSHT`) |

`AutoAlgorithm` (the default) picks by *counted* cost — retained `(ℓ,m,ring)` triples, FFT
butterflies, NUFFT writes — with per-unit constants from `benchmark/calibrate.jl`.

Spin-`s` uses the same `Plan(...; spin = s)`; the Wigner-`d` there is the spin analogue of an
associated Legendre function (a degree recurrence at fixed `(m,−s)`), not a rotation.

NUFFT backends are types: `DirectSumBackend()` (reference), `FINUFFTBackend()`,
`NonuniformFFTsBackend()`, `AutoNUFFT()`. Execution backends come from ComputationalBackends.

## What partial data can and cannot give you

Synthesis and the adjoint are exact at **any** sky coverage. Inversion is not, and the package
says so rather than returning numbers that look like truth.

```julia
r = PartialFSHT.recoverability(θ, φ, lmax)   # fsky, Shannon number, recoverable fraction, Δℓ
```

- **The local spectrum is recoverable, well.** With `N_w` wavelengths across the patch,
  wavenumbers are pinned to `≈1/N_w`; resolution is `Δℓ ≈ π/Λ`. A 3000 km basin observing
  `ℓ ≈ 1000` gives `Δℓ ≈ 7` and ~670 independent modes per band, i.e. band-powers to ~5%.
  Use [`spectrum`](@ref), which accounts for the mask's own coupling.
- **Individual `a_ℓm` are not.** Only `~N_S = (lmax+1)²·f_sky` directions are constrained, and
  that fraction is `f_sky` *independently of `ℓ`* — small scales do not help, because higher `ℓ`
  brings proportionally more coefficients. It is a conditioning failure, not a uniqueness
  failure: exact noiseless data does determine the field, but the Slepian eigenvalues fall off a
  cliff past `N_S`.
- **`plan' * f` is not "the coefficients of my data."** It is the exact adjoint, and therefore
  the *masked* field's coefficients — mode-coupled and leakage-biased.

To compare partial observations against a global model in coefficient space, window both sides
identically — one extra forward transform — and compare in `(ℓ,m)`:

```julia
PartialFSHT.apply_resolution!(ã_model, plan, alm_model, w)   # A† W A
```

`solve!` names its regularisation (`:cg`, `:wiener`, `:minnorm`) because there is no
regularisation-free answer, and returns the Shannon number, `Δℓ`, iterations and residual
alongside the coefficients.

See `docs/` for the derivations and the recoverability discussion.
