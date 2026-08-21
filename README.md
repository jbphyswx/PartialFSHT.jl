# PartialFSHT.jl

Matrix-free spherical harmonic synthesis on an arbitrary point set \(R=\{\Omega_i\}\):

\[
(\mathcal{Y}_R a)_i = \sum_{\ell m} a_{\ell m}\, Y_{\ell m}(\Omega_i),
\]

with Euclidean adjoint `plan'`. Coefficients are complex orthonormal Condon–Shortley
harmonics in FastSphericalHarmonics `sph_mode` layout
(`(lmax+1) × (2lmax+1)`). This is not a quadrature inverse: `solve!` runs CG on
\(\mathcal{Y}^*\mathcal{Y}a=\mathcal{Y}^*f\) and lives in the Slepian subspace of \(R\).

```julia
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra

plan = PartialFSHT.Plan(θ, φ; lmax = 32)
LinearAlgebra.mul!(f, plan, alm)
LinearAlgebra.mul!(alm, plan', f)
```

`method` is an algorithm type: `AutoAlgorithm()` (default), `DirectAlgorithm()`,
`StretchAlgorithm()`, `ClusterAlgorithm()`, or `NUFSHTAlgorithm()` (needs `using NUFSHT`).

| Algorithm | When |
|-----------|------|
| Direct | Holmes–Featherstone associated Legendres; correctness oracle, tiny problems |
| Stretch | Carron–Reinecke polar stretch on a compact bounding cap |
| Cluster | One stretch leaf per spatially coherent cluster (bands, islands) |
| NUFSHT | Optional full-sphere leaf (scalar or spin; needs `using NUFSHT`) |

Spin-`s` uses the same `Plan(...; spin = s)` with Direct, stretch, cluster, or
NUFSHT. Spin coefficients use `spin_coeff_index(ℓ, m, lmax)`.

NUFFT backends are types too: `DirectSumBackend()`, `FINUFFTBackend()`,
`NonuniformFFTsBackend()`, `AutoNUFFT()`. Execution backends come from
ComputationalBackends (`SerialBackend`, `ThreadedBackend` via OhMyThreads,
`MPIBackend`, `DistributedBackend`).

Apply is bang-first (`mul!`); `*` allocates then calls `mul!`.

NUFSHT always pays a full-sphere Fourier transform of degree `L`, then interpolates to
the sample points. Carron stretch is for a **compact** bounding cap: after rotating
the cap to the pole, the interpolating Fourier sphere has degree
`Lmax ~ L θ*/π`, not `L`. Auto picks stretch on a compact cap while a thin `d(β)`
fits, cluster for a few compact islands, NUFSHT when the bounding hull is large,
the partition needs many leaves, or Wigner would be `O(L³)`. Direct when `M L²`
is tiny.

Measured `mul!` on a 20° cap, `M ≈ 4(L θ*)²`, FINUFFT, `tol=1e-8`, 8 threads
(`experiments/timing/run.jl`). Stretch vs Direct at `L=128` is `6.6e-4`.
Stretch wins on this cap through `L=256`. Ring longitude uses a planned FFT, and
a thin real `d(β)` is packed at construction while it fits in 1.5 GB
(`~1 GB` at `L=512`, `mmax=271`). Apply is then `O(L² mmax)` GEMVs. Setup fills
those planes with an `ℓ`-recurrence. At `L=1024` the packed cache is skipped and
apply streams the same recurrence through three dense planes (fill + GEMV).
That is still slower than NUFSHT, so Auto picks NUFSHT:

| L | M | Lmax | Direct | Stretch | NUFSHT | stretch/NUFSHT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.0e3 | 37 | 7.1 ms | 2.3 ms | 6.4 ms | 0.36 |
| 128 | 8.0e3 | 51 | 139 ms | 7.9 ms | 17 ms | 0.48 |
| 256 | 3.2e4 | 76 | — | 27 ms | 37 ms | 0.72 |
| 512 | 1.3e5 | 120 | — | 173 ms | 156 ms | 1.11 |
| 1024 | 5.1e5 | 198 | — | 2.78 s | 0.88 s | 3.14 |
| 2048 | 2.0e6 | 342 | — | 20 s | 8.3 s | 2.44 |

See `docs/` for the operator, rank experiments, and API.
