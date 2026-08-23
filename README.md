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
`NUFSHT.nusht_type2!` on the *same* points. Measured by `benchmark/calibrate.jl` at 8 threads
(both sides thread); the machine was under other load, so absolute times are pessimistic.

| `lmax` | `M` | serial | threaded | NUFSHT | vs serial | vs threaded |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3.2e4 | 6.2 ms | 4.0 ms | 17.5 ms | 2.8× | 4.4× |
| 512 | 1.3e5 | 23.0 ms | 13.9 ms | 84.8 ms | 3.7× | 6.1× |
| 1024 | 5.1e5 | 128 ms | 65 ms | 416 ms | 3.3× | 6.4× |
| 2048 | 2.0e6 | 633 ms | 286 ms | 2.61 s | 4.1× | 9.1× |

Plan construction is 6.2 s versus 75 s at `lmax = 2048`.

Threading is over spherical harmonic order in the ring pass, which scales at **3.99× on 4
physical cores** (8 logical) and gives 2.0–2.2× end to end — the NUFFT is now the co-equal share
of an apply. The threaded result is **bit-identical** to the serial one, since orders are
independent and each is summed in the same sequence regardless of which task runs it.

Batching raises arithmetic intensity, because one `P̄` value then feeds `2·ntrans` accumulations
against unchanged recurrence-state traffic: `ntrans = 4` costs 1.43 ns per retained
`(degree, order, ring)` triple against 1.54 ns at `ntrans = 1`.

The speedup is close to **independent of point density**. Sweeping `M` over a 16× range at
`lmax = 2048` moves it only within measurement scatter (8–12×, against 2.5–3.3 s of run-to-run
variation in the NUFSHT baseline on a shared machine), and the NUFFT stage grows just 1.36× over
that range — it is dominated by its internal oversampled FFT on the reduced grid rather than by
point spreading. So a Nyquist-sampled patch and a 4×-oversampled one cost about the same, and the
table above is not an artefact of its sampling choice.

Applies allocate nothing after construction — both directions, every leaf, at any thread count.
FFT plans are deliberately built single-threaded: a multithreaded FFTW plan spawns Julia tasks on
every execution (4 kB per 2-D transform), and the FFT is 2% of an apply while the ring pass, which
is threaded directly, is 56%. Pass `fft_threads` to override.

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
`NonuniformFFTsBackend()`, `AutoNUFFT()`. `AutoNUFFT` errors when no fast backend is loaded
rather than silently selecting the `O(M·n₁n₂)` reference — and only leaves that use a NUFFT
demand one, so a `DirectAlgorithm` plan needs no extension.

## Parallelism and devices

Execution backends come from ComputationalBackends.

| backend | what it parallelises | needs |
| --- | --- | --- |
| `SerialBackend` | — | — |
| `ThreadedBackend` | spherical harmonic order in the ring pass; boxes in a multi-box leaf | `using OhMyThreads` |
| `MPIBackend` | points across ranks — synthesis local, adjoint `Allreduce`d | `using MPI` |
| `DistributedBackend` | independent node sets across worker processes, via `mul_nodeset` | `using Distributed` |

The ring pass and the DFS periodisation are KernelAbstractions kernels, so they run on any
device backend (`using KernelAbstractions`). The ring kernel parallelises over `(ring, order)`
with the whole recurrence state — including the block-scaling exponent — in registers; the
forward direction is race-free and reproduces the host result bit-for-bit, the adjoint scatters
through atomic adds and so agrees to round-off.

**Validation status of the device path is worth being precise about.** The ring pass and DFS are
tested against the host on `JLArrays` in the normal suite. The FFT and NUFFT stages cannot be
exercised there — `JLArrays` provides no FFT — so an end-to-end device transform needs real
hardware. Two test entry points are deliberately separate from `runtests.jl`, because a
single-rank or CPU-only run of them would assert nothing:

```
mpiexec -n 2 julia --project=test test/mpi_ranks.jl    # MPI point decomposition
julia --project=test benchmark/calibrate.jl            # performance contract + cost constants
```

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
