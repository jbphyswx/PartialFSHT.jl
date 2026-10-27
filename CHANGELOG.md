# Changelog

## Unreleased

### Added

- `examples/` — eight runnable examples with their own environment and a runner. Each asserts the
  property it demonstrates, so a broken example fails rather than printing plausible numbers.
- `gpu/` — device-parity tests on real hardware. The main suite covers the three
  KernelAbstractions kernels against the host on `JLArrays`, but `JLArrays` has no FFT, so the FFT
  and NUFFT stages of a device apply are only reachable here.
- KernelAbstractions kernels for the **spin** ring pass, forward and adjoint. The forward direction
  is bit-identical to the host.
- Documentation: a Guide page, an Implementation-notes page, and six figures generated from the
  package by `docs/make_figures.jl` so they cannot drift from the behaviour they illustrate.
- `Plan` now forwards leaf options that were previously unreachable: `fft_threads`,
  `nufft_kwargs`, `oversample_θ`, `oversample_φ`, `εapo_max`.
- `try_stretch_geometry`, a non-throwing query form of `stretch_geometry` for the cost model.
- `partition_boxes_costed`, which returns the box and predicted cost it already computed per group.
- CI: test matrix, a threaded run, a 2-rank MPI job, an examples job, docs deployment, coverage,
  CompatHelper, TagBot and an environment-resolution check.

### Fixed

- `SpinDirectLeaf` construction was **quadratic in the point count** — `φ[perm]` was re-sliced on
  every iteration of a comprehension. 505 MB at 8000 points, and unusable at the point counts the
  README benchmarks.
- `pseudo_spectrum!` silently used only the first batch member with `ntrans > 1`; the member is now
  an explicit `batch` argument.
- `Plan` declared `θ` and `φ` with a single type parameter, so mixed inputs were silently converted
  and mixed precision silently narrowed.
- The NUFSHT leaf hardcoded `ComplexF64`, so a `Float32` plan widened on that leaf alone.
- `alf(ℓ, m, θ)` evaluated in `Float64` regardless of `θ`'s type.
- `mask_coupling` produced `Inf`/`NaN` for a mask spectrum reaching past `ℓ = lmax`: the tabulated
  `loggamma` was sized for `J ≤ 3·lmax` where the 3-j sum reaches `4·lmax`.
- Both direct oracles accumulated the azimuthal phase as a running product, drifting `O(L·eps)` in
  argument and modulus. They now form `cis(mφ)` exactly, since they are the reference the fast
  paths are asserted against at round-off level.
- `DistributedBackend` silently ran serially when no worker was present.
- `dfs_grid_coords` always returned `Float64`.

### Changed

- The north/south reflection is realised as one sign on `cosθ` at build time instead of a
  coefficient-shaped table of `±1` (64 MB at `lmax = 2048`) plus a full-array multiply per apply.
- `partition_boxes` no longer sorts: each largest gap comes from a bucketed `O(n)` maximum-gap scan.
  2.39 s and 198 MB → 0.48 s and 92 MB at two million points.
- `synthesis_matrix` runs the degree recurrence once per `(point, order)` instead of one full
  transform per column: measured exponent in `lmax` 3.96 → 2.0.
- `mask_coupling` tabulates `loggamma` and threads over rows: 2.56 s → 0.21 s at `lmax = 384`.
- The spin seed's log-binomial and per-ring logarithms are tabulated at construction, so the spin
  ring pass now costs the same as the scalar one instead of paying `loggamma` per (order, ring) on
  every apply.
- The order pack is two contiguous block copies instead of `2·m_max + 1` strided operations.
- Retained ring ranges come from one nested-interval walk in `O(nθ + lmax)` rather than a scan per
  order; the cost model and the engine now provably count the same way.
- `spectrum` accepts a precomputed `coupling`, since it is `O(L³)` and a function of the mask alone.
- File-header prose that was written as bare string literals — never reachable as documentation —
  is now either a comment or a section of the manual.

### Documentation corrections

Several published figures did not reproduce on a quiet machine and were replaced by measured ones:
threaded speedup at `lmax = 2048` is 8.35× rather than 9.1×, the ring pass threads at 2.3–3.6×
rather than 3.99×, and the blanket "applies allocate nothing at any thread count" is now stated
precisely — the package's own stages are 0 B, while the NUFFT backends and the threaded scheduler
each carry a documented, measured cost. Batching is 1.09–1.12× at `lmax ≤ 1024` and a 0.93× *loss*
at 2048, where the accumulators exceed L1.
