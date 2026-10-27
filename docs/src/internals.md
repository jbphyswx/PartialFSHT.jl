```@meta
CurrentModule = PartialFSHT
```

# Implementation notes

What the code does and why, at the level a maintainer needs. The mathematics is in
[Theory](theory.md); this page is about the parts where a correct algorithm can still be built
wrongly, and about several designs that were measured and rejected.

## The ring engine

`src/Legendre.jl` holds the one associated-Legendre implementation in the package. The oracle uses
it too, with polar optimisation switched off, so an oracle and the path it certifies cannot drift
apart.

**Block scaling is not optional.** The sectorial seed carries ``\sin^m\theta``, which flushes to
zero in IEEE-754 long before it stops mattering — on a 20° cap at ``\ell_{\max}=4096`` it vanishes
at ``m=801`` while orders out to ``m=1636`` still hold 67% of the outermost ring's Legendre energy.
The recurrence therefore runs as mantissa × ``2^E`` with one shared exponent per ring chunk,
renormalised on a fixed stride, and the partial sums are carried in the same scale so the only
terms dropped are already negligible against the running maximum.

**Chunking is what makes the shared exponent safe.** A chunk may grow while
``m_{\max}\,|\log_2(\sin\theta_{hi}/\sin\theta_{lo})| \le`` `_CHUNK_HEADROOM_BITS`, which is exactly
the condition for renormalising against the chunk maximum not to flush its smallest ring. The spin
engine needs the same guarantee with its own span measure: its seed's ring dependence is
``p_c\log_2\cos(\theta/2) + p_s\log_2\sin(\theta/2)`` with ``p_c+p_s = 2\ell_0``, so it splits on
``2\ell_0`` times the larger log ratio.

**Retained ranges come from a nested-interval walk.** `mlim[j]` tracks ``\sin\theta``, so it rises
and then falls over ascending colatitude — it is *unimodal*, and the retained sets
``\{j : m_{\lim}[j] \ge m\}`` are therefore contiguous *and* nested as the order grows. One outward
walk from the peak produces every range in ``O(n_\theta + \ell_{\max})``. Scanning every ring for
every order is ``O(\ell_{\max}\,n_\theta)``, which is half a million comparisons at
``\ell_{\max}=2048`` inside a function the planner calls several times per plan. `_nested_ranges`
checks unimodality rather than assuming it.

**Fusion, and the two shapes that lost.** For a single transform the degree recurrence is fused with
its accumulation so each new ``\bar P`` is consumed in the register that produced it — worth about
10%, in both directions.

Two alternatives measured worse and are not used. Putting the ring loop outermost in blocks of a few
with the degree loop innermost and the state entirely in registers — the libsharp-style shape — is
**2× slower**, because it caps SIMD at the unroll width while the current loop vectorises across the
whole ring chunk. And extending the fusion to `ntrans > 1` is **0.64× at `ntrans = 2`**: it puts the
batch loop innermost, where `accp[k, b]` strides by ``n_r`` in ``b``, so neither loop vectorises. The
batched path keeps the two-pass form. See [Theory](theory.md) for both measurements.

## The reflection is a sign on `cosθ`

A south-pole periodisation maps increasing ``\Theta`` to *decreasing* ``\theta``. The ring engine
walks ascending ``\theta`` and the grid's ``\theta``-FFT needs ascending ``\Theta``, so those two
orders are reversed relative to each other — which silently transposes the grid and makes the result
``O(1)`` wrong. The fix is to reflect the whole problem to the north.

The reflection costs nothing. Since ``\sin(\pi-\theta) = \sin\theta``, the seed, the
polar-optimisation limits and the chunking are all unchanged, and the entire reflection is one sign
on ``\cos\theta``: the recurrence
``\bar P_\ell^m = a\cos\theta\,\bar P_{\ell-1}^m - b\,\bar P_{\ell-2}^m`` with
``\cos\theta \to -\cos\theta`` reproduces ``(-1)^{\ell+m}\bar P_\ell^m`` by induction from a seed
that does not involve ``\cos\theta`` at all. That *is* the reflection identity
``Y_{\ell m}(\pi-\theta,\varphi) = (-1)^{\ell+m} Y_{\ell m}(\theta,\varphi)``.

For spin the two half-angle factors swap as well (``\cos((\pi-\theta)/2) = \sin(\theta/2)``), which
realises ``{}_sY_{\ell m}(\pi-\theta,\varphi) = (-1)^{\ell+m}\,{}_{-s}Y_{\ell m}(\theta,\varphi)``
with the caller's original spin weight.

Doing this as a pass over the coefficients instead costs a ``(\ell_{\max}+1)\times(2\ell_{\max}+1)``
table of ``\pm 1`` — 64 MB at ``\ell_{\max}=2048`` — plus a full-array multiply on every apply, in
each direction.

## The order pack is two block copies

``m \mapsto \operatorname{mod}(m, n_\varphi) + 1`` looks like a scatter and is a *permutation onto
two contiguous blocks*: orders ``0..m_{\text{keep}}`` land in columns ``1..m_{\text{keep}}+1`` and
orders ``-m_{\text{keep}}..-1`` in ``n_\varphi-m_{\text{keep}}+1..n_\varphi``, and both source ranges
are contiguous too. Copying block-wise turns ``2m_{\max}+1`` strided operations into two — 2049 of
them at ``\ell_{\max}=1024``. On a host that is a modest win; on a device it is 2049 kernel launches
against two.

## Grid sizes

Every planner-chosen size is rounded up to 5-smooth, because oversampling only improves the
trigonometric interpolation while a large prime factor puts FFTW on a Rader/Bluestein path for
nothing.

The longitude grid must additionally be **even**. The DFS reflection realises
``\varphi \to \varphi+\pi`` as a shift of ``N_\varphi/2`` cells, which is a half-turn only for even
``N_\varphi``; at ``N_\varphi = 135`` it shifts 3.118 rad instead of 3.1416 and caps accuracy at
``10^{-5}`` whatever tolerance is requested. A pole or full-sphere periodisation is DFS-doubled, so
the ``\theta`` grid is rounded at the *half* and then doubled — rounding the doubled size can land
on an odd 5-smooth value (125 for 122) and then the doubling no longer matches the buffer.

## The planner

Costs are *counted* operations — retained ``(\ell,m,\text{ring})`` triples, ``n\log n`` butterflies,
NUFFT kernel writes — scaled by per-unit constants measured once by `benchmark/calibrate.jl`. There
are no thresholds on `f_sky` or on "is the patch compact": a band wins on the stretch leaf because
its colatitude interval is thin, a near-full sky loses to NUFSHT because the counted work converges
to NUFSHT's own. `counted_ring_work` and `ring_work(engine)` count the same way, and a test asserts
they agree exactly — otherwise a planner decision is made about work that never happens.

A degenerate geometry is a modelling answer, not an exception: `try_stretch_geometry` returns
`nothing` where `stretch_geometry` throws, so the cost model can ask without a `catch` that would
also swallow a genuinely bad argument.

**Partitioning does not sort.** Both the box and the split site need one quantity — the largest gap
in a coordinate — and that is exactly computable in ``O(n)`` by bucketing: distribute the values
into ``n-1`` equal buckets, and since the largest gap is at least the bucket width it cannot lie
*inside* a bucket, so one pass over the buckets in order finds it. Sorting per candidate cost
2.39 s and 198 MB at ``M = 2\times10^6``, to return a single box; the bucketed search is 0.48 s and
92 MB, which is under 10% of plan construction.

## Allocation

The package's own stages allocate nothing per apply, in either direction, at any thread count, and
`test/test_allocs.jl` gates that on every leaf against an explicit `SerialBackend` with
`DirectSumBackend` for the NUFFT seam — so the gate measures this package and not a dependency.

Two things outside that boundary do allocate, and the README says so rather than claiming a blanket
zero:

- **FFT plans are built at one thread on purpose.** A multithreaded FFTW plan spawns Julia tasks on
  *every execution* — FFTW.jl installs a Julia-task threading callback into the shared library — so
  a threaded FFT would make every apply allocate. `fft_threads` overrides it.
- **The fast NUFFT backends allocate per exec at more than one Julia thread**, for the same reason
  reached from the other side: FINUFFT's internal FFTW plan takes its thread count from its own
  OpenMP setting, and executing it goes through FFTW.jl's callback. Measured 4032 B per apply at
  `lmax = 256`–512 and four threads, from eight `Task` objects — confirmed by allocation profiling,
  which names `FFTW.spawnloop` as the site. Pinning FFTW's *global* thread count while the NUFFT
  plans are built does **not** remove it. NonuniformFFTs allocates per exec even single-threaded,
  from a `Threads.@threads` region in its deconvolution step. Both are upstream.

## Device support

The ring pass, the spin ring pass and the DFS periodisation are KernelAbstractions kernels,
parallelised over ``(\text{ring}, \text{order})`` with the whole recurrence state — including the
block-scaling exponent — in registers. The forward direction is race-free and reproduces the host
result bit-for-bit; the adjoint scatters through atomic adds on a real-reinterpreted view of the
coefficient array (there is no atomic add for a `Complex`, and two independent real adds are
equivalent because the parts never interact), so it agrees to round-off.

The engine is seeded from a ring vector of the *node set's* array type, so its tables and scratch
land on the same device as the buffers. Handing it a host vector while the grid is on the device is
a launch failure on real hardware and silently works on a CPU-backed KernelAbstractions backend,
which is why `_require_device` checks it rather than leaving it to the launch.

**Validation status.** The ring passes and the DFS are tested against the host on `JLArrays` in the
normal suite. `JLArrays` provides no FFT, so the FFT and NUFFT stages cannot be exercised there and
an end-to-end device transform still needs real hardware.

## Test entry points kept separate

Two would assert nothing in a single-rank or CPU-only run, so they are not in `runtests.jl`:

```
mpiexec -n 2 julia --project=test test/mpi_ranks.jl    # MPI point decomposition
julia --project=test benchmark/calibrate.jl            # performance contract + cost constants
```

## Internal reference

Everything below is implementation. Names are not part of the public surface and may change without
a version bump; they are documented because a maintainer reading the source needs them.

### The ring engines

```@docs
PartialFSHT.RingLegendre
PartialFSHT.RingScratch
PartialFSHT.rings_from_alm!
PartialFSHT.alm_from_rings!
PartialFSHT.nslots
PartialFSHT.retained_triples
PartialFSHT.sectorial_log2c
PartialFSHT.polar_pad_factor
PartialFSHT._nested_ranges
PartialFSHT._ring_chunks
PartialFSHT._walk_orders!
PartialFSHT._seed_exponent
PartialFSHT._renorm!
```

### Spin

```@docs
PartialFSHT.SpinRings
PartialFSHT.SpinScratch
PartialFSHT.spin_rings_from_alm!
PartialFSHT.spin_alm_from_rings!
PartialFSHT._spin_ring_chunks
PartialFSHT._spin_walk_orders!
PartialFSHT._spin_seed_log2
PartialFSHT._d_extremal
PartialFSHT._half_lbinom
```

### Leaves

```@docs
PartialFSHT.DirectLeaf
PartialFSHT.SpinDirectLeaf
PartialFSHT.StretchLeaf
PartialFSHT.MultiBoxLeaf
PartialFSHT.synthesize_direct!
PartialFSHT.adjoint_direct!
PartialFSHT.naccumulators
PartialFSHT.boxes
```

### The stretched grid

```@docs
PartialFSHT.stretch_rings
PartialFSHT.stretch_to_theta
PartialFSHT.theta_to_stretch
PartialFSHT.stretch_window!
PartialFSHT.try_stretch_geometry
PartialFSHT._order_blocks
PartialFSHT._rings_to_grid!
PartialFSHT._grid_to_rings!
PartialFSHT._periodize!
PartialFSHT._periodize_adjoint!
PartialFSHT._with_fftw_threads
```

### Periodisation

```@docs
PartialFSHT.dfs_double!
PartialFSHT.dfs_fold!
PartialFSHT.dfs_grid_coords
```

### Geometry

```@docs
PartialFSHT.box_wraps_phi
PartialFSHT.partition_boxes_costed
PartialFSHT._max_gap
PartialFSHT._GapWorkspace
PartialFSHT._group_box
PartialFSHT._best_gap_split
PartialFSHT._split_below
PartialFSHT._circular_extent
```

### Cost model

```@docs
PartialFSHT.CostConstants
PartialFSHT.counted_ring_work
PartialFSHT.cost_direct
PartialFSHT.cost_stretch
PartialFSHT.cost_nufsht
PartialFSHT.select_method
```

### Analysis internals

```@docs
PartialFSHT._loggamma_table
PartialFSHT._coeff_host
```
