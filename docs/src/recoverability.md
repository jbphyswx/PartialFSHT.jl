# Recoverability

Synthesis and the adjoint are exact at any sky coverage. *Inversion* is not, and the distinction
is the one thing a user of a partial-sky transform has to get right.

## Three different maps

With ``A_{i,(\ell m)} = Y_{\ell m}(\Omega_i)``:

| map | what it computes | exact on a patch? | is it "the coefficients"? |
| --- | --- | --- | --- |
| synthesis ``A`` | coefficients → values at the points | **yes**, always | n/a |
| adjoint ``A^\dagger`` | ``\sum_i \overline{Y_{\ell m}(\Omega_i)} f_i`` | **yes**, always | **no** |
| inversion | values → coefficients | **no** — rank-deficient | that is the goal, and it is ill-posed |

**The trap is the middle row.** ``A^\dagger f`` is exactly computable on a patch but it is not
the underlying field's harmonic coefficients — it is the *masked* field's, mode-coupled and
leakage-biased. Computing it and calling the result ``a_{\ell m}`` gives a wrong answer. What the
adjoint is correct for is being the exact transpose of ``A``: what an iterative solver, a
likelihood gradient, or a quadratic estimator needs.

Synthesis needs no support condition because it is a function *evaluation*, not a fit: the
coefficients already define the field everywhere, and the DFS extension of a degree-``L`` field is
an exact 2-D trigonometric polynomial of degree ``(L,L)``, so the NUFFT evaluates a known
polynomial from exact Fourier coefficients rather than interpolating scattered data.

## What is recoverable

- **The local spectrum, well.** With ``N_w = \Lambda/\lambda`` wavelengths across the patch,
  wavenumber magnitude and direction are determined to ``\approx 1/N_w``; the resolution is
  ``\Delta\ell \approx \pi/\Lambda``. The correction for the gaps is the direct analogue of the
  window correction in any 1-D periodogram — a matrix instead of a scalar only because a
  spherical window couples ``\ell``'s.
- **Individual ``a_{\ell m}``, no.** Only ``N_S = (L+1)^2 f_{\rm sky}`` directions are
  constrained. It is a *conditioning* failure, not a uniqueness failure: a degree-``L`` field is a
  polynomial restricted to ``S^2``, so exact noiseless patch data determines it — but recovering
  the ``k``-th direction needs data precision ``\sim\lambda_k``, and the Slepian eigenvalues fall
  off a cliff past ``N_S``, with a transition band growing only logarithmically in SNR.

**Small scales do not help.** Within any band the well-determined fraction is ``f_{\rm sky}``,
independently of ``\ell``, because higher ``\ell`` brings proportionally more coefficients. 300
wavelengths across a basin make the *spectrum* excellent without moving that fraction at all.

The barrier is continuous in ``f_{\rm sky}``:

| geometry | ``f_{\rm sky}`` | ``N_S/(L+1)^2`` | ``\Delta\ell \approx \pi/\Lambda`` |
| --- | ---: | ---: | ---: |
| 4°×4° patch, ``L = 3600`` | 3.9e-4 | 0.04% | ~45 |
| SPT-3G main | 0.04 | 4% | ~8 |
| Atlantic basin | ~0.2 | 20% | ~3 |
| hemisphere | 0.5 | 50% | ~1 |

[`recoverability`](@ref) reports this for a given point set before any work is spent.

## What to do instead

Ordered by how often it is the right answer.

1. **Compare against a global model in coefficient space** — window both sides identically and
   compare in ``(\ell,m)``. From the caller's side this is *one extra forward transform*:

   ```julia
   PartialFSHT.apply_resolution!(ã_model, plan, alm_model, w)   # A† W A
   ã_obs = plan' * (w .* f_obs)
   ```

   Both sides equal the same operator applied to their own field, so the comparison is exact and
   unbiased with nothing to deconvolve. This is forward-modelling the model into data space
   rather than deconvolving data into truth space.

2. **Energy per mode from a region** — a band-power spectrum with the mask's own coupling
   accounted for, via [`spectrum`](@ref). Individual ``\tilde a_{\ell m}`` are a known but
   non-invertible mixture; the *quadratic* aggregate has a tractable bias,
   ``\langle \tilde C_\ell\rangle = \sum_{\ell'} M_{\ell\ell'} C_{\ell'}``, with ``M`` built from
   the mask's own power spectrum. Nothing about the unobserved sky enters. Why this escapes the
   ill-posedness is counting: coefficients are ``(L+1)^2`` unknowns against ``\sim N_S``
   constraints, the spectrum is ``\sim L/\Delta\ell`` unknowns against the same constraints. The
   cost is **variance, not bias**: ``\Delta C_\ell/C_\ell \approx \sqrt{2/((2\ell+1) f_{\rm sky}
   \Delta\ell)}``.

3. **Work inside the patch** — `solve!(…, mode = :minnorm)`. Optimal there; its field decays
   outside over the reproducing-kernel scale ``\sim 1/L``, so it is the wrong choice if the
   coefficients then feed a global quantity.

4. **Need a global field** — `mode = :wiener` with a stated prior, i.e. a background and an
   assimilation step. Minimum-norm is the wrong answer here: a regional feature surrounded by
   dead calm provokes spurious adjustment.

## Two things not to do

**Zero-padding.** Masking a map to the patch, filling the rest with zeros and running a
full-sphere transform creates a *discontinuity* at the mask edge whose power-law spectral tail
aliases into the band. The bandlimited minimum-norm fit has the same "≈zero outside" but no edge
and no out-of-band power — the effective taper is the optimally-concentrated one, with mainlobe
``\Delta\ell \approx \pi/\Lambda`` and sidelobes at the ``(1-\lambda_k)`` level.

**Intersecting masks** when comparing two regions. Non-overlapping or chiral footprints have a
near-empty intersection, and when shape *is* the signal, intersecting destroys the experiment.
What must be common is the **spectral resolution**: bin both to the coarser ``\Delta\ell``, which
discards no data and merely declines to claim resolution the narrower region cannot support.

## On minimum-norm

It is the *least*-energy exact fit, not an extreme one, and ``AA^\dagger`` is well conditioned at
Nyquist patch sampling. The zero-extension is a property of the truncation level, not of
minimum-norm: with exact data, uniqueness means it returns the true field and continues correctly
outside. Note also that any *unique, deterministic* estimate must decay outside the data, since
the minimum-variance estimate of an unobserved zero-mean quantity is zero — "the pattern
continues similarly" is inherently a random draw. Unique and deterministic, or plausible
continuation; not both.
