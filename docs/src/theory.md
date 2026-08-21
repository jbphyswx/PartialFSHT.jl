# Theory

The public object is the synthesis operator on a point set \(R=\{\Omega_i\}\),

\[
(\mathcal{Y}_R a)_i
=
\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}
a_{\ell m}\,Y_{\ell m}(\Omega_i),
\]

with complex orthonormal Condon–Shortley \(Y_{\ell m}\). The Euclidean adjoint is

\[
(\mathcal{Y}_R^* f)_{\ell m}
=
\sum_i
\overline{Y_{\ell m}(\Omega_i)}\,f_i.
\]

This is **not** a quadrature inverse. `solve!` runs CG on \(\mathcal{Y}^*\mathcal{Y}a=\mathcal{Y}^*f\)
and therefore lives in the well-posed Slepian subspace of \(R\) without forming Slepian functions.

## Direct oracle

Holmes–Featherstone associated-Legendre recurrence, vectorized over points at fixed \(m\).
Cost \(O(M L^2)\). Used as the correctness reference and for tiny problems.

## Carron–Reinecke stretch

For a bounding polar cap of colatitude \(\theta_*\) and tolerance \(\varepsilon\):

\[
\varepsilon^{\mathrm{apo}}
=
\sqrt{\frac{2|\ln\varepsilon|}{\theta_* L}},
\qquad
\theta^{\max}=\theta_*(1+\varepsilon^{\mathrm{apo}}),
\qquad
L_{\max}
=
\Bigl\lfloor
L\frac{\theta_*}{\pi}(1+\varepsilon^{\mathrm{apo}})^2
\Bigr\rfloor.
\]

Apply: rotate `alm` so the cap sits at the pole (partial Wigner \(D\)), evaluate the
**degree-\(L\)** field on \(L_{\max}+1\) rings (polar-optimized \(|m|\)), multiply by an
exponential-of-semicircle window that is identically 1 on the user cap, DFS + FFT at
bandlimit \(L_{\max}\), NUFFT to the stretched coordinates \((\Theta_i,\phi_i)\).
The Euler angles are fixed on a leaf, so construction packs a thin real
\(d^\ell(\beta)\) of size \((2m_{\max}+1)\times(2\ell+1)\) per degree via an
\(\ell\)-recurrence and apply is GEMVs (\(O(L^2 m_{\max})\); Euclidean adjoint \(d^\top\)).
If that cache would exceed the packed-size budget, apply streams the same
recurrence through three dense planes (\(O(L m_{\max})\) RAM) and Auto prefers NUFSHT
when that apply is still slower than a full-sphere transform.
Spin-\(s\) uses the same pipeline with a Trapani–Navaza ring SHT
\(G_{jm}=\sum_\ell {}_s a_{\ell m} N_\ell d^\ell_{m,-s}(\theta_j)\), DFS factor \((-1)^s\),
and a pointwise \(e^{is\chi}\) that converts the rotated \(\hat\varphi\) basis back to the
caller's spherical frame.

Carron does **not** truncate the original `alm` to \(L_{\max}\). The ring SHT is still
degree \(L\); only the interpolating Fourier sphere is reduced.

## Clustered caps

When a single bounding cap is nearly the whole sphere (equatorial band, many islands),
partition **points** and sum stretch leaves. Synthesis is a disjoint union; the adjoint
sums. Auto will not pick one-cap stretch for a large bounding hull (`fsky ≳ 0.3`):
that operator is inaccurate when \(\theta_*\sim\pi\). Cluster vs NUFSHT is decided by
whether the clusters actually cover the sphere or only a thin subset. A wrapping
band that needs many leaves still loses to a full-sphere NUFSHT; a few compact
islands keep cluster. At the small
\(L\) used in tests, window / \(L_{\max}\) truncation (a few \(10^{-3}\) relative)
dominates NUFFT `tol`; that is not an \(O(1)\) phase error.

## NUFSHT leaf

When \(f_{\mathrm{sky}}\) is large, `NUFSHTAlgorithm` optionally calls NUFSHT's public
plans. Scalar uses `make_plan` / `nusht_type2!` / the documented Euclidean adjoint,
converting complex Condon–Shortley coefficients to FastTransforms packing. Spin
uses `make_spin_plan` / `nusht_type2_spin!` / `nusht_type1_spin!` on the same dense
`spin_coeff_index` layout as Direct.

## What this is not

- **Slepian functions** are a diagnostic (`shannon_number`, `solve!` subspace), not the
  stored representation.
- **Giner Mascarell & Bunn (arXiv:2507.15703)** is a thin-box 3D FFT *simulator* matching
  \(\xi(\theta)\). It is not \(\mathcal{Y}_R\) and is not a leaf.
- **Butterfly / ID** of associated-Legendre-on-an-interval is gated on experiment C
  (see [ranks](ranks.md)).
