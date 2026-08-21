# Rank laboratory

Phase 0 measures where compression actually lives. Production `Plan` apply never forms
these dense blocks. From `experiments/rank/run.jl` (CPU, small \(L\)):

## Experiment A — global SVD vs Shannon

Dense \(\mathcal{Y}_R\) vs \(W^{1/2}\mathcal{Y}_R\) vs the Shannon number
\(N_S=(L+1)^2(1-\cos\theta_*)/2\). Unweighted SVD of \(A\) is **not** the concentration
operator. Factoring \(A\approx UV^*\) with \(r\sim N_S\) does not beat a naive matvec.

At \(L=16\), \(\theta_*=20^\circ\), \(N=200\) random cap samples: \(N_S\approx 9.3\) while
\(\mathrm{rank}(A,10^{-3})=\mathrm{rank}(W^{1/2}A,10^{-3})=43\) (the point-sampled matrix
is not the concentration operator).

## Experiment B — per-degree rank

At fixed \(\ell\), numerical rank of \(A^{(\ell)}_{i,m}=Y_{\ell m}(\Omega_i)\) tracks
\(O(\ell\,\theta_*)\) (polar optimization), not \(2\ell+1\).

Same cap as A: ranks
`[1, 3, 5, 7, 9, 11, 11, 13, 13, 15, 15, 17, 17, 19, 19, 21, 21]`
for \(\ell=0{:}16\). At \(\ell=16\), \(21\) vs \(2\ell+1=33\).

## Experiment C — associated-Legendre complementary ranks (butterfly gate)

For fixed \(m\), the block \(P_{i,\ell}=\bar P_\ell^m(\cos\theta_i)\) with
\(\theta_i\in[0,\theta_*]\). Dyadic \(\theta\)–\(\ell\) blocks have complementary
low rank (FIO / butterfly). At \(L=32\), \(m=2\), \(4\times 4\) boxes, ranks are
\(2\)–\(4\) (not the full box size). Stretch already spends that rank with an
FFT. A Tygert/Wavemoth butterfly for associated Legendres is not implemented:
it would duplicate the ring transform, not the Wigner rotation.

Run:

```julia
include("experiments/rank/run.jl")
```

## Experiment D — irregular masks

A thin equatorial band has small Shannon number but a bounding polar cap near
\(\theta_*=\pi\). Clustering into many small caps is the hierarchical partition;
HODLR of \(Y_{\ell m}\) is the wrong matrix class.

| mask | bounding \(\theta_*\) | \(k\) | \(\sum_c\theta_c^2\) |
|------|----------------------:|------:|---------------------:|
| cap | 0.307 | 1 | 0.094 |
| equatorial band | 3.132 | 7 | 1.339 |
| two antipodal caps | 1.867 | 2 | 0.086 |
| random 10% lat-lon | 3.089 | 8 | 4.963 |
| ocean-like | 2.664 | 8 | 3.38 |
