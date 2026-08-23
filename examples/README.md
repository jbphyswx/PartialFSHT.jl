# Examples

Runnable end to end. Each prints what it computed and asserts the property it is demonstrating, so
a broken example fails rather than printing plausible numbers.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/01_synthesis_and_adjoint.jl
```

| file | what it shows |
| --- | --- |
| [`01_synthesis_and_adjoint.jl`](01_synthesis_and_adjoint.jl) | the two operators, the adjoint identity, and that both are exact on a patch |
| [`02_accuracy_and_tolerance.jl`](02_accuracy_and_tolerance.jl) | delivered accuracy against a direct evaluation, and what `tol` buys |
| [`03_choosing_a_method.jl`](03_choosing_a_method.jl) | what `AutoAlgorithm` picks for a cap, a band, two basins, and a full sky — and why |
| [`04_threading_and_batching.jl`](04_threading_and_batching.jl) | `ThreadedBackend`, `ntrans` batching, and the zero-allocation apply |
| [`05_comparing_against_a_model.jl`](05_comparing_against_a_model.jl) | the recommended workflow: window both sides identically, compare in `(ℓ,m)` |
| [`06_regional_spectrum.jl`](06_regional_spectrum.jl) | band-powers with the mask coupling accounted for, and their error bars |
| [`07_what_the_adjoint_is_not.jl`](07_what_the_adjoint_is_not.jl) | why `plan' * f` is **not** "the coefficients of my data", shown numerically |
| [`08_spin_weighted.jl`](08_spin_weighted.jl) | spin-`s` transforms against `sYlm` |

Run all of them:

```bash
julia --project=examples examples/run_all.jl
```
