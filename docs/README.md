# Documentation

Run these commands from the repository root to build the manual:

```sh
julia --project=docs -e 'using Pkg: Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The documentation workflow builds and deploys this manual using the committed PNGs in
`docs/src/assets/`. It does not regenerate figures or load CairoMakie.

## Regenerate figures

Figure generation has its own environment in `docs/generate_assets/`, outside the root workspace.
Run it explicitly when the numerical results or plots need updating:

```sh
julia --project=docs/generate_assets -e 'using Pkg: Pkg; Pkg.instantiate()'
julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
```

The generator writes all six figures into `docs/src/assets/`. Review and commit the resulting PNGs
with the relevant code or documentation changes before publishing the manual.
