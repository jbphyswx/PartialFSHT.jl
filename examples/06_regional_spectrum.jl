#=
Band-powers from a region, with the mask's own coupling accounted for.

"Which wavenumbers carry what energy over this basin" is the well-posed question, and the reason is
pure counting: coefficients are `(L+1)²` unknowns against `~N_S` constraints (impossible), while a
spectrum binned at the geometry's own `Δℓ` is `~L/Δℓ` unknowns against the same constraints
(massively overdetermined).

The bias is *computable* rather than merely bounded, because the mask is chosen: `⟨C̃_ℓ⟩ =
Σ_ℓ' M_{ℓℓ'} C_ℓ'` with `M` built from the mask's own power spectrum and Wigner 3-j symbols
(Hivon et al. 2002). Nothing about the unobserved sky enters.

The cost is variance, not bias: `ΔC_ℓ/C_ℓ ≈ √(2/((2ℓ+1) f_sky Δℓ))`, reported with every estimate.
=#

using PartialFSHT: PartialFSHT
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads             # lets the O(L³) coupling build thread
using ComputationalBackends: ComputationalBackends
using Printf: Printf
using Random: Random
const PF = PartialFSHT
const CB = ComputationalBackends

Random.seed!(5)

lmax = 128
n = 20_000
θ = 1.0 .+ 0.6 .* (rand(n) .- 0.5)
φ = 2π .* rand(n)

plan = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8, backend = CB.ThreadedBackend())
ws = PF.AnalysisWorkspace(plan)
box = PF.bounding_box(θ, φ)
Printf.@printf("region: fsky = %.4f, Δℓ ≈ %.2f\n", PF.box_fsky(box), PF.spectral_resolution(box))

# A field with a known power spectrum: C_ℓ ∝ (ℓ+1)^-2.
Cl_true = [1.0 / (1 + ℓ)^2 for ℓ in 0:lmax]
alm = PF.zeros_alm(lmax)
for ℓ in 0:lmax, m in (-ℓ):ℓ
    alm[PF.sph_mode(ℓ, m)] = sqrt(Cl_true[ℓ + 1]) * randn(ComplexF64)
end
f = plan * alm
w = ones(n)

# The mask's own power spectrum. For a real analysis this comes from transforming the mask itself.
# A smooth region of angular extent Λ has its power concentrated below ℓ ≈ π/Λ, so an exponential
# with that scale is the right stand-in. A *flat* W_ℓ would not be: the 3-j sum rule then makes
# M_{ℓℓ'} = (2ℓ'+1)W/4π exactly, which is rank one and cannot be deconvolved at all.
Λ = max(PF.box_theta_extent(box), box.φextent * sin((box.θlo + box.θhi) / 2))
ℓmask = π / Λ
Wℓ = [exp(-ℓ / ℓmask) * PF.box_fsky(box) / (4π) for ℓ in 0:(2lmax)]
Printf.@printf("mask power concentrated below ℓ ≈ %.1f (region extent Λ = %.2f rad)\n", ℓmask, Λ)

# `M_{ℓℓ'}` is O(L³) and a function of the mask alone, so build it once and reuse it.
t = @elapsed coupling = PF.mask_coupling(Wℓ, lmax, CB.ThreadedBackend())
Printf.@printf("coupling matrix %dx%d built in %.3f s (threaded)\n\n", size(coupling)..., t)

est = PF.spectrum(plan, f, w, Wℓ; coupling = coupling, ws = ws)
Printf.@printf("%14s %12s %12s %12s\n", "band", "bandpower", "±", "truth (mid)")
for (i, b) in enumerate(est.bands)
    i > 8 && break
    ℓc = (first(b) + last(b)) ÷ 2
    Printf.@printf("%14s %12.4e %12.1e %12.4e\n", "$(first(b))–$(last(b))",
            est.bandpowers[i], est.sigma[i], Cl_true[ℓc + 1])
end
Printf.@printf("\nΔℓ used: %.2f over %d bands; deconvolved: %s\n",
        est.Δℓ, length(est.bands), est.deconvolved)

# Forward-modelling is the preferred direction: push the model spectrum through the coupling and
# compare in pseudo space, rather than inverting the coupling.
pseudo = PF.spectrum(plan, f, w, Wℓ; coupling = coupling, deconvolve = false, ws = ws)
Printf.@printf("\npseudo-space comparison (no matrix inverted): first band %.4e ± %.1e\n",
        pseudo.bandpowers[1], pseudo.sigma[1])

# `ntrans > 1` carries several independent maps, and there is no single spectrum of a batch — the
# member is an argument rather than a silent choice.
Printf.@printf("\nbatch member is explicit: pseudo_spectrum!(..., batch = 1)\n")
println("\nok")
