#=
The recommended workflow: compare partial observations against a global model in coefficient space.

This is the answer to "I have basin observations and a global forecast, and I want to compare
wavenumber content" — and it needs no inversion at all.

Window both sides identically:

    ã_obs   = A† W f_obs             one adjoint transform of the windowed observations
    ã_model = A† W A a_model         synthesise at the obs points, window, transform back

Both sides are then the *same* operator applied to their own field, so the comparison in `(ℓ,m)` is
exact and unbiased. Nothing is deconvolved, nothing is forced to zero, and no Slepian functions
appear. From the caller's side it costs one extra forward transform.

What not to do: invert the observations to `a_ℓm` and compare those against the model's. That
comparison is contaminated by the projection, and it is avoidable.
=#

using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using FINUFFT: FINUFFT
using Printf: Printf
using Random: Random
const PF = PartialFSHT

Random.seed!(4)

lmax = 256
# A mid-latitude basin-shaped region rather than a polar cap, to show the frame is not required.
n = 40_000
θ = 1.1 .+ 0.45 .* (rand(n) .- 0.5)
φ = 2.0 .+ 1.6 .* (rand(n) .- 0.5)

plan = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8)
ws = PF.AnalysisWorkspace(plan)
r = PF.recoverability(θ, φ, lmax)
Printf.@printf("region: fsky = %.4f, N_S = %.0f of %d coefficients, Δℓ ≈ %.1f\n",
        r.fsky, r.shannon, r.ncoeff, r.Δℓ)
Printf.@printf("→ %.2f%% of coefficient directions are constrained; band-powers at Δℓ ≈ %.0f are not.\n\n",
        100 * r.recoverable_fraction, r.Δℓ)

# The "truth" and a "model" that differs from it at a known scale.
a_true = PF.zeros_alm(lmax)
a_model = PF.zeros_alm(lmax)
for ℓ in 0:lmax, m in (-ℓ):ℓ
    v = randn(ComplexF64) / (1 + ℓ)
    a_true[PF.sph_mode(ℓ, m)] = v
    # the model agrees below ℓ = 100 and is wrong above it
    a_model[PF.sph_mode(ℓ, m)] = ℓ <= 100 ? v : v + 0.5 * randn(ComplexF64) / (1 + ℓ)
end

# Observations of the truth at those points, with a uniform weight for simplicity.
f_obs = plan * a_true
w = ones(length(θ))

# Observation side: one adjoint transform of the windowed data.
ã_obs = PF.zeros_alm(lmax)
fw = copy(f_obs)
fw .*= w
LinearAlgebra.mul!(ã_obs, LinearAlgebra.adjoint(plan), fw)

# Model side: push the model through the identical operator. This is `apply_resolution!`.
ã_model = PF.zeros_alm(lmax)
PF.apply_resolution!(ã_model, plan, a_model, w, ws)

# And, to show the property a third party relies on: the obs side equals the resolution operator
# applied to the truth, to round-off.
ã_truth_pushed = PF.zeros_alm(lmax)
PF.apply_resolution!(ã_truth_pushed, plan, a_true, w, ws)
rel = maximum(abs.(ã_obs .- ã_truth_pushed)) / maximum(abs, ã_truth_pushed)
Printf.@printf("A†W(A a_true) == apply_resolution!(a_true):  rel = %.3e\n", rel)
@assert rel < 1e-10 "the two sides of the comparison are not the same operator"

# Now the comparison itself, per degree, in Y_ℓm labels throughout.
println("\nwindowed coefficient mismatch by degree band:")
Printf.@printf("%12s %14s %14s\n", "band", "‖Δã‖ / ‖ã‖", "verdict")
for band in (0:50, 51:100, 101:180, 181:256)
    num = 0.0
    den = 0.0
    for ℓ in band, m in (-ℓ):ℓ
        num += abs2(ã_obs[PF.sph_mode(ℓ, m)] - ã_model[PF.sph_mode(ℓ, m)])
        den += abs2(ã_obs[PF.sph_mode(ℓ, m)])
    end
    rd = sqrt(num / den)
    Printf.@printf("%12s %14.4f %14s\n", "$(first(band))–$(last(band))", rd,
            rd < 0.05 ? "agrees" : "model differs")
end
println("\n(the model was built to agree below ℓ = 100 — the comparison recovers exactly that)")
println("\nok")
