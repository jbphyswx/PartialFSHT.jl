#=
`plan' * f` is not "the coefficients of my data", and this shows it numerically.

The adjoint is exactly computable on a patch — it is a finite sum — and it is the exact transpose of
synthesis, which is what an iterative solver, a likelihood gradient or a quadratic estimator needs.
It is *not* the underlying field's harmonic coefficients. It is the **masked** field's:
mode-coupled and leakage-biased.

This is the one place the package could silently hand back something that looks like an answer, so
it is worth seeing the size of the error, and seeing that `solve!` names its regularisation instead
of pretending there is a regularisation-free inverse.
=#

using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using FINUFFT: FINUFFT
using Printf: Printf
using Random: Random
const PF = PartialFSHT

Random.seed!(6)

"""Squared ℓ² norm over the slots that are real modes, not the array's padding."""
function mode_norm2(a, lmax)
    s = 0.0
    for ℓ in 0:lmax, m in (-ℓ):ℓ
        s += abs2(a[PF.sph_mode(ℓ, m)])
    end
    return s
end

function mode_diff2(a, b, lmax, scale = 1)
    s = 0.0
    for ℓ in 0:lmax, m in (-ℓ):ℓ
        i = PF.sph_mode(ℓ, m)
        s += abs2(a[i] / scale - b[i])
    end
    return s
end

function main()
    lmax = 64
    n = 8_000
    θstar = 30 * π / 180
    θ = θstar .* sqrt.(rand(n))
    φ = 2π .* rand(n)

    plan = PF.Plan(θ, φ; lmax = lmax, tol = 1e-10)
    r = PF.recoverability(θ, φ, lmax)
    Printf.@printf("fsky = %.4f → N_S = %.0f of %d coefficients (%.1f%%)\n\n",
            r.fsky, r.shannon, r.ncoeff, 100 * r.recoverable_fraction)

    a_true = PF.zeros_alm(lmax)
    for ℓ in 0:lmax, m in (-ℓ):ℓ
        a_true[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
    end
    f = plan * a_true
    den = mode_norm2(a_true, lmax)

    # 1. The naive reading of the adjoint. Scale out the point measure first, so the comparison is
    #    about shape rather than normalisation — this is the *most* favourable reading of A†f.
    a_adj = PF.zeros_alm(lmax)
    LinearAlgebra.mul!(a_adj, LinearAlgebra.adjoint(plan), f)
    scale = LinearAlgebra.dot(a_adj, a_true) / LinearAlgebra.dot(a_true, a_true)
    rel_adj = sqrt(mode_diff2(a_adj, a_true, lmax, scale) / den)
    Printf.@printf("A†f (best-scaled) vs the truth:  relative error %.3f\n", rel_adj)
    Printf.@printf("  → this is NOT the coefficients: off by %.0f%%, and no tolerance fixes it.\n\n",
            100 * rel_adj)
    @assert rel_adj > 0.1 "the leakage this example exists to show did not appear"

    # 2. Synthesis, by contrast, is exact — the same plan, the other direction.
    pd = PF.Plan(θ[1:200], φ[1:200]; lmax = lmax, method = PF.DirectAlgorithm())
    f_ref = pd * a_true
    rel_syn = maximum(abs.(f[1:200] .- f_ref)) / maximum(abs, f_ref)
    Printf.@printf("synthesis vs the direct oracle:  relative error %.3e  (exact at any coverage)\n\n",
            rel_syn)
    @assert rel_syn < 1e-8

    # 3. `solve!` requires the regularisation to be named. No default hides the choice.
    for mode in (:cg, :minnorm)
        alm = PF.zeros_alm(lmax)
        _, info = PF.solve!(alm, plan, f; mode = mode, rtol = 1e-8, maxiter = 300)
        Printf.@printf("solve!(mode = :%-8s) iters %3d  relres %.2e  → coefficient error %.3f\n",
                mode, info.iters, info.relres, sqrt(mode_diff2(alm, a_true, lmax) / den))
        Printf.@printf("%22s N_S = %.0f, Δℓ = %.2f reported alongside\n", "", info.shannon, info.Δℓ)
    end

    # :wiener refuses to run without a prior, because the prior is what makes it well posed.
    try
        PF.solve!(PF.zeros_alm(lmax), plan, f; mode = :wiener)
        error("expected :wiener to demand a prior")
    catch err
        err isa ArgumentError || rethrow()
        Printf.@printf("\nsolve!(mode = :wiener) without a prior: %s\n",
                first(split(sprint(showerror, err), '.')))
    end
end

main()
println("\nok")
