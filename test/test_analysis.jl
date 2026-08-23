Test.@testset "recoverability reports what the geometry supports" begin
    Random.seed!(51)
    # A 4° patch: the local spectrum is recoverable, the individual coefficients are not, and
    # the recoverable *fraction* is f_sky independently of ℓ.
    half = 2 * π / 180
    θ = (0.9) .+ half .* (2 .* rand(2000) .- 1)
    φ = 1.0 .+ (half / sin(0.9)) .* (2 .* rand(2000) .- 1)
    for L in (512, 1024, 2048)
        r = PartialFSHT.recoverability(θ, φ, L)
        Test.@test r.recoverable_fraction ≈ r.fsky
        Test.@test r.shannon ≈ (L + 1)^2 * r.fsky
        Test.@test r.Δℓ > 1
    end
    # the fraction does not improve with degree
    f1 = PartialFSHT.recoverability(θ, φ, 512).recoverable_fraction
    f2 = PartialFSHT.recoverability(θ, φ, 4096).recoverable_fraction
    Test.@test f1 ≈ f2

    # a hemisphere is a different regime: half the directions, near per-ℓ resolution
    θh = acos.(rand(4000))
    φh = 2π .* rand(4000)
    rh = PartialFSHT.recoverability(θh, φh, 256)
    Test.@test 0.4 < rh.fsky < 0.6
    Test.@test rh.Δℓ < 3
end

Test.@testset "mode coupling" begin
    Test.@testset "3-j closed form against direct evaluation" begin
        # the all-zero-order symbol vanishes for odd J and outside the triangle
        Test.@test PartialFSHT.threej000_sq_log(1, 1, 1) == -Inf
        Test.@test PartialFSHT.threej000_sq_log(1, 2, 5) == -Inf
        # (0 0 0; 000) = 1
        Test.@test exp(PartialFSHT.threej000_sq_log(0, 0, 0)) ≈ 1
        # (1 1 0; 000)² = 1/3
        Test.@test exp(PartialFSHT.threej000_sq_log(1, 1, 0)) ≈ 1 / 3
        # (2 2 0; 000)² = 1/5
        Test.@test exp(PartialFSHT.threej000_sq_log(2, 2, 0)) ≈ 1 / 5
        # (1 1 2; 000)² = 2/15
        Test.@test exp(PartialFSHT.threej000_sq_log(1, 1, 2)) ≈ 2 / 15
    end

    Test.@testset "a full-sky unit mask gives the identity" begin
        # W_ℓ for w ≡ 1 is 4π at ℓ=0 and zero elsewhere, and then M_{ℓℓ'} = δ_{ℓℓ'}
        L = 12
        W = zeros(2L + 1)
        W[1] = 4π
        M = PartialFSHT.mask_coupling(W, L)
        Test.@test M ≈ LinearAlgebra.I(L + 1) atol = 1e-10
    end

    Test.@testset "a dense mask spectrum, against the untabulated closed form" begin
        # The tabulated `loggamma` must cover the largest triple the sum reaches: ℓ, ℓ' ≤ L and
        # ℓ'' ≤ min(ℓ+ℓ', nW), so J can reach 2L + min(2L, nW). A mask spectrum with only its first
        # few entries nonzero never exercises that, because the zeros are skipped — so this uses a
        # dense W_ℓ over the whole 2L+1 range, which is what a real apodised mask looks like.
        for L in (8, 16, 24)
            W = [1.0 / (1 + ℓ)^2 for ℓ in 0:(2L)]
            M = PartialFSHT.mask_coupling(W, L)
            Test.@test all(isfinite, M)
            Test.@test all(M .>= -1e-14)
            # element-by-element against the reference that calls `loggamma` directly
            ref = zeros(L + 1, L + 1)
            for ℓ in 0:L, ℓp in 0:L
                acc = 0.0
                for ℓpp in abs(ℓ - ℓp):min(ℓ + ℓp, 2L)
                    lg = PartialFSHT.threej000_sq_log(ℓ, ℓp, ℓpp)
                    isfinite(lg) || continue
                    acc += (2ℓpp + 1) * W[ℓpp + 1] * exp(lg)
                end
                ref[ℓ + 1, ℓp + 1] = (2ℓp + 1) * acc / (4π)
            end
            Test.@test M ≈ ref rtol = 1e-10
        end
    end

    Test.@testset "a flat mask spectrum exercises the 3-j sum rule" begin
        # With W_ℓ ≡ w the inner sum is Σ_{ℓ''} (2ℓ''+1) (ℓ ℓ' ℓ''; 000)² = 1 exactly, so
        # M_{ℓℓ'} = (2ℓ'+1) w / 4π — every row proportional to the same vector, hence rank one and
        # singular *by construction*, not by numerical accident. That makes the flat case a sharp
        # test of the whole kernel: it must reproduce the sum rule to round-off, and it reaches the
        # largest ℓ'' the sum admits, which is where an undersized `loggamma` table shows up.
        for L in (8, 24)
            w = 0.25 / (4π)
            W = fill(w, 2L + 1)
            M = PartialFSHT.mask_coupling(W, L)
            Test.@test all(isfinite, M)
            for ℓ in 0:L, ℓp in 0:L
                # the sum rule truncates at ℓ'' ≤ ℓ+ℓ', which W_ℓ covers here
                Test.@test M[ℓ + 1, ℓp + 1] ≈ (2ℓp + 1) * w / (4π) rtol = 1e-10
            end
            Test.@test LinearAlgebra.rank(M; rtol = 1e-8) == 1
        end
    end

    Test.@testset "a decaying mask spectrum gives an invertible per-degree system" begin
        # `spectrum` at Δℓ = 1 inverts the unbinned coupling directly. A real mask's power is
        # concentrated at low ℓ, and then the coupling is a banded smoother rather than rank one.
        L = 24
        W = [exp(-ℓ / 3.0) / (4π) for ℓ in 0:(2L)]
        M = PartialFSHT.mask_coupling(W, L)
        Test.@test all(isfinite, M)
        Mb = PartialFSHT.bin_coupling(M, PartialFSHT.band_edges(L, 1))
        Test.@test all(isfinite, Mb)
        Test.@test abs(LinearAlgebra.det(Mb)) > 0
    end

    Test.@testset "coupling conserves power and binning is invertible" begin
        L = 16
        W = zeros(2L + 1)
        W[1] = 4π * 0.3
        W[2] = 0.4
        W[3] = 0.1
        M = PartialFSHT.mask_coupling(W, L)
        Test.@test all(M .>= -1e-14)
        bands = PartialFSHT.band_edges(L, 4)
        Mb = PartialFSHT.bin_coupling(M, bands)
        Test.@test size(Mb) == (length(bands), length(bands))
        Test.@test abs(LinearAlgebra.det(Mb)) > 0
    end
end

Test.@testset "inversion: modes are named, and the null space is real" begin
    Random.seed!(52)
    L, n = 8, 400
    θ = 0.6 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    ws = PartialFSHT.AnalysisWorkspace(plan)
    alm_true = rand_alm(L)
    f = plan * alm_true

    rec = PartialFSHT.zeros_alm(L)
    rec, info = PartialFSHT.solve!(rec, plan, f; mode = :cg, rtol = 1e-10,
                                   maxiter = 2000, ws = ws)
    # the solve reports what it did rather than returning bare numbers
    Test.@test info.mode === :cg
    Test.@test info.iters >= 1
    Test.@test haskey(info, :shannon) && haskey(info, :Δℓ) && haskey(info, :relres)
    # On a 0.6 rad cap at lmax = 8, N_S ≈ 7 of 81 coefficient directions are constrained, so the
    # normal equations are severely ill-conditioned and the stopping rule *is* the
    # regularisation. Assert that, rather than a fit accuracy the geometry does not support:
    # more iterations must keep reducing the residual, and the reported relres must be honest.
    Test.@test info.shannon < 0.2 * (L + 1)^2
    r_short, i_short = PartialFSHT.solve!(PartialFSHT.zeros_alm(L), plan, f;
                                          mode = :cg, rtol = 1e-10, maxiter = 50, ws = ws)
    res_short = LinearAlgebra.norm((plan * r_short) .- f) / LinearAlgebra.norm(f)
    res_long = LinearAlgebra.norm((plan * rec) .- f) / LinearAlgebra.norm(f)
    Test.@test i_short.iters == 50 && i_short.relres > info.relres
    Test.@test res_long < res_short
    Test.@test res_long < 1e-4

    Test.@testset "a well-covered geometry does converge" begin
        # Same solver, near-global coverage: now the system is well conditioned and the fit is
        # tight, which is what makes the ill-conditioned case above a property of the geometry.
        Random.seed!(56)
        ng = 3000
        θg = acos.(2 .* rand(ng) .- 1)
        φg = 2π .* rand(ng)
        pg = PartialFSHT.Plan(θg, φg; lmax = 6, method = PartialFSHT.DirectAlgorithm())
        wsg = PartialFSHT.AnalysisWorkspace(pg)
        at = rand_alm(6)
        fg = pg * at
        rg, ig = PartialFSHT.solve!(PartialFSHT.zeros_alm(6), pg, fg;
                                    mode = :cg, rtol = 1e-12, maxiter = 500, ws = wsg)
        Test.@test ig.shannon > 0.8 * 49
        Test.@test LinearAlgebra.norm((pg * rg) .- fg) / LinearAlgebra.norm(fg) < 1e-9
        Test.@test alm_reldiff(rg, at, 6) < 1e-8
    end

    Test.@testset ":wiener requires its prior" begin
        Test.@test_throws ArgumentError PartialFSHT.solve!(rec, plan, f; mode = :wiener)
        prior = [1.0 / (1 + ℓ)^2 for ℓ in 0:L]
        r2, i2 = PartialFSHT.solve!(PartialFSHT.zeros_alm(L), plan, f;
                                    mode = :wiener, prior = prior, rtol = 1e-8, ws = ws)
        Test.@test i2.mode === :wiener
        Test.@test i2.iters >= 1
    end

    Test.@test_throws ArgumentError PartialFSHT.solve!(rec, plan, f; mode = :nonsense)
end

Test.@testset "A†f is not the field's coefficients" begin
    # The trap: the adjoint is exactly computable on a patch but returns the *masked* field's
    # coefficients, mode-coupled and leakage-biased. Nobody may read it as an analysis.
    Random.seed!(53)
    L, n = 10, 600
    θ = 0.5 .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    alm_true = rand_alm(L)
    f = plan * alm_true
    adj = plan' * f
    # not proportional to the truth, and not close to it after any single rescaling
    s = LinearAlgebra.dot(adj, alm_true) / LinearAlgebra.dot(alm_true, alm_true)
    Test.@test LinearAlgebra.norm(adj .- s .* alm_true) / LinearAlgebra.norm(adj) > 0.1
end

Test.@testset "resolution operator is what a comparison must use" begin
    Random.seed!(54)
    L, n = 8, 300
    θ = 0.5 .* sqrt.(rand(n)); φ = 2π .* rand(n)
    plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    ws = PartialFSHT.AnalysisWorkspace(plan)
    w = fill(1.0, n)
    alm = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
    out = similar(alm)
    PartialFSHT.apply_resolution!(out, plan, alm, w, ws)
    # A†WA applied to a model equals the adjoint of that model's own synthesis: both sides of a
    # comparison carry the identical coupling, so nothing needs deconvolving
    manual = plan' * (plan * alm)
    Test.@test maximum(abs.(out .- manual)) / maximum(abs.(manual)) < 1e-12
    # and it is self-adjoint
    a2 = randn(ComplexF64, PartialFSHT.coeff_size(L)...)
    o2 = similar(a2)
    PartialFSHT.apply_resolution!(o2, plan, a2, w, ws)
    Test.@test abs(LinearAlgebra.dot(out, a2) - LinearAlgebra.dot(alm, o2)) <
               1e-10 * abs(LinearAlgebra.dot(out, a2))
end

Test.@testset "synthesis_matrix agrees with the operator" begin
    Random.seed!(55)
    L, n = 6, 20
    θ = 0.5 .* rand(n); φ = 2π .* rand(n)
    A = PartialFSHT.synthesis_matrix(θ, φ, L)
    Test.@test size(A) == (n, (L + 1)^2)
    plan = PartialFSHT.Plan(θ, φ; lmax = L, method = PartialFSHT.DirectAlgorithm())
    alm = PartialFSHT.zeros_alm(L)
    x = zeros(ComplexF64, (L + 1)^2)
    k = 1
    for ℓ in 0:L, m in -ℓ:ℓ
        v = randn(ComplexF64)
        alm[PartialFSHT.sph_mode(ℓ, m)] = v
        x[k] = v
        k += 1
    end
    Test.@test maximum(abs.((A * x) .- (plan * alm))) < 1e-12
end
