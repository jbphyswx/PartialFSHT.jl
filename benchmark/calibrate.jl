"""
Calibration and the performance contract.

Measures, on one machine, the things the planner's cost model and the README's speedup table
are allowed to claim:

1. the resolved configuration — printed first, because a timing whose backend or thread count
   is unknown is not a measurement;
2. stretched-leaf apply vs a full-sphere `NUFSHT.nusht_type2!` on the same points;
3. a per-stage breakdown of the stretched leaf, so a regression can be attributed;
4. per-unit costs for [`PartialFSHT.set_cost_constants!`](@ref).

Run with the test environment, which carries NUFSHT and a NUFFT backend:

    julia --project=test benchmark/calibrate.jl
"""

using PartialFSHT: PartialFSHT
using NUFSHT: NUFSHT
using NonuniformFFTs: NonuniformFFTs
using OhMyThreads: OhMyThreads
using ComputationalBackends: ComputationalBackends
using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: @printf, @sprintf

const PF = PartialFSHT

const CAP_DEG = 20.0
const TOL = 1e-8
const NREP = 3

function config_line()
    nufft = PF._resolve_nufft(PF.AutoNUFFT())
    la = try
        strip(read(`uptime`, String))
    catch
        "unknown"
    end
    @printf("config | julia %s | threads %d | BLAS %d | FFTW %d | NUFFT %s | tol %.0e\n",
            string(VERSION), Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(),
            PF.FFTW.get_num_threads(), nameof(typeof(nufft)), TOL)
    println("       | ", la)
    println("       | a loaded machine inflates every number below; treat load average > ncpu/2 as noise")
    println()
    return nufft
end

cap_points(n, θstar) = (θstar .* sqrt.(rand(n)), 2π .* rand(n))

function best(f, n = NREP)
    f()
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

"""Per-stage timing of one stretched-leaf synthesis, so a regression is attributable."""
function stage_times(leaf, alm3)
    ser = ComputationalBackends.SerialBackend()
    thr = ComputationalBackends.ThreadedBackend()
    tr = best(() -> PF._fill_rings!(leaf.G, alm3, leaf.eng, ser))
    trt = Threads.nthreads() > 1 ?
        best(() -> PF._fill_rings!(leaf.G, alm3, leaf.eng, thr)) : tr
    tg = best(() -> PF._rings_to_grid!(leaf.F, leaf.G, leaf.lmax, leaf.ring_bfft, leaf.Fφ))
    tp = best(() -> PF._periodize!(leaf.F̃, leaf.F, leaf.geomθ, leaf.spin))
    tf = best(() -> LinearAlgebra.mul!(leaf.Fhat, leaf.fft_plan, leaf.F̃))
    tn = best(() -> PF._nufft_exec!(leaf.nufft2, leaf.Fhat, leaf.fbuf))
    return (ring = tr, ring_thr = trt, phi = tg, dfs = tp, fft = tf, nufft = tn)
end

function run_L(L, nufft)
    Random.seed!(0)
    θstar = CAP_DEG * π / 180
    M = min(4 * round(Int, (L * θstar)^2), 2_000_000)
    θ, φ = cap_points(M, θstar)
    alm = randn(ComplexF64, PF.coeff_size(L)...)
    alm3 = reshape(alm, size(alm)..., 1)

    ts_setup = @elapsed ps = PF.Plan(θ, φ; lmax = L, method = PF.StretchAlgorithm(),
                                     nufft = nufft, tol = TOL,
                                     backend = ComputationalBackends.SerialBackend())
    f = zeros(ComplexF64, M)
    ts = best(() -> LinearAlgebra.mul!(f, ps, alm))
    # The adjoint writes coefficients, so it needs its own destination: timing it into `alm`
    # would corrupt the input before the threaded comparison below reads it.
    almadj = similar(alm)
    ta = best(() -> LinearAlgebra.mul!(almadj, ps', f))
    tt = ts
    if Threads.nthreads() > 1
        pt = PF.Plan(θ, φ; lmax = L, method = PF.StretchAlgorithm(), nufft = nufft,
                     tol = TOL, backend = ComputationalBackends.ThreadedBackend())
        ft = zeros(ComplexF64, M)
        tt = best(() -> LinearAlgebra.mul!(ft, pt, alm))
        maximum(abs.(ft .- f)) == 0 ||
            @printf("    WARNING: threaded result differs from serial at L=%d\n", L)
    end
    leaf = ps.leaf
    st = stage_times(leaf, alm3)

    tn_setup = NaN
    tn = NaN
    rel = NaN
    try
        tn_setup = @elapsed np = NUFSHT.make_plan(ComplexF64, θ, φ, L; tol = TOL)
        C = similar(np.C)
        fill!(C, 0)
        fn = zeros(ComplexF64, M)
        tn = best(() -> NUFSHT.nusht_type2!(fn, C, np))
        NUFSHT.close!(np)
    catch err
        @printf("    NUFSHT baseline unavailable at L=%d: %s\n", L, sprint(showerror, err))
    end

    @printf("L=%5d M=%8d | leaf: K=%4d nth=%5d nphi=%5d ring_work=%.3e\n",
            L, M, leaf.geomθ.K, leaf.nθ, leaf.nφ, Float64(PF.ring_work(ps)))
    @printf("            | stretch  setup %7.3f s   synth %8.4f s   adj %8.4f s   threaded %8.4f s (%.2fx)\n",
            ts_setup, ts, ta, tt, ts / tt)
    @printf("            |   stages ring %7.4f (thr %7.4f, %.2fx)  phi %7.4f  dfs %7.4f  fft %7.4f  nufft %7.4f\n",
            st.ring, st.ring_thr, st.ring / st.ring_thr, st.phi, st.dfs, st.fft, st.nufft)
    if isfinite(tn)
        @printf("            | NUFSHT   setup %7.3f s   synth %8.4f s   -> speedup %6.2fx\n",
                tn_setup, tn, tn / ts)
    end
    println()
    return (L = L, M = M, ts = ts, ta = ta, tt = tt, tn = tn,
            ring_work = PF.ring_work(ps), stages = st)
end

function calibrate_constants(rows)
    # ring cost per retained (ℓ,m,ring) triple, from the ring stage alone
    rs = [(r.stages.ring, r.ring_work) for r in rows if r.ring_work > 0]
    isempty(rs) && return nothing
    ring = sum(t for (t, w) in rs) / sum(Float64(w) for (t, w) in rs)
    println("="^92)
    @printf("measured ring cost: %.3e s per retained (l,m,ring) triple\n", ring)
    println("install with: PartialFSHT.set_cost_constants!(ring = <s/triple>, …)")
    println("="^92)
    return ring
end

function main()
    println("="^92)
    println("PartialFSHT calibration — $(CAP_DEG)° polar cap, M ≈ 4(L·θ⋆)²")
    println("="^92)
    nufft = config_line()
    rows = Any[]
    for L in (256, 512, 1024, 2048)
        try
            push!(rows, run_L(L, nufft))
        catch err
            @printf("L=%d failed: %s\n\n", L, sprint(showerror, err))
        end
    end
    calibrate_constants(rows)
    println()
    println("summary (synthesis):")
    @printf("%6s %10s %12s %12s %12s %10s %10s\n",
            "L", "M", "serial s", "threaded s", "NUFSHT s", "vs serial", "vs threaded")
    for r in rows
        @printf("%6d %10d %12.4f %12.4f %12s %10s %10s\n", r.L, r.M, r.ts, r.tt,
                isfinite(r.tn) ? @sprintf("%.4f", r.tn) : "-",
                isfinite(r.tn) ? @sprintf("%.2fx", r.tn / r.ts) : "-",
                isfinite(r.tn) ? @sprintf("%.2fx", r.tn / r.tt) : "-")
    end
    return rows
end

main()
