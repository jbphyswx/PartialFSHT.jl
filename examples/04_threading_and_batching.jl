#=
Threading, batching, and why the apply allocates nothing.

Run with `julia --project=examples -t auto examples/04_threading_and_batching.jl`.

Threading is over spherical harmonic order in the ring pass. Orders are independent — going
forward, order `m` writes only `G[:, ±m]`; coming back it writes only the `(ℓ, ±m)` coefficients —
so the loop parallelises with no reduction, and the threaded result is **bit-identical** to the
serial one because each order is summed in the same sequence regardless of which task runs it.

Batching (`ntrans`) raises arithmetic intensity rather than parallelism: one `P̄` value then feeds
`2·ntrans` accumulations against unchanged recurrence-state traffic.

FFT plans are deliberately built single-threaded. A multithreaded FFTW plan spawns Julia tasks on
*every execution*, so a threaded FFT would make each apply allocate; the FFT is a small share of an
apply while the ring pass, which is threaded directly, is most of it. Pass `fft_threads` to
override.
=#

using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using FINUFFT: FINUFFT
using OhMyThreads: OhMyThreads          # required for ThreadedBackend
using ComputationalBackends: ComputationalBackends
using Printf: Printf
using Random: Random
const PF = PartialFSHT
const CB = ComputationalBackends

best(f, n = 3) = (f(); minimum(@elapsed f() for _ in 1:n))

Random.seed!(3)
Printf.@printf("threads = %d\n\n", Threads.nthreads())

lmax = 512
θstar = 20 * π / 180
M = 4 * round(Int, (lmax * θstar)^2)
θ = θstar .* sqrt.(rand(M))
φ = 2π .* rand(M)

alm = PF.zeros_alm(lmax)
for ℓ in 0:lmax, m in (-ℓ):ℓ
    alm[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
end

ps = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8, backend = CB.SerialBackend())
pt = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8, backend = CB.ThreadedBackend())
fs = zeros(ComplexF64, M)
ft = zeros(ComplexF64, M)

ts = best(() -> LinearAlgebra.mul!(fs, ps, alm))
tt = best(() -> LinearAlgebra.mul!(ft, pt, alm))
Printf.@printf("serial   %.4f s\nthreaded %.4f s   (%.2fx)\n", ts, tt, ts / tt)

# Bit-identical, not merely close.
Printf.@printf("\nthreaded == serial bitwise: %s\n", fs == ft)
@assert fs == ft "threading changed the result"

# Allocation. This package's own stages are 0 B in both directions at any thread count; the fast
# NUFFT backends are not, above one Julia thread, because FFTW.jl's threading callback spawns tasks
# from inside FINUFFT's internal FFT. `nufft_kwargs = (nthreads = 1,)` removes that.
function alloc_of(p, alm, npts)
    f = zeros(ComplexF64, npts)
    a = PF.zeros_alm(p.lmax; ntrans = p.ntrans)
    LinearAlgebra.mul!(f, p, alm); LinearAlgebra.mul!(f, p, alm)
    fw = min(@allocated(LinearAlgebra.mul!(f, p, alm)),
             @allocated(LinearAlgebra.mul!(f, p, alm)))
    LinearAlgebra.mul!(a, LinearAlgebra.adjoint(p), f)
    LinearAlgebra.mul!(a, LinearAlgebra.adjoint(p), f)
    ad = min(@allocated(LinearAlgebra.mul!(a, LinearAlgebra.adjoint(p), f)),
             @allocated(LinearAlgebra.mul!(a, LinearAlgebra.adjoint(p), f)))
    return fw, ad
end

println()
for (nm, p) in (("serial", ps), ("threaded", pt))
    af, aa = alloc_of(p, alm, M)
    Printf.@printf("%-9s default        : forward %7d B  adjoint %7d B\n", nm, af, aa)
end

# With the NUFFT pinned, a serial apply is exactly zero. A threaded apply keeps a task-spawn floor
# — OhMyThreads allocates a Task per slot on every call — which is a *fixed* cost: it must not grow
# with the problem, and that is the property worth asserting.
p_ser = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8, backend = CB.SerialBackend(),
                nufft_kwargs = (nthreads = 1,))
af, aa = alloc_of(p_ser, alm, M)
Printf.@printf("serial    + NUFFT pinned : forward %7d B  adjoint %7d B\n", af, aa)
@assert af == 0 && aa == 0 "a serial apply allocated with the NUFFT pinned to one thread"

floors = Int[]
for L in (256, 512)
    almL = PF.zeros_alm(L)
    for ℓ in 0:L, m in (-ℓ):ℓ
        almL[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
    end
    p = PF.Plan(θ, φ; lmax = L, tol = 1e-8, backend = CB.ThreadedBackend(),
                nufft_kwargs = (nthreads = 1,))
    f, _ = alloc_of(p, almL, M)
    push!(floors, f)
    Printf.@printf("threaded  + NUFFT pinned : forward %7d B at lmax = %4d\n", f, L)
end
@assert floors[2] <= floors[1] + 64 "the threaded task-spawn cost grows with lmax"
println("  → a fixed task-spawn floor, not growth: the work itself allocates nothing")

# Batching: cost per retained triple falls because the recurrence state is reused.
println()
Printf.@printf("%8s %12s %14s %12s\n", "ntrans", "apply s", "s per transform", "triples")
for B in (1, 2, 4)
    almB = PF.zeros_alm(lmax; ntrans = B)
    for b in 1:B, ℓ in 0:lmax, m in (-ℓ):ℓ
        almB[PF.sph_mode(ℓ, m), b] = randn(ComplexF64) / (1 + ℓ)
    end
    p = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8, ntrans = B, backend = CB.SerialBackend())
    fB = zeros(ComplexF64, M, B)
    t = best(() -> LinearAlgebra.mul!(fB, p, almB))
    Printf.@printf("%8d %12.4f %14.4f %12.3e\n", B, t, t / B, PF.ring_work(p))
end

println("\nok")
