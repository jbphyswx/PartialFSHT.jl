#=
The two operators.

    mul!(f, plan, alm)     synthesis: coefficients → values at the points
    mul!(alm, plan', f)    the exact Euclidean adjoint

Both are exact at any sky coverage, and the pair satisfies ⟨𝒴a, f⟩ = ⟨a, 𝒴*f⟩ to round-off rather
than to the transform's tolerance — the adjoint is the forward pipeline transposed stage by stage,
not a quadrature inverse.
=#

using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using FINUFFT: FINUFFT          # any fast NUFFT backend; NonuniformFFTs works too
using Printf: Printf
using Random: Random
const PF = PartialFSHT

Random.seed!(0)

# A 20° polar cap, sampled at roughly 4× Nyquist for its own bandlimit.
lmax = 512
θstar = 20 * π / 180
M = 4 * round(Int, (lmax * θstar)^2)
θ = θstar .* sqrt.(rand(M))
φ = 2π .* rand(M)

plan = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8)
Printf.@printf("plan: %s\n", plan)
Printf.@printf("  retained (ℓ,m,ring) triples per apply: %.3e\n", PF.ring_work(plan))

# A red-spectrum field, filled only in the slots that are real modes: a coefficient array has
# padding the transform never reads, so `randn` over the whole thing is fine for comparing fields
# and wrong for comparing coefficients.
alm = PF.zeros_alm(lmax)
for ℓ in 0:lmax, m in (-ℓ):ℓ
    alm[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
end

f = zeros(ComplexF64, M)
LinearAlgebra.mul!(f, plan, alm)
Printf.@printf("\nsynthesis: |f|_∞ = %.6f over %d points\n", maximum(abs, f), M)

back = PF.zeros_alm(lmax)
LinearAlgebra.mul!(back, LinearAlgebra.adjoint(plan), f)
Printf.@printf("adjoint:   |A†f|_∞ = %.6f\n", maximum(abs, back))

# The adjoint identity. This is the property that makes the pair usable inside a solver.
g = randn(ComplexF64, M)
Ag = PF.zeros_alm(lmax)
LinearAlgebra.mul!(Ag, LinearAlgebra.adjoint(plan), g)
lhs = LinearAlgebra.dot(g, f)                 # ⟨f, 𝒴a⟩
rhs = LinearAlgebra.dot(Ag, alm)              # ⟨𝒴*f, a⟩
rel = abs(lhs - rhs) / abs(lhs)
Printf.@printf("\n⟨𝒴a, f⟩ = %.12e %+.12eim\n", real(lhs), imag(lhs))
Printf.@printf("⟨a, 𝒴*f⟩ = %.12e %+.12eim\n", real(rhs), imag(rhs))
Printf.@printf("relative difference: %.3e  (round-off, not the 1e-8 transform tolerance)\n", rel)
@assert rel < 1e-12 "adjoint identity failed at $rel"

# Allocation after construction.
#
# The package's own stages allocate nothing. The fast NUFFT backends do, at more than one Julia
# thread: FINUFFT's internal FFTW plan inherits its own OpenMP thread count, and executing it goes
# through FFTW.jl's Julia-task threading callback. `nufft_kwargs = (nthreads = 1,)` pins that away
# at roughly twice the NUFFT time — a trade this package leaves to the caller rather than making
# for them.
function alloc_of(p, alm, npts)
    f = zeros(ComplexF64, npts)
    a = PF.zeros_alm(p.lmax)
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
Printf.@printf("Julia threads = %d\n", Threads.nthreads())
fw, ad = alloc_of(plan, alm, M)
Printf.@printf("  default              : forward %8d B  adjoint %8d B\n", fw, ad)

pinned = PF.Plan(θ, φ; lmax = lmax, tol = 1e-8, nufft_kwargs = (nthreads = 1,))
fwp, adp = alloc_of(pinned, alm, M)
Printf.@printf("  nufft_kwargs=(nthreads=1,) : forward %8d B  adjoint %8d B\n", fwp, adp)
@assert fwp == 0 && adp == 0 "an apply allocated with the NUFFT pinned to one thread"

println("\nok")
