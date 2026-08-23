#=
Spin-weighted transforms.

    ₛY_{ℓm}(θ,φ) = √((2ℓ+1)/4π) · d^ℓ_{m,−s}(θ) · e^{imφ}

The Wigner-`d` here is *not* a rotation. It is the spin analogue of an associated Legendre
function: a three-term recurrence in *degree* at fixed `(m, −s)`, seeded at `ℓ₀ = max(|m|,|s|)` from
the extremal closed form and block-scaled exactly as the scalar recurrence, so `sin^{ℓ+m}(β/2)`
never underflows.

Same `Plan`, same backends, same adjoint guarantee — only `spin` changes.
=#

using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using FINUFFT: FINUFFT
using Printf: Printf
using Random: Random
const PF = PartialFSHT

Random.seed!(7)

# 1 and 2. A single spin mode against the closed-form harmonic, and spin 0 reducing to the scalar.
function single_mode()
    lmax, s, n = 32, 2, 300
    θ = 0.2 .+ 0.5 .* rand(n)
    φ = 2π .* rand(n)
    plan = PF.Plan(θ, φ; lmax = lmax, spin = s, method = PF.DirectAlgorithm())
    alm = PF.zeros_alm(lmax)
    alm[PF.spin_coeff_index(20, 7, lmax)] = 1
    ref = PF.sYlm.(s, 20, 7, θ, φ)
    rel = maximum(abs.((plan * alm) .- ref)) / maximum(abs, ref)
    Printf.@printf("spin %+d, single (ℓ,m) = (20,7) vs sYlm: rel = %.3e\n", s, rel)
    @assert rel < 1e-12

    p0 = PF.Plan(θ, φ; lmax = lmax, spin = 0, method = PF.DirectAlgorithm())
    a0 = PF.zeros_alm(lmax)
    a0[PF.sph_mode(20, 7)] = 1
    r0 = PF.ylm.(20, 7, θ, φ)
    rel0 = maximum(abs.((p0 * a0) .- r0)) / maximum(abs, r0)
    Printf.@printf("spin  0 reduces to ylm:                rel = %.3e\n", rel0)
    @assert rel0 < 1e-13
end

# 3. The fast path against the oracle, and the adjoint identity, at each spin weight.
function sweep_spins()
    lmax, n = 256, 400
    θ = (20 * π / 180) .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    alm = PF.zeros_alm(lmax)
    for ℓ in 0:lmax, m in (-ℓ):ℓ
        alm[PF.spin_coeff_index(ℓ, m, lmax)] = randn(ComplexF64) / (1 + ℓ)
    end
    g = randn(ComplexF64, n)
    Printf.@printf("\n%6s %14s %16s\n", "spin", "vs oracle", "adjoint identity")
    for s in (-2, -1, 0, 1, 2)
        pd = PF.Plan(θ, φ; lmax = lmax, spin = s, method = PF.DirectAlgorithm())
        ps = PF.Plan(θ, φ; lmax = lmax, spin = s, tol = 1e-8)
        fd = pd * alm
        fsv = ps * alm
        rel = maximum(abs.(fsv .- fd)) / maximum(abs, fd)
        Ag = ps' * g
        lhs = LinearAlgebra.dot(g, fsv)
        ident = abs(lhs - LinearAlgebra.dot(Ag, alm)) / abs(lhs)
        Printf.@printf("%6d %14.3e %16.3e\n", s, rel, ident)
        @assert rel < 1e-7 "spin $s fast path off by $rel"
        @assert ident < 1e-11 "spin $s adjoint identity off by $ident"
    end
end

# 4. The Wigner-d itself, against BigFloat, at a degree where a factorial sum has long since died.
function wigner_vs_bigfloat()
    Printf.@printf("\n%6s %6s %6s %14s %12s\n", "ℓ", "m", "n", "d^ℓ_mn", "rel vs BigFloat")
    for (ℓ, m, nn, β) in ((512, 100, -2, 0.8), (2048, 700, -1, 0.5), (4096, 30, -2, 1.1))
        v = PF.wigner_d(Float64, ℓ, m, nn, β)
        r = PF.wigner_d(BigFloat, ℓ, m, nn, BigFloat(β))
        rel = Float64(abs(v - r) / abs(r))
        Printf.@printf("%6d %6d %6d %14.6e %12.3e\n", ℓ, m, nn, v, rel)
        @assert rel < 1e-10
    end
end

single_mode()
sweep_spins()
wigner_vs_bigfloat()
println("\nok")
