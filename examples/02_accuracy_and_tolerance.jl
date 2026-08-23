#=
What `tol` buys, measured against the oracle.

`DirectAlgorithm` evaluates the scaled associated Legendre functions per point: `O(M L²)`, and
correct at every degree. Everything fast is gated against it.

Two things worth seeing here. Delivered accuracy sits comfortably below the request — the
error budget is spent on the window's spectral tail, the NUFFT tolerance and the grid rounding, and
none of them is tight. And the construction *refuses* combinations it cannot deliver rather than
returning a loose answer: a small cap at low bandlimit has `lmax·h ≤ 2|ln tol|`, where the
apodisation would be wider than the data.
=#

using PartialFSHT: PartialFSHT
using FINUFFT: FINUFFT
using Printf: Printf
using Random: Random
const PF = PartialFSHT

Random.seed!(1)

# Everything runs inside functions: at top level, assignment inside a `for` is a *new local*, so a
# script that accumulates in a bare loop silently reads an undefined variable.
function tolerance_sweep(θ, φ, θstar)
    Printf.@printf("%6s %10s %14s %10s %8s\n", "lmax", "tol", "delivered", "ratio", "rings")
    for lmax in (128, 256, 512, 1024)
        pd = PF.Plan(θ, φ; lmax = lmax, method = PF.DirectAlgorithm())
        alm = PF.zeros_alm(lmax)
        for ℓ in 0:lmax, m in (-ℓ):ℓ
            alm[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
        end
        fd = pd * alm
        scale = maximum(abs, fd)
        for tol in (1e-4, 1e-8, 1e-12)
            if PF.try_stretch_geometry(0.0, θstar, lmax, tol) === nothing
                Printf.@printf("%6d %10.0e %14s\n", lmax, tol, "refused: lmax·h ≤ 2|ln tol|")
                continue
            end
            ps = PF.Plan(θ, φ; lmax = lmax, method = PF.StretchAlgorithm(), tol = tol)
            fs = ps * alm
            err = maximum(abs.(fs .- fd)) / scale
            Printf.@printf("%6d %10.0e %14.3e %9.2gx %8d\n", lmax, tol, err, err / tol, ps.leaf.nθ_half)
            @assert err < 10 * tol "delivered $err against a request of $tol"
        end
    end
end

# The oracle itself is verified against BigFloat, at orders *near the turning point* — the regime
# the block scaling exists for. There the sectorial seed `sin^m θ` has already underflowed to zero
# in IEEE-754 while the function itself is nowhere near zero, so an unscaled recurrence returns 0
# and takes the transform with it.
#
# Well *past* the turning point the function is genuinely below IEEE range (and polar optimisation
# drops it), so those orders are not a test of anything; the cases below stay where the answer is
# representable, and assert that it is.
#
# Both conditions hold at once only above `lmax ≈ 1925`: at the turning point `m = ν sinθ`, the seed
# underflows when `ν sinθ · |log₁₀ sinθ| > 308`, and `s|log₁₀ s|` peaks at `0.16` for `s = 1/e`.
# That is exactly why the underflow does not show up in a small-`lmax` test.
function oracle_vs_bigfloat()
    Printf.@printf("%6s %6s %10s %14s %14s %12s\n", "ℓ", "m", "νsinθ", "Float64", "BigFloat", "rel")
    for (ℓ, m, t) in ((2048, 700, 0.35), (4096, 1400, 0.35), (8192, 1600, 0.20))
        g = PF.alf(Float64, ℓ, m, t)
        r = PF.alf(BigFloat, ℓ, m, BigFloat(t))
        # the seed alone has underflowed: sin(θ)^m is exactly 0 in Float64 here
        @assert sin(t)^m == 0 "($ℓ, $m) does not exercise the scaled seed"
        @assert abs(r) > 1e-280 "($ℓ, $m) is below IEEE range — not a test of the recurrence"
        rel = Float64(abs(g - r) / abs(r))
        Printf.@printf("%6d %6d %10.1f %14.6e %14.6e %12.3e\n",
                ℓ, m, (ℓ + 0.5) * sin(t), g, Float64(r), rel)
        @assert rel < 1e-11 "alf disagrees with BigFloat at ($ℓ, $m) by $rel"
    end
end

let θstar = 20 * π / 180, npts = 150
    θ = θstar .* sqrt.(rand(npts))
    φ = 2π .* rand(npts)
    tolerance_sweep(θ, φ, θstar)
    println()
    oracle_vs_bigfloat()
end

println("\nok")
