#=
What `AutoAlgorithm` picks, and why.

The planner counts: retained `(ℓ,m,ring)` triples for the ALF pass, `n log n` butterflies for the
FFTs, kernel writes for the NUFFT — each scaled by a per-unit constant measured once by
`benchmark/calibrate.jl`. There are no thresholds on `f_sky` or "is the patch compact": a band wins
on the stretch leaf because its colatitude interval is thin, a near-full sky loses to NUFSHT
because the counted work converges to NUFSHT's own, and a tiny point set loses to a direct
evaluation because `M L²` is smaller than a grid.
=#

using PartialFSHT: PartialFSHT
using FINUFFT: FINUFFT
using NUFSHT: NUFSHT          # makes NUFSHTAlgorithm available to the planner
using Printf: Printf
using Random: Random
const PF = PartialFSHT

Random.seed!(2)

function report(name, θ, φ, lmax)
    box = PF.bounding_box(θ, φ)
    cd = PF.cost_direct(length(θ), lmax, 1)
    cs = PF.cost_stretch(box, length(θ), lmax, 1, 1e-8)
    cn = PF.cost_nufsht(length(θ), lmax, 1, 1e-8)
    chosen = PF.select_method(θ, φ, lmax, 1e-8, PF.AutoAlgorithm(), 1, 0)
    Printf.@printf("%-22s M=%8d fsky=%7.4f | direct %9.2e  stretch %9.2e  nufsht %9.2e | %s\n",
            name, length(θ), PF.box_fsky(box), cd, cs, cn, nameof(typeof(chosen)))
    return chosen
end

lmax = 512
Printf.@printf("lmax = %d, tol = 1e-8   (costs are counted operations in one consistent unit)\n\n", lmax)

# 1. a compact polar cap — the case the stretched leaf exists for
n = 200_000
θc = (20 * π / 180) .* sqrt.(rand(n))
φc = 2π .* rand(n)
@assert report("20° polar cap", θc, φc, lmax) isa PF.StretchAlgorithm

# 2. an equatorial band — thin in colatitude, full in longitude. One box, not many caps.
θb = (π / 2) .+ (16 * π / 180) .* (rand(n) .- 0.5)
φb = 2π .* rand(n)
@assert report("±8° equatorial band", θb, φb, lmax) isa PF.StretchAlgorithm

# 3. two genuinely disjoint caps — one leaf each, only if counted cost falls
half = n ÷ 2
θ2 = vcat((10 * π / 180) .* sqrt.(rand(half)),
          π .- (10 * π / 180) .* sqrt.(rand(half)))
φ2 = 2π .* rand(n)
got = report("two polar caps", θ2, φ2, lmax)
Printf.@printf("%-22s %s\n", "", "→ boxes: $(length(PF.partition_boxes(θ2, φ2, (b, np) -> PF.cost_stretch(b, np, lmax, 1, 1e-8))))")

# 4. the whole sphere — the stretched interval degenerates and NUFSHT's full-sphere leaf wins
θf = acos.(1 .- 2 .* rand(n))
φf = 2π .* rand(n)
report("full sphere", θf, φf, lmax)

# 5. a handful of points — a grid costs more than evaluating them
θs = (20 * π / 180) .* sqrt.(rand(40))
φs = 2π .* rand(40)
@assert report("40 points on a cap", θs, φs, lmax) isa PF.DirectAlgorithm

Printf.@printf("\ncost constants calibrated on this machine: %s\n", PF.cost_calibrated())
println("  (run `julia --project=test benchmark/calibrate.jl` to measure them here)")
println("\nok")
