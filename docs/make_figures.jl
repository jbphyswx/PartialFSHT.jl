"""
Generate the manual's figures into `docs/src/assets/`.

Run standalone (`julia --project=docs docs/make_figures.jl`) or let `docs/make.jl` call it. Every
panel is computed from the package rather than drawn by hand, so a figure cannot drift away from the
behaviour it illustrates — if the window changes shape, the plot changes with it.
"""

using CairoMakie: CairoMakie
using PartialFSHT: PartialFSHT
using LinearAlgebra: LinearAlgebra
using Printf: Printf
using Random: Random
const PF = PartialFSHT

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)

CairoMakie.activate!(; type = "png", px_per_unit = 2)
const THEME = CairoMakie.Theme(
    fontsize = 13,
    Axis = (xgridvisible = false, ygridvisible = false,
            topspinevisible = false, rightspinevisible = false),
    Legend = (framevisible = false, patchsize = (12, 12)),
)
CairoMakie.set_theme!(THEME)

const PALETTE = (:steelblue, :firebrick, :seagreen, :darkorange3, :rebeccapurple)

save_fig(name, fig) = (CairoMakie.save(joinpath(ASSETS, name), fig); @info "wrote" name)

"""Decade ticks labelled `10⁻⁸` rather than Makie's default `10^-8.0`."""
function decades(lo::Real, hi::Real; step::Int = 1)
    es = collect(ceil(Int, log10(lo)):step:floor(Int, log10(hi)))
    sup = Dict('-' => '⁻', '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
               '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹')
    return (float.(10.0 .^ es), ["10" * map(c -> sup[c], string(e)) for e in es])
end

# ---------------------------------------------------------------------------------------
# 1. What the algorithm sees: a patch, its box, and the periodised interval
# ---------------------------------------------------------------------------------------
function fig_geometry()
    Random.seed!(1)
    θstar = 20 * π / 180
    n = 3000
    θ = θstar .* sqrt.(rand(n))
    φ = 2π .* rand(n)
    L, tol = 512, 1e-8
    box = PF.bounding_box(θ, φ)
    geom = PF.stretch_geometry(box.θlo, box.θhi, L, tol)
    # The ring count is the leaf's own, taken from the plan rather than chosen for drawing, so the
    # caption cannot claim a grid the transform does not build.
    plan = PF.Plan(θ, φ; lmax = L, tol = tol, method = PF.StretchAlgorithm(),
                   nufft = PF.DirectSumBackend())
    nrings = plan.leaf.nθ_half
    θr, _ = PF.stretch_rings(geom, nrings)

    fig = CairoMakie.Figure(size = (900, 380))

    ax1 = CairoMakie.Axis(fig[1, 1]; title = "the data, in the plane of the cap",
               xlabel = "θ cos φ  [rad]", ylabel = "θ sin φ  [rad]", aspect = CairoMakie.DataAspect())
    CairoMakie.scatter!(ax1, θ .* cos.(φ), θ .* sin.(φ); markersize = 2, color = (:steelblue, 0.35))
    # the padded interval the transform periodises over
    θmax = geom.h * (1 + geom.εapo)
    ϕc = range(0, 2π; length = 400)
    CairoMakie.lines!(ax1, box.θhi .* cos.(ϕc), box.θhi .* sin.(ϕc); color = :firebrick, linewidth = 2,
           label = Printf.@sprintf("data extent θ_hi = %.3f", box.θhi))
    CairoMakie.lines!(ax1, θmax .* cos.(ϕc), θmax .* sin.(ϕc); color = :darkorange3, linewidth = 2,
           linestyle = :dash, label = Printf.@sprintf("periodised to %.3f  (ε_apo = %.2f)", θmax, geom.εapo))
    CairoMakie.axislegend(ax1; position = :lt, labelsize = 10)

    ax2 = CairoMakie.Axis(fig[1, 2]; title = "rings actually evaluated, and the window on them",
               xlabel = "colatitude θ  [rad]", ylabel = "window w(θ)")
    Θr = [PF.theta_to_stretch(geom, t) for t in θr]
    wr = [PF.stretch_window(geom, Θ) for Θ in Θr]
    CairoMakie.vspan!(ax2, 0, box.θhi; color = (:steelblue, 0.12))
    CairoMakie.scatterlines!(ax2, θr, wr; color = :seagreen, markersize = 6)
    CairoMakie.text!(ax2, box.θhi / 2, 0.5; text = "data", align = (:center, :center), fontsize = 11,
          color = :steelblue)
    CairoMakie.text!(ax2, (box.θhi + θmax) / 2, 0.5; text = "apodisation\nannulus",
          align = (:center, :center), fontsize = 10, color = :darkorange3)
    CairoMakie.ylims!(ax2, -0.05, 1.08)

    CairoMakie.Label(fig[0, :], Printf.@sprintf("lmax = %d, tol = %.0e: %d rings and %d orders carry the whole \
                               transform (%.2e retained triples)",
                              L, tol, nrings, plan.leaf.mmax, PF.ring_work(plan));
          fontsize = 14, font = :bold)
    save_fig("geometry.png", fig)
end

# ---------------------------------------------------------------------------------------
# 2. The apodisation window and its derived parameters
# ---------------------------------------------------------------------------------------
function fig_window()
    fig = CairoMakie.Figure(size = (900, 350))
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "window on the stretched circle",
               xlabel = "Θ  [rad]", ylabel = "w(Θ)")
    ax2 = CairoMakie.Axis(fig[1, 2]; title = "derived apodisation fraction",
               xlabel = "lmax · h  (bandlimit × data half-extent)",
               ylabel = "ε_apo = √(2β / lmax·h)",
               xscale = log10, yscale = log10,
               yticks = ([0.05, 0.1, 0.25, 0.5, 1.0], ["0.05", "0.1", "0.25", "0.5", "1"]))

    for (i, tol) in enumerate((1e-4, 1e-8, 1e-12))
        geom = PF.stretch_geometry(0.0, 0.35, 1024, tol)
        Θ = range(0, π; length = 600)
        CairoMakie.lines!(ax1, Θ, [PF.stretch_window(geom, t) for t in Θ]; color = PALETTE[i],
               linewidth = 2, label = Printf.@sprintf("tol = %.0e  (Θ₁ = %.2f, σ = %.2f)",
                                               tol, geom.Θ1, geom.σ))
    end
    CairoMakie.axislegend(ax1; position = :lb, labelsize = 10)
    CairoMakie.ylims!(ax1, -0.05, 1.08)

    for (i, tol) in enumerate((1e-4, 1e-8, 1e-12))
        β = PF.es_beta(tol)
        Lh = 10 .^ range(log10(2.2 * β), 4.5; length = 200)
        CairoMakie.lines!(ax2, Lh, sqrt.(2β ./ Lh); color = PALETTE[i], linewidth = 2,
               label = Printf.@sprintf("tol = %.0e", tol))
    end
    CairoMakie.hlines!(ax2, [1.0]; color = :black, linestyle = :dash, linewidth = 1)
    CairoMakie.text!(ax2, 2e3, 0.72;
          text = "ε_apo → 1: padding as wide as the data.\nThe construction is refused past here —\nuse a direct evaluation.",
          fontsize = 9, align = (:center, :top))
    CairoMakie.ylims!(ax2, 0.03, 1.5)
    CairoMakie.axislegend(ax2; position = :lb, labelsize = 10)
    save_fig("window.png", fig)
end

# ---------------------------------------------------------------------------------------
# 3. Polar optimisation: the transition width is not a constant
# ---------------------------------------------------------------------------------------
function fig_polar()
    fig = CairoMakie.Figure(size = (900, 350))
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "orders retained per ring",
               xlabel = "colatitude θ  [rad]", ylabel = "m_max(θ)")
    θ = range(0.002, 0.6; length = 400)
    for (i, L) in enumerate((256, 1024, 4096))
        CairoMakie.lines!(ax1, θ, [PF.polar_mmax(t, L, 1e-8) for t in θ]; color = PALETTE[i],
               linewidth = 2, label = "lmax = $L")
        CairoMakie.lines!(ax1, θ, [min(L, (L + 0.5) * sin(t)) for t in θ]; color = PALETTE[i],
               linewidth = 1, linestyle = :dot)
    end
    CairoMakie.text!(ax1, 0.03, 2100; text = "dotted: the turning point ν sinθ alone —\nthe gap above it is the transition width",
          fontsize = 9, align = (:left, :top))
    CairoMakie.axislegend(ax1; position = :rb, labelsize = 10)

    ax2 = CairoMakie.Axis(fig[1, 2]; title = "transition width past the turning point (θ = 0.394)",
               xlabel = "degree ℓ", ylabel = "m_max − ν sinθ  [orders]",
               xscale = log10,
               xticks = ([100, 300, 1000, 3000, 10000], ["100", "300", "1000", "3000", "10⁴"]))
    ℓs = round.(Int, 10 .^ range(2, 4, length = 40))
    for (i, tol) in enumerate((1e-4, 1e-8, 1e-12))
        pad = [PF.polar_mmax(0.394, L, tol) - (L + 0.5) * sin(0.394) for L in ℓs]
        CairoMakie.scatterlines!(ax2, ℓs, pad; color = PALETTE[i], markersize = 5, linewidth = 2,
                      label = Printf.@sprintf("tol = %.0e", tol))
    end
    # the ℓ^{1/3} law the derivation predicts, at the middle tolerance
    ref = [0.5 * cbrt(9.0) * cbrt((L + 0.5) * sin(0.394)) * cbrt(log(1e8)^2) for L in ℓs]
    CairoMakie.lines!(ax2, ℓs, ref; color = :black, linestyle = :dash, linewidth = 1,
           label = "½·3^(2/3)·(ν sinθ)^(1/3)·ln(1/ε)^(2/3), ε = 1e-8")
    CairoMakie.axislegend(ax2; position = :lt, labelsize = 9)
    save_fig("polar.png", fig)
end

# ---------------------------------------------------------------------------------------
# 4. Delivered accuracy against the requested tolerance
# ---------------------------------------------------------------------------------------
function fig_accuracy()
    Random.seed!(2)
    θstar = 20 * π / 180
    npts = 120
    θ = θstar .* sqrt.(rand(npts))
    φ = 2π .* rand(npts)
    # `lmax·h ≥ 2|ln tol|` is the construction's own admissibility condition, so the grid of
    # (lmax, tol) is chosen to satisfy it everywhere rather than skipping cells silently —
    # a 20° cap at lmax = 128 genuinely cannot deliver 1e-10 and the package says so.
    Ls = (256, 512, 1024)
    tols = (1e-4, 1e-6, 1e-8, 1e-10)
    for L in Ls, tol in tols
        @assert PF.try_stretch_geometry(0.0, θstar, L, tol) !== nothing "(lmax=$L, tol=$tol) inadmissible"
    end

    fig = CairoMakie.Figure(size = (560, 380))
    ax = CairoMakie.Axis(fig[1, 1]; title = "delivered vs requested accuracy, 20° cap",
              xlabel = "requested tol", ylabel = "relative ℓ∞ error vs direct evaluation",
              xscale = log10, yscale = log10,
              xticks = decades(1e-10, 1e-4), yticks = decades(1e-13, 1e-5))
    CairoMakie.lines!(ax, [1e-11, 1e-3], [1e-11, 1e-3]; color = :black, linestyle = :dash,
           linewidth = 1, label = "delivered = requested")
    CairoMakie.text!(ax, 3e-9, 3e-12; text = "below the line: delivered better than requested",
          fontsize = 9, align = (:left, :center), color = :gray30)
    for (i, L) in enumerate(Ls)
        pd = PF.Plan(θ, φ; lmax = L, method = PF.DirectAlgorithm())
        alm = PF.zeros_alm(L)
        for ℓ in 0:L, m in (-ℓ):ℓ
            alm[PF.sph_mode(ℓ, m)] = randn(ComplexF64) / (1 + ℓ)
        end
        fd = pd * alm
        errs = Float64[]
        for tol in tols
            ps = PF.Plan(θ, φ; lmax = L, method = PF.StretchAlgorithm(), tol = tol,
                         nufft = PF.DirectSumBackend())
            fs = ps * alm
            push!(errs, maximum(abs.(fs .- fd)) / maximum(abs.(fd)))
        end
        CairoMakie.scatterlines!(ax, collect(tols), errs; color = PALETTE[i], markersize = 8,
                      linewidth = 2, label = "lmax = $L")
    end
    CairoMakie.axislegend(ax; position = :lt, labelsize = 10)
    save_fig("accuracy.png", fig)
end

# ---------------------------------------------------------------------------------------
# 5. Counted work: why a partial-sky leaf wins, and by how much
# ---------------------------------------------------------------------------------------
function fig_work()
    fig = CairoMakie.Figure(size = (900, 350))
    degtk = ([2, 5, 10, 20, 45, 90], ["2", "5", "10", "20", "45", "90"])
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "retained (ℓ, m, ring) triples vs cap radius",
               xlabel = "cap radius θ⋆  [deg]", ylabel = "triples per apply",
               xscale = log10, yscale = log10,
               xticks = degtk, yticks = decades(1e5, 1e9))
    ax2 = CairoMakie.Axis(fig[1, 2]; title = "…as a fraction of a full-sphere ring pass",
               xlabel = "cap radius θ⋆  [deg]", ylabel = "counted work / full sphere",
               xscale = log10, yscale = log10,
               xticks = degtk, yticks = decades(1e-3, 1e0))
    degs = 2.0 .^ range(1, log2(90), length = 22)
    for (i, L) in enumerate((256, 512, 1024))
        w = Float64[]
        for d in degs
            θstar = d * π / 180
            geom = PF.try_stretch_geometry(0.0, θstar, L, 1e-8)
            if geom === nothing
                push!(w, NaN)
                continue
            end
            nθ_half = PF.next_smooth(geom.K + 3)
            θr, _ = PF.stretch_rings(geom, nθ_half)
            push!(w, float(first(PF.counted_ring_work(θr, L, 1e-8))))
        end
        full = float(first(PF.counted_ring_work(collect(range(1e-6, π - 1e-6;
                                                             length = L + 1)), L, 1e-8)))
        CairoMakie.lines!(ax1, degs, w; color = PALETTE[i], linewidth = 2, label = "lmax = $L")
        CairoMakie.hlines!(ax1, [full]; color = PALETTE[i], linestyle = :dot, linewidth = 1)
        CairoMakie.lines!(ax2, degs, w ./ full; color = PALETTE[i], linewidth = 2, label = "lmax = $L")
    end
    CairoMakie.hlines!(ax2, [1.0]; color = :black, linestyle = :dash, linewidth = 1)
    CairoMakie.text!(ax1, 2.1, 4e8; text = "dotted: full sphere at the same lmax.\nCurves start where the construction\nbecomes admissible.",
          fontsize = 9, align = (:left, :top))
    CairoMakie.axislegend(ax1; position = :rb, labelsize = 10)
    CairoMakie.axislegend(ax2; position = :lt, labelsize = 10)
    save_fig("work.png", fig)
end

# ---------------------------------------------------------------------------------------
# 6. What partial data supports — the figure a user should look at before spending a CPU hour
# ---------------------------------------------------------------------------------------
function fig_recoverability()
    fig = CairoMakie.Figure(size = (900, 350))
    ax1 = CairoMakie.Axis(fig[1, 1]; title = "individual coefficients: the fraction constrained",
               xlabel = "f_sky", ylabel = "N_S / (lmax+1)²  =  f_sky",
               xscale = log10, yscale = log10,
               xticks = decades(1e-4, 1e0), yticks = decades(1e-4, 1e0))
    fs = 10 .^ range(-4, 0; length = 120)
    CairoMakie.lines!(ax1, fs, fs; color = :steelblue, linewidth = 2)
    named = (("4°×4° patch", 3.9e-4), ("SPT-3G main", 0.04),
             ("Atlantic basin", 0.2), ("hemisphere", 0.5))
    CairoMakie.scatter!(ax1, [x for (_, x) in named], [x for (_, x) in named];
             color = :firebrick, markersize = 9)
    for (nm, x) in named
        # the rightmost label would run off the axis
        right = x > 0.3
        CairoMakie.text!(ax1, x, x; text = right ? nm * "  " : "  " * nm,
              align = (right ? :right : :left, :top), fontsize = 10)
    end
    CairoMakie.text!(ax1, 1.2e-4, 0.5;
          text = "independent of ℓ — higher lmax brings\nproportionally more coefficients",
          fontsize = 10, align = (:left, :center))
    CairoMakie.xlims!(ax1, 8e-5, 2.0)

    # The payoff side: what *is* recoverable, and how well. Band-power scatter is
    # √(2 / ((2ℓ+1) f_sky Δℓ)), with Δℓ = π/Λ set by the region's own extent (cap: f_sky =
    # (1-cosθ⋆)/2, Λ = 2θ⋆). This is the number the manual quotes as "band-powers to ~5%".
    ax2 = CairoMakie.Axis(fig[1, 2]; title = "band-powers: the fractional error they come with",
               xlabel = "f_sky", ylabel = "ΔC_ℓ / C_ℓ  at the geometry's own Δℓ",
               xscale = log10, yscale = log10,
               xticks = decades(1e-4, 1e0), yticks = decades(1e-3, 1e0))
    for (i, ℓ) in enumerate((100, 1000, 3000))
        y = Float64[]
        for f in fs
            θstar = acos(1 - 2f)
            Δℓ = max(1.0, π / (2 * θstar))
            push!(y, sqrt(2 / ((2ℓ + 1) * f * Δℓ)))
        end
        CairoMakie.lines!(ax2, fs, y; color = PALETTE[i], linewidth = 2, label = "ℓ = $ℓ")
    end
    CairoMakie.hlines!(ax2, [0.05]; color = :black, linestyle = :dash, linewidth = 1)
    CairoMakie.text!(ax2, 1.5e-4, 0.058; text = "5%", fontsize = 9, align = (:left, :bottom))
    CairoMakie.axislegend(ax2; position = :lb, labelsize = 10)
    CairoMakie.xlims!(ax2, 8e-5, 1.2)
    save_fig("recoverability.png", fig)
end

fig_geometry()
fig_window()
fig_polar()
fig_accuracy()
fig_work()
fig_recoverability()
@info "figures written to $ASSETS"
