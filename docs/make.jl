using Documenter
using PartialFSHT

# Figures are computed from the package, so they cannot drift from the behaviour they illustrate.
# Set PARTIALFSHT_SKIP_FIGURES=1 to reuse whatever is already in `src/assets`.
if get(ENV, "PARTIALFSHT_SKIP_FIGURES", "0") != "1"
    include(joinpath(@__DIR__, "make_figures.jl"))
end

makedocs(;
    modules = [PartialFSHT],
    sitename = "PartialFSHT.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
        assets = String[],
        size_threshold = 400 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Theory" => "theory.md",
        "Recoverability" => "recoverability.md",
        "Implementation notes" => "internals.md",
        "API" => "api.md",
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(;
    repo = "github.com/jbphyswx/PartialFSHT.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
