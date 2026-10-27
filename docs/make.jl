using Documenter: Documenter
using PartialFSHT: PartialFSHT

# Figures are committed in src/assets. Regenerate them explicitly with the separate
# docs/generate_assets environment; see docs/README.md.

Documenter.makedocs(;
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

Documenter.deploydocs(;
    repo = "github.com/jbphyswx/PartialFSHT.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
