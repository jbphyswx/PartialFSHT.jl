using Documenter
using PartialFSHT

makedocs(;
    modules = [PartialFSHT],
    sitename = "PartialFSHT.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Recoverability" => "recoverability.md",
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
