using Documenter, LogDensityProblems, ForwardDiff, Tracker, Zygote

makedocs(
    sitename = "LogDensityProblems.jl",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [LogDensityProblems],
    clean = true,
    authors = "Tamás K. Papp",
    checkdocs = :export,
    pages = Any["Documentation" => "index.md"]
)

deploydocs(
    repo = "github.com/tpapp/LogDensityProblems.jl.git",
    push_preview = true,
)
