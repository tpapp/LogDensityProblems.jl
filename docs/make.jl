using Documenter, LogDensityProblems, Flux, ForwardDiff

makedocs(
    sitename = "LogDensityProblems.jl",
    modules = [LogDensityProblems],
    format = Documenter.HTML(),
    clean = true,
    checkdocs = :export,
    pages = Any["Overview" => "index.md"]
)

deploydocs(
    repo = "github.com/tpapp/LogDensityProblems.jl.git",
)
