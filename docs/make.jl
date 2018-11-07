using Documenter, LogDensityProblems, Flux, ForwardDiff

makedocs(
    sitename = "LogDensityProblems.jl",
    modules = [LogDensityProblems],
    format = :html,
    clean = true,
    checkdocs = :export,
    pages = Any["Overview" => "index.md"]
)
