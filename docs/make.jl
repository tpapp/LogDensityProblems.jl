using Documenter, LogDensityProblems

makedocs(
    sitename = "LogDensityProblems.jl",
    modules = [LogDensityProblems],
    format = :html,
    clean = true,
    pages = Any["Overview" => "index.md"]
)
