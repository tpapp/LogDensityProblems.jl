using Documenter, LogDensityProblems, Flux, ForwardDiff

makedocs(
    sitename = "LogDensityProblems.jl",
    modules = [LogDensityProblems],
    format = Documenter.HTML(),
    clean = true,
    authors = "Tamás K. Papp",
    checkdocs = :export,
    pages = Any["Documentation" => "index.md"]
)

deploydocs(
    repo = "github.com/tpapp/LogDensityProblems.jl.git",
)
