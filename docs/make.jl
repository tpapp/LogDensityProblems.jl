using Documenter, LogDensityProblems, Flux, ForwardDiff

makedocs(
    sitename = "LogDensityProblems.jl",
    modules = [LogDensityProblems],
    format = Documenter.HTML(),
    clean = true,
    authors = "Tamás K. Papp",
    checkdocs = :export,
    pages = Any["Home" => "index.md",
                "Internals" => "internals.md"]
)

deploydocs(
    repo = "github.com/tpapp/LogDensityProblems.jl.git",
)
