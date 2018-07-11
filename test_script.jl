using Pkg
Pkg.add("https://github.com/tpapp/TransformVariables.jl")
Pkg.activate(".")
Pkg.build()
Pkg.test(; coverage=true)
