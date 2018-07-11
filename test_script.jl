Pkg.add("https://github.com/tpapp/TransformVariables.jl")
Pkg.clone(pwd())
Pkg.build("LogDensityFramework")
Pkg.test("LogDensityFramework"; coverage=true)
