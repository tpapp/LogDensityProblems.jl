using Pkg
pkg"add https://github.com/tpapp/TransformVariables.jl"
pkg"activate ."
pkg"build"
Pkg.test(; coverage=true)
