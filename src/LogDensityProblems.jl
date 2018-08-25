module LogDensityProblems

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
using Parameters: @unpack
using TransformVariables: TransformReals, transform_logdensity

import ForwardDiff
import DiffResults

import Base: length

abstract type LogDensityProblem{R,T} end




include("evaluation.jl")
include("problem.jl")

end # module
