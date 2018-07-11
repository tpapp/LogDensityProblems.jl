__precompile__()
module LogDensityFramework

using ArgCheck: @argcheck

using DocStringExtensions: SIGNATURES
using Random: AbstractRNG
using Parameters: @unpack
using TransformVariables: TransformReals, transform_logdensity

import ForwardDiff
import DiffResults

import Base: length

include("evaluation.jl")

end # module
