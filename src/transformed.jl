#####
##### Transformed log density (typically Bayesian inference)
#####

export TransformedLogDensity

"""
    TransformedLogDensity(transformation, log_density_function)

A problem in Bayesian inference. Vectors of length `dimension(transformation)` are
transformed into a general object `θ` (unrestricted type, but a named tuple is recommended
for clean code), correcting for the log Jacobian determinant of the transformation.

It is recommended that `log_density_function` is a callable object that also encapsulates
the data for the problem.

`log_density_function(θ)` is expected to return *real numbers*. For zero densities or
infeasible `θ`s, `-Inf` or similar should be returned, but for efficiency of inference most
methods recommend using `transformation` to avoid this.

Use the property accessors `ℓ.transformation` and `ℓ.log_density_function` to access the
arguments of `ℓ::TransformedLogDensity`, these are part of the API.
"""
struct TransformedLogDensity{T <: AbstractTransform, L} <: AbstractLogDensityProblem
    transformation::T
    log_density_function::L
end

function Base.show(io::IO, ℓ::TransformedLogDensity)
    print(io, "TransformedLogDensity of dimension $(dimension(ℓ.transformation))")
end

"""
$(SIGNATURES)

The dimension of the problem, ie the length of the vectors in its domain.
"""
TransformVariables.dimension(p::TransformedLogDensity) = dimension(p.transformation)

function logdensity(::Type{Real}, p::TransformedLogDensity, x::AbstractVector)
    @unpack transformation, log_density_function = p
    transform_logdensity(transformation, log_density_function, x)
end
