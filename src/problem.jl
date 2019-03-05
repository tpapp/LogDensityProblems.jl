#####
##### interface for problems
#####

export logdensity, dimension

"""
Abstract type for log density representations, which support the following
interface for `ℓ::AbstractLogDensityProblem`:

1. [`dimension`](@ref) returns the *dimension* of the domain of `ℓ`,

2. [`logdensity`](@ref) evaluates the log density `ℓ` at a given point.

See also [`LogDensityProblems.stresstest`](@ref) for stress testing.
"""
abstract type AbstractLogDensityProblem end

"""
    logdensity(resulttype, ℓ, x)

Evaluate the [`AbstractLogDensityProblem`](@ref) `ℓ` at `x`, which has length compatible
with its [`dimension`](@ref).

The argument `resulttype` determines the type of the result:

1. `Real` for an unchecked evaluation of the log density which should return a `::Real`
number (that could be `NaN`, `Inf`, etc),

1. [`Value`](@ref) for a checked log density, returning a `Value`,

2. [`ValueGradient`](@ref) also calculates the gradient, returning a `ValueGradient`,

3. [`ValueGradientBuffer`](@ref) calculates a `ValueGradient` *potentially* (but always
consistently for argument types) using the provided buffer for the gradient. In this case,
the element type of the array may determine the result element type.

# Implementation note

Most types should just define the methods for `Real` and `ValueGradientBuffer` (when
applicable), as `Value` and `ValueGradient` fall back to these, respectively.
"""
logdensity(::Type{Value}, ℓ, x::AbstractVector) = Value(logdensity(Real, ℓ, x))

function logdensity(::Type{ValueGradient}, ℓ, x::AbstractVector{T}) where T
    S = (T <: Real && isconcretetype(T)) ? T : Float64
    logdensity(ValueGradientBuffer(Vector{S}(undef, dimension(ℓ))), ℓ, x)
end

####
#### wrappers — general
####

"""
An abstract type that wraps another log density in its field `ℓ`.

# Notes

Implementation detail, *not exported*.
"""
abstract type LogDensityWrapper <: AbstractLogDensityProblem end

Base.parent(w::LogDensityWrapper) = w.ℓ

TransformVariables.dimension(w::LogDensityWrapper) = dimension(parent(w))
