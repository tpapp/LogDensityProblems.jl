module LogDensityProblems

import Base: eltype, getproperty, propertynames, isfinite, isinf

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
import DiffResults
import ForwardDiff
using Parameters: @unpack

using TransformVariables: AbstractTransform, transform_logdensity, RealVector
import TransformVariables: dimension

export logdensity, dimension, TransformedBayesianProblem, ForwardDiffLogDensity


# result types

struct Value{T <: Real}
    value::T
    function Value{T}(value::T) where {T <: Real}
        @argcheck isfinite(value) || value == -Inf
        new{T}(value)
    end
end

"""
$(SIGNATURES)

Holds the value of a logdensity at a given point.

Constructor ensures that the value is either finite, or ``-∞``.

All other values (eg `NaN` or `Inf` for the `value`) lead to an error.

See also [`logdensity`](@ref).
"""
Value(value::T) where {T <: Real} = Value{T}(value)

eltype(::Type{Value{T}}) where T = T

struct ValueGradient{T, V <: AbstractVector{T}}
    value::T
    gradient::V
    function ValueGradient{T,V}(value::T, gradient::V
                                ) where {T <: Real, V <: AbstractVector{T}}
        @argcheck (isfinite(value) && all(isfinite, gradient))|| value == -Inf
        new{T,V}(value, gradient)
    end

end

"""
$(SIGNATURES)

Holds the value and gradient of a logdensity at a given point.

Constructor ensures that either

1. both the value and the gradient are finite,

2. the value is ``-∞`` (then gradient is not checked).

All other values (eg `NaN` or `Inf` for the `value`) lead to an error.

See also [`logdensity`](@ref).
"""
ValueGradient(value::T, gradient::V) where {T <: Real, V <: AbstractVector{T}} =
    ValueGradient{T,V}(value, gradient)

function ValueGradient(value::T1, gradient::AbstractVector{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    ValueGradient(T(value), T ≡ T2 ? gradient : map(T, gradient))
end

eltype(::Type{ValueGradient{T,V}}) where {T,V} = T

isfinite(v::Union{Value, ValueGradient}) = isfinite(v.value)

isinf(v::Union{Value, ValueGradient}) = isinf(v.value)


# interface for problems

"""
Abstract type for log density representations, which support the following
interface for `ℓ::AbstractLogDensityProblem`:

1. [`dimension`](@ref) returns the *dimension* of the domain of `ℓ`,

2. [`logdensity`](@ref) evaluates the log density `ℓ` at a given point.
"""
abstract type AbstractLogDensityProblem end

"""
    logdensity(resulttype, ℓ, x)

Evaluate the [`AbstractLogDensityProblem`](@ref) `ℓ` at `x`, which has length
compatible with its [`dimension`](@ref).

The argument `resulttype` determines the type of the result. [`Value`]@(ref)
results in the log density, while [`ValueGradient`](@ref) also calculates the
gradient, both returning eponymous types.
"""
function logdensity end

struct TransformedBayesianProblem{P, L, T <: AbstractTransform} <: AbstractLogDensityProblem
    transformation::T
    loglikelihood::L
    logprior::P
end

"""
    TransformedBayesianProblem(transformation, loglikelihood, [logprior])

A problem in Bayesian inference. Vectors of length `dimension(transformation)`
are transformed into a general object `θ` (unrestricted type, but a named tuple
is recommended for clean code).

`logprior(θ)` and `loglikelihood(θ)` are then called, returning *real numbers*,
the sum of which determining the log posterior, correcting for the log Jacobian
determinant of the transformation.

When `logprior` is omitted, it is taken to be `0` (in `θ`). In this case it is
assumed that `loglikelihood` also contains the prior density, or a flat prior is
used. This is for convenience only.

For zero densities or infeasible `θ`s, `-Inf` or similar should be returned, but
for efficiency of inference most methods recommend using `transformation` to
avoid this.

It is recommended that `loglikelihood` is a callable object that also
encapsulates the data for the problem.
"""
function TransformedBayesianProblem(transformation::AbstractTransform, loglikelihood)
    TransformedBayesianProblem(transformation, loglikelihood, _ -> 0.0)
end

"""
$(SIGNATURES)

The dimension of the problem, ie the length of the vectors in its domain.
"""
dimension(p::TransformedBayesianProblem) = dimension(p.transformation)

function logdensity(::Type{Value}, p::TransformedBayesianProblem, x::RealVector)
    @unpack transformation, loglikelihood, logprior = p
    Value(transform_logdensity(transformation, θ -> logprior(θ) + loglikelihood(θ), x))
end


# wrappers — general

"""
An abstract type that wraps another log density in its field `ℓ`.

# Notes

Implementation detail, *not exported*.

Forwards properties other than its field names to `ℓ`.
"""
abstract type LogDensityWrapper <: AbstractLogDensityProblem end

dimension(w::LogDensityWrapper) = dimension(w.ℓ)

propertynames(w::LogDensityWrapper) =
    unique((fieldnames(typeof(w))..., propertynames(w.ℓ)...))

function getproperty(w::LogDensityWrapper, name::Symbol)
    if name ∈ fieldnames(typeof(w))
        getfield(w, name)
    else
        getproperty(w.ℓ, name)
    end
end


# AD using ForwardDiff

struct ForwardDiffLogDensity{L, C} <: LogDensityWrapper
    ℓ::L
    gradientconfig::C
end

@inline _value_closure(ℓ) = x -> logdensity(Value, ℓ, x).value

_anyargument(ℓ) = zeros(dimension(ℓ))

_default_chunk(ℓ) = ForwardDiff.Chunk(_anyargument(ℓ))

_default_gradientconfig(ℓ, chunk) =
    ForwardDiff.GradientConfig(_value_closure(ℓ), _anyargument(ℓ), chunk)

"""
$(SIGNATURES)

Wrap a log density that supports evaluation of `Value` to handle
`ValueGradient`, using `ForwardDiff`.

Keywords are passed on to `ForwardDiff.GradientConfig` to customize the setup.
"""
function ForwardDiffLogDensity(ℓ::AbstractLogDensityProblem;
                               chunk = _default_chunk(ℓ),
                               gradientconfig = _default_gradientconfig(ℓ, chunk))
    ForwardDiffLogDensity(ℓ, gradientconfig)
end

logdensity(::Type{Value}, fℓ::ForwardDiffLogDensity, x::RealVector) =
    logdensity(Value, fℓ.ℓ, x)

function logdensity(::Type{ValueGradient}, fℓ::ForwardDiffLogDensity, x::RealVector)
    @unpack ℓ, gradientconfig = fℓ
    result = DiffResults.GradientResult(_anyargument(ℓ)) # allocate a new result
    result = ForwardDiff.gradient!(result, _value_closure(ℓ), x, gradientconfig)
    ValueGradient(DiffResults.value(result), DiffResults.gradient(result))
end

end # module
