module LogDensityProblems

import Base: eltype, getproperty, propertynames, isfinite, isinf

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
import DiffResults
import ForwardDiff
using Parameters: @unpack

using TransformVariables: TransformReals, transform_logdensity, RealVector
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

abstract type AbstractLogDensityProblem end

# """FIXME
# """
function logdensity end

struct TransformedBayesianProblem{P, L, T <: TransformReals} <: AbstractLogDensityProblem
    logprior::P
    loglikelihood::L
    transformation::T
end

dimension(p::TransformedBayesianProblem) = dimension(p.transformation)

function logdensity(::Type{Value}, p::TransformedBayesianProblem, x::RealVector)
    @unpack logprior, loglikelihood, transformation = p
    Value(transform_logdensity(transformation, θ -> logprior(θ) + loglikelihood(θ), x))
end


# wrappers — general

"FIXME: has a field ℓ, properties and dimenson are forwarded"
abstract type LogDensityWrapper <: AbstractLogDensityProblem end

dimension(w::LogDensityWrapper) = dimension(w.ℓ)

propertynames(w::LogDensityWrapper) = unique((fieldnames(typeof(w))..., propertynames(w.ℓ)...))

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

function ForwardDiffLogDensity(ℓ;
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
