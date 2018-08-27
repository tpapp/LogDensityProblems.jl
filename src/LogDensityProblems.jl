module LogDensityProblems

import Base: eltype, getproperty, propertynames

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
import DiffResults
import ForwardDiff
using Parameters: @unpack

using TransformVariables: TransformReals, transform_logdensity, RealVector
import TransformVariables: dimension

export logdensity, dimension, TransformedBayesianProblem, ForwardDiffLogDensity


# result types

finite_or_nothing(::Type{T}, ::Nothing) where T = nothing

@inline function _checked_minusinf(value)
    @argcheck value == -Inf "value is neither finite nor -∞"
    nothing
end

struct Value{T <: Real}
    value::T

    global function finite_or_nothing(::Type{Value}, value::T) where T
        T2 = promote_type(T, Float64)
        if isfinite(value)
            new{T2}(T2(value))
        else
            _checked_minusinf(value)
        end
    end

    function Value(value::Real)
        v = finite_or_nothing(Value, value)
        @argcheck v isa Value
        v
    end
end

eltype(::Type{Value{T}}) where T = T

struct ValueGradient{T, V <: AbstractVector{T}}
    value::T
    gradient::V

    global function finite_or_nothing(::Type{ValueGradient}, value::T1,
                               gradient::AbstractVector{T2}
                               ) where {T1 <: Real, T2 <: Real}
        if isfinite(value)
            @argcheck all(isfinite, gradient) "Finite value with non-finite gradient."
            T = promote_type(T1, T2, Float64)
            g = T ≡ T2 ? gradient : map(T, gradient)
            new{T,typeof(g)}(T(value), g)
        else
            _checked_minusinf(value)
        end
    end

    function ValueGradient(value::Real, gradient::AbstractVector{<: Real})
        vg = finite_or_nothing(ValueGradient, value, gradient)
        @argcheck vg isa ValueGradient
        vg
    end

end

eltype(::Type{ValueGradient{T,V}}) where {T,V} = T


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
    finite_or_nothing(Value,
                      transform_logdensity(transformation,
                                           θ -> logprior(θ) + loglikelihood(θ), x))
end


# wrappers

"FIXME: has a field ℓ, properties and dimenson are forwarded"
abstract type LogDensityWrapper <: AbstractLogDensityProblem end

dimension(w::LogDensityWrapper) = dimension(w.ℓ)

propertynames(w::LogDensityWrapper) = unique((fieldnames(w)..., propertynames(w.ℓ)...))

function getproperty(w::LogDensityWrapper, name::Symbol)
    if name ∈ fieldnames(typeof(w))
        getfield(w, name)
    else
        getproperty(w.ℓ, name)
    end
end

struct ForwardDiffLogDensity{L, C} <: LogDensityWrapper
    ℓ::L
    gradientconfig::C
end

@inline _value_closure(ℓ) =
    x -> (fx = logdensity(Value, ℓ, x); fx ≡ nothing ? oftype(eltype(x), -Inf) : fx.value)

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
    finite_or_nothing(ValueGradient,
                      DiffResults.value(result), DiffResults.gradient(result))
end

end # module
