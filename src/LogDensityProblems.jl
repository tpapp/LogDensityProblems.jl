module LogDensityProblems

import Base: eltype, getproperty, propertynames, isfinite, isinf, show

using ArgCheck: @argcheck
using BenchmarkTools: @belapsed
using DocStringExtensions: SIGNATURES, TYPEDEF
import DiffResults
import ForwardDiff
using Parameters: @unpack
using Random: AbstractRNG, GLOBAL_RNG

using TransformVariables: AbstractTransform, transform_logdensity, RealVector
import TransformVariables: dimension

export logdensity, dimension, TransformedLogDensity, ForwardDiffLogDensity


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

"""
    TransformedLogDensity(transformation, logposterior)

A problem in Bayesian inference. Vectors of length `dimension(transformation)` are
transformed into a general object `θ` (unrestricted type, but a named tuple is recommended
for clean code), correcting for the log Jacobian determinant of the transformation.

`logposterior(θ)` is expected to return *real numbers*. For zero densities or infeasible
`θ`s, `-Inf` or similar should be returned, but for efficiency of inference most methods
recommend using `transformation` to avoid this.

It is recommended that `logposterior` is a callable object that also
encapsulates the data for the problem.
"""
struct TransformedLogDensity{T <: AbstractTransform, L} <: AbstractLogDensityProblem
    transformation::T
    logposterior::L
end

show(io::IO, ℓ::TransformedLogDensity) =
    print(io, "TransformedLogDensity of dimension $(dimension(ℓ.transformation))")

"""
$(SIGNATURES)

The dimension of the problem, ie the length of the vectors in its domain.
"""
dimension(p::TransformedLogDensity) = dimension(p.transformation)

function logdensity(::Type{Value}, p::TransformedLogDensity, x::RealVector)
    @unpack transformation, logposterior = p
    Value(transform_logdensity(transformation, logposterior, x))
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

function show(io::IO, ℓ::ForwardDiffLogDensity)
    print(io, "ForwardDiff AD wrapper for ", ℓ.ℓ,
          ", w/ chunk size ", length(ℓ.gradientconfig.seeds))
end

@inline _value_closure(ℓ) = x -> logdensity(Value, ℓ, x).value

_anyargument(ℓ) = zeros(dimension(ℓ))

_default_chunk(ℓ) = ForwardDiff.Chunk(_anyargument(ℓ))

_default_gradientconfig(ℓ, chunk) =
    ForwardDiff.GradientConfig(_value_closure(ℓ), _anyargument(ℓ), chunk)

"""
$(SIGNATURES)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keywords are passed on to `ForwardDiff.GradientConfig` to customize the setup. In
particular, chunk size can be set with a `chunk = ForwardDiff.Chunk(n)` argument.
"""
function ForwardDiffLogDensity(ℓ::AbstractLogDensityProblem;
                               chunk::ForwardDiff.Chunk = _default_chunk(ℓ),
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

"""
$(SIGNATURES)

Default chunk sizes to try for benchmarking. Fewer than `M`, always contains `1` and `N`.
"""
function heuristic_chunks(N, M = 20)
    step = max(N ÷ M, 1)
    Ns = 1:step:N
    if N ∉ Ns
        Ns = vcat(Ns, N)
    end
    Ns
end

"""
$(SIGNATURES)

Benchmark a log density problem with various chunk sizes using ForwardDiff.

`chunks`, which defaults to all possible chunk sizes, determines the chunks that are tried.

The function returns `chunk => time` pairs, where `time` is the benchmarked runtime in
seconds, as determined by `BenchmarkTools.@belapsed`.

*Runtime may be long* because of tuned benchmarks, so when `markprogress == true` (the
 default), dots are printed to mark progress.

This function is not exported, but part of the API.
"""
function benchmark_ForwardDiff_chunks(ℓ::AbstractLogDensityProblem;
                                      chunks = heuristic_chunks(dimension(ℓ), 20),
                                      resulttype = ValueGradient,
                                      markprogress = true)
    map(chunks) do chunk
        ∇ℓ = ForwardDiffLogDensity(ℓ; chunk = ForwardDiff.Chunk(chunk))
        x = zeros(dimension(ℓ))
        markprogress && print(".")
        chunk => @belapsed logdensity($(resulttype), $(∇ℓ), $(x))
    end
end


# stress testing

"""
$(SIGNATURES)

Test `ℓ` with random values.

Random values are drawn from a standard multivariate Cauchy distribution, scaled with
`scale` (which can be a scalar or a conformable vector).

`N` elements are drawn, using `rng`. In case the call produces an error, the value is
recorded as a failure, failures are returned at the end.

Not exported, but part of the API.
"""
function stresstest(ℓ::AbstractLogDensityProblem;
                    N = 1000, rng::AbstractRNG = GLOBAL_RNG, scale = 1, resulttype = Value)
    failures = Vector{Float64}[]
    for _ in 1:N
        x = randn(dimension(ℓ))  .* scale ./ abs2(randn())
        try
            logdensity(resulttype, ℓ, x)
        catch e
            push!(failures, x)
        end
    end
    failures
end

end # module
