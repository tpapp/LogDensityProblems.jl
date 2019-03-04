#####
##### Types for containing the result of log density value and gradient evaluations.
#####
##### Also used to specify the kind of log density evaluation. Types are not exported, but
##### part of the API.
#####

export InvalidLogDensityException

"""
$(TYPEDEF)

Thrown when `Value` or `ValueGradient` is called with invalid arguments.
"""
struct InvalidLogDensityException{T} <: Exception
    "Location information: 0 is the value, positive integers are the gradient."
    index::Int
    "The invalid value."
    value::T
end

function Base.showerror(io::IO, ex::InvalidLogDensityException)
    @unpack index, value = ex
    print(io, "InvalidLogDensityException: ",
          index == 0 ? "value" : "gradient[$(index)]",
          " is $(value)")
end

struct Value{T <: Real}
    value::T
    function Value{T}(value::T) where {T <: Real}
        @argcheck (isfinite(value) || value == -Inf) InvalidLogDensityException(0, value)
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

Base.eltype(::Type{Value{T}}) where T = T

struct ValueGradient{T, V <: AbstractVector{T}}
    value::T
    gradient::V
    function ValueGradient{T,V}(value::T, gradient::V
                                ) where {T <: Real, V <: AbstractVector{T}}
        if value ≠ -Inf
            @argcheck isfinite(value) InvalidLogDensityException(0, value)
            invalid = findfirst(!isfinite, gradient)
            if invalid ≢ nothing
                throw(InvalidLogDensityException(invalid, gradient[invalid]))
            end
        end
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

Base.eltype(::Type{ValueGradient{T,V}}) where {T,V} = T

Base.isfinite(v::Union{Value, ValueGradient}) = isfinite(v.value)

Base.isinf(v::Union{Value, ValueGradient}) = isinf(v.value)

"""
$(TYPEDEF)

A wrapper for a vector that indicates that the vector *may* be used for the gradient in a
`ValueGradient`. Consequences are undefined if it is modified later, implicitly the caller
guarantees that it will not be used for anything else while the gradient is retrieved.

See [`logdensity`](@ref).
"""
struct ValueGradientBuffer{T <: Real, V <: AbstractVector{T}}
    buffer::V
    function ValueGradientBuffer(buffer::AbstractVector{T}) where T
        @argcheck T <: Real && isconcretetype(T)
        new{T,typeof(buffer)}(buffer)
    end
end

###
### conversion to DiffResult types — these are not part of the API
###

function DiffResults.MutableDiffResult(vgb::ValueGradientBuffer)
    @unpack buffer = vgb
    DiffResults.MutableDiffResult(first(buffer), (buffer, ))
end

function ValueGradient(result::DiffResults.DiffResult)
    ValueGradient(DiffResults.value(result), DiffResults.gradient(result))
end

###
### utilities
###

"""
$(SIGNATURES)

If `expr` evaluates to a non-finite value, `return` with that, otherwise evaluate to that
value. Useful for returning early from non-finite likelihoods.

Part of the API, but not exported.
"""
macro iffinite(expr)
    quote
        result = $(esc(expr))
        isfinite(result) || return result
        result
    end
end
