module LogDensityProblems

import Base: eltype

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
using Parameters: @unpack
using TransformVariables: TransformReals, transform_logdensity


# result types

function finite_or_nothing end

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

end # module
