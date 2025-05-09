"""
A unified interface for log density problems, for

1. defining mappings to a log density (eg Bayesian for inference),

2. optionally obtaining a gradient using automatic differentiation,

3. defining a common interface for working with such log densities and gradients (eg MCMC).

These use cases utilize different parts of this package, make sure you read the
documentation.
"""
module LogDensityProblems

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
using Random: AbstractRNG, default_rng

public LogDensityOrder, capabilities, dimension, logdensity, logdensity_and_gradient,
    logdensity_gradient_and_hessian, precompute, move, move!

####
#### interface for problems
####

"""
$(TYPEDEF)

A trait that means that a log density supports evaluating derivatives up to order `K`.

Typical values for `K` are `0` (just the log density) and `1` (log density and gradient).
"""
struct LogDensityOrder{K}
    function LogDensityOrder{K}() where K
        _K = Int(K)
        @argcheck _K ≥ 0
        new{_K}()
    end
end

LogDensityOrder(K::Integer) = LogDensityOrder{K}()

Base.isless(::LogDensityOrder{A}, ::LogDensityOrder{B}) where {A, B} = A < B

"""
$(SIGNATURES)

Test if the type (or a value, for convenience) supports the log density interface.

When `nothing` is returned, it doesn't support this interface. When
`LogDensityOrder{K}()` is returned (typically with `K == 0`, `K = 1`,
or `K == 2`), derivatives up to order `K` are supported. *All other
return values are invalid*.

# Interface description

The following methods **need to be implemented** for the interface:

1. [`dimension`](@ref) returns the *dimension* of the domain,

2. [`logdensity`](@ref) evaluates the log density at a given point.

3. [`logdensity_and_gradient`](@ref) when `K ≥ 1`.

4. [`logdensity_gradient_and_hessian`](@ref) when `K ≥ 2`.

The precomputation API (see below) has sensible fallbacks and should only be implemented
as needed.

# Coordinate points with precomputed information

The interface also allows for encapsulating extra information associated with
coordinates. The idea is that a coordinate ``x ∈ ℝⁿ`` may be associated with quantities
precomputed from `x` (such as solutions to implicit equations), which can be updated at
a lower computational cost when the position is changed instead of recomputed from
scratch.

The following methods **may be implemented**; but if they are not needed for your
application this package provides sensible defaults.

1. A type that encapsulates the precomputed information. It should be
   `<:AbstractVector{T}` for some `T` and support the read-only interface for vectors,
   ie `Base.size` and `Base.getindex`. When used as a vector, it should just correspond
   to the position in ``ℝⁿ``.

2. [`precompute`](@ref), which precomputes the relevant information and returns objects of the
   type above.

3. [`move`](@ref) and [`move!`](@ref) to change the coordinates and recompute the
   associated information.

# See also

[`LogDensityProblems.stresstest`](@ref) for stress testing.
"""
capabilities(T::Type) = nothing

capabilities(x) = capabilities(typeof(x)) # convenience function

"""
    dimension(ℓ)

Dimension of the input vectors `x` for log density `ℓ`. See [`logdensity`](@ref),
[`logdensity_and_gradient`](@ref).

!!! note
    This function is *distinct* from `TransformedVariables.dimension`.
"""
function dimension end

"""
    logdensity(ℓ, x)

Evaluate the log density `ℓ` at `x`, which has length compatible with its
[`dimension`](@ref).

Return a real number, which may or may not be finite (can also be `NaN`). Non-finite values
other than `-Inf` are invalid but do not error, caller should deal with these appropriately.

# Note about constants

Log densities can be shifted by *the same constant*, as long as it is consistent between
calls. For example,

```julia
logdensity(::StandardMultivariateNormal) = -0.5 * sum(abs2, x)
```

is a valid implementation for some callable `StandardMultivariateNormal` that would
implement the standard multivariate normal distribution (dimension ``k``) with pdf
```math
(2\\pi)^{-k/2} e^{-x'x/2}
```
"""
function logdensity end

"""
    logdensity_and_gradient(ℓ, x)

Evaluate the log density `ℓ` and its gradient at `x`, which has length
compatible with its [`dimension`](@ref).

Return two values:

- the log density as real number, which equivalent to `logdensity(ℓ, x)`

- *if* the log density is finite, the gradient, an `::AbstractVector` of real numbers,
   otherwise this value is arbitrary and should be ignored.

!!! note
    Caller may assume ownership of results, ie that the gradient vector will not be
    overwritten or reused for a different purpose.

The first argument (the log density) can be shifted by a constant, see the note for
[`logdensity`](@ref).
"""
function logdensity_and_gradient end

"""
    logdensity_gradient_and_hessian(ℓ, x)

Evaluate the log density `ℓ`, its gradient, and Hessian at `x`, which
has length compatible with its [`dimension`](@ref).

Return three values:

- the log density as real number, which equivalent to `logdensity(ℓ, x)`

- *if* the log density is finite, the gradient, an `::AbstractVector` of real numbers,
   otherwise this value is arbitrary and should be ignored.

- *if* the log density is finite, the Hessian, an `::AbstractMatrix` of real numbers,
   otherwise this value is arbitrary and should be ignored.

!!! note
    Caller may assume ownership of results, ie that the gradient and
    Hessian will not be overwritten or reused for a different purpose.

The first argument (the log density) can be shifted by a constant, see the note for
[`logdensity`](@ref).
"""
function logdensity_gradient_and_hessian end

"""
$(SIGNATURES)

Precompute information associated with the coordinates `x` and return it as a
user-defined type.

Cf [`move`](@ref).
"""
function precompute(ℓ, x::AbstractVector{T}) where T
    if T <: AbstractFloat
        x
    else
        float.(x)
    end
end

"""
$(SIGNATURES)

Change the coordinates by `Δ`. Conceptually equivalent to `x .+ Δ`, which is the
fallback implementation, but user-defined types for the second argument can take
advantage of the information there to compute the new associated information for small
changes.

The result should be **the same type** as `x` if `eltype(x) ≡ eltype(Δ)`.

Caller ensures that `x` is the result of [`precompute`](@ref). It is valid for an
implementation to error in all other cases.
"""
move(ℓ, x::AbstractVector, Δ) = x .+ Δ

"""
$(SIGNATURES)

Equivalent to [`move`](@ref), but *may* modify `x`, which is returned, or choose to
return a new value.

If a new value is returned, it should be the same type as `x` if `eltype(x) ≡ eltype(Δ)`.

Caller ensures that `x` is the result of [`precompute`](@ref). It is valid for an
implementation to error in all other cases.
"""
function move!(ℓ, x::AbstractVector, Δ)
    x .+= Δ
    x
end

include("utilities.jl")

end # module
