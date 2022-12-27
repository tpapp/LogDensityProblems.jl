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
using UnPack: @unpack

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

When `nothing` is returned, it doesn't support this interface.  When `LogDensityOrder{K}()`
is returned (typically with `K == 0` or `K = 1`), derivatives up to order `K` are supported.
*All other return values are invalid*.

# Interface description

The following methods need to be implemented for the interface:

1. [`dimension`](@ref) returns the *dimension* of the domain,

2. [`logdensity`](@ref) evaluates the log density at a given point.

3. [`logdensity_and_gradient`](@ref) when `K ≥ 1`.

See also [`LogDensityProblems.stresstest`](@ref) for stress testing.
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

Evaluate the log density `ℓ` and its gradient at `x`, which has length compatible with its
[`dimension`](@ref).

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

include("utilities.jl")

end # module
