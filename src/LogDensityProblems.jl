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

The following methods need to be implemented for the interface:

1. [`dimension`](@ref) returns the *dimension* of the domain,

2. [`logdensity`](@ref) evaluates the log density at a given point.

3. [`logdensity_and_gradient`](@ref) when `K ≥ 1`.

4. [`logdensity_gradient_and_hessian`](@ref) when `K ≥ 2`.

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
Cf [`is_valid_result`](@ref).

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

Caller should be prepared to handle non-finite derivatives, even if they are incorrect.
Cf [`is_valid_result`](@ref).

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

Caller should be prepared to handle non-finite derivatives, even if they are incorrect.
Cf [`is_valid_result`](@ref).
"""
function logdensity_gradient_and_hessian end

"""
    is_valid_result(f, [∇f],  [∇²f])::Bool

Return `true` if and only if the log density `y` and its derivaties (optional) are *valid* in the sense defined below, otherwise `false`.

# Discussion

The API of this package defines an *interface* for working with log densities and gradients, but since the latter are implemented by the user and/or AD frameworks, it cannot impose *correctness* of these results. This function allows the caller to check for some common numerical problems conveniently.

Ideally, log densities are almost everywhere finite and differentiable, but practical computation often violates this assumption. The caller should be prepared to deal with this, either by throwing an error, rejecting that point, or some other way. Caller functions may of course allow the user to skip this check, which may result in a minor speedup, but could lead to bugs that are very hard to diagnose (as eg propagation of `NaN`s could cause problems much later in the code).

An example using this function would be
```julia
ℓq, ∇ℓq = logdensity_and_gradient(ℓ, q)
if is_valid_result(ℓq, ∇ℓq)
    # everything is finite, or log density is -Inf, proceed accordingly
    # ...
elseif !strict # an option in the API of the caller
    # something went wrong, but proceed and treat it an `-Inf`
    # ...
elseif is_valid_result(ℓq)
    error("Gradient has non-finite elements.")
else
    error("Invalid log posterior.")
end
```

# Definitions

Log densities are *valid* if they are *finite* real numbers or equal to ``-\\infty``.

Derivatives are *valid* if all elements are finite. But for ``-\\infty`` log density, the derivatives should be *ignored*.

*All other possibilities are invalid*, including

1. log densities that are not `::Real` (eg `1+2im`, `missing`),
2. non-finite log densities that are not `-Inf` (eg `NaN`, `Inf`),
3. derivatives (gradients or Hessians) with non-finite elements for finite log densities

Note that this function does not check

1. *dimensions* --- it is assumed that those kind of bugs are much more rare in AD implementations.

2. *symmetry of the Hessian* (cf Schwarz's/Clairaut's/Young's theorem).
"""
function is_valid_result end

# a version of `isfinite` that is only true for real numbers
_is_finite(x) = x isa Real && isfinite(x)

is_valid_result(args...) = false

is_valid_result(x::Real) = _is_finite(x) || x == -Inf

function is_valid_result(x::Real, ∇x)
    is_valid_result(x) || return false
    x == -Inf || (∇x isa AbstractVector && all(_is_finite, ∇x))
end

function is_valid_result(x::Real, ∇x, ∇²x)
    is_valid_result(x) || return false
    x == -Inf || (∇x isa AbstractVector && all(_is_finite, ∇x) && ∇²x isa AbstractMatrix && all(_is_finite, ∇²x))
end

include("utilities.jl")

end # module
