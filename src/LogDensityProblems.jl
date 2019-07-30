"""
A unified interface for log density problems, for

1. defining mappings to a log density (eg Bayesian for inference),

2. optionally obtaining a gradient using automatic differentiation,

3. defining a common interface for working with such log densities and gradients (eg MCMC).

These use cases utilize different parts of this package, make sure you read the
documentation.
"""
module LogDensityProblems

export TransformedLogDensity, ADgradient

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES, TYPEDEF
using Parameters: @unpack
import Random
using Requires: @require
import TransformVariables

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

Base.isless(::LogDensityOrder{A}, ::LogDensityOrder{B}) where {A, B} = A ≤ B

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

- *if* the log density is finite, the gradient, a vector of real numbers, otherwise this
   value is arbitrary and should be ignored.

The first argument (the log density) can be shifted by a constant, see the note for
[`logdensity`](@ref).
"""
function logdensity_and_gradient end

#####
##### Transformed log density (typically Bayesian inference)
#####

"""
    TransformedLogDensity(transformation, log_density_function)

A problem in Bayesian inference. Vectors of length compatible with the dimension (obtained
from `transformation`) are transformed into a general object `θ` (unrestricted type, but a
named tuple is recommended for clean code), correcting for the log Jacobian determinant of
the transformation.

`log_density_function(θ)` is expected to return *real numbers*. For zero densities or
infeasible `θ`s, `-Inf` or similar should be returned, but for efficiency of inference most
methods recommend using `transformation` to avoid this. It is recommended that
`log_density_function` is a callable object that also encapsulates the data for the problem.

Use the property accessors `ℓ.transformation` and `ℓ.log_density_function` to access the
arguments of `ℓ::TransformedLogDensity`, these are part of the API.

# Usage note

This is the most convenient way to define log densities, as `capabilities`, `logdensity`,
and `dimension` are automatically defined. To obtain a type that supports derivatives, use
[`ADgradient`](@ref).
"""
struct TransformedLogDensity{T <: TransformVariables.AbstractTransform, L}
    transformation::T
    log_density_function::L
end

function Base.show(io::IO, ℓ::TransformedLogDensity)
    print(io, "TransformedLogDensity of dimension $(dimension(ℓ))")
end

capabilities(::Type{<:TransformedLogDensity}) = LogDensityOrder{0}()

dimension(p::TransformedLogDensity) = TransformVariables.dimension(p.transformation)

function logdensity(p::TransformedLogDensity, x::AbstractVector)
    @unpack transformation, log_density_function = p
    TransformVariables.transform_logdensity(transformation, log_density_function, x)
end

#####
##### AD wrappers --- interface and generic code
#####

"""
An abstract type that wraps another log density for calculating the gradient via AD.

Automatically defines the methods `capabilities`, `dimension`, and `logdensity` forwarding
to the field `ℓ`, subtypes should define a [`logdensity_and_gradientent`](@ref).

This is an implementation helper, not part of the API.
"""
abstract type ADGradientWrapper end

logdensity(ℓ::ADGradientWrapper, x::AbstractVector) = logdensity(ℓ.ℓ, x)

capabilities(::Type{<:ADGradientWrapper}) = LogDensityOrder{1}()

dimension(ℓ::ADGradientWrapper) = dimension(ℓ.ℓ)

Base.parent(ℓ::ADGradientWrapper) = ℓ.ℓ

"""
$(SIGNATURES)

Wrap `P` using automatic differentiation to obtain a gradient.

`kind` is usually a `Val` type with a symbol that refers to a package, for example
```julia
ADgradient(Val(:ForwardDiff), P)
```

The symbol can also be used directly as eg

```julia
ADgradient(:ForwardDiff, P)
```

and should mostly be equivalent if the compiler manages to fold the constant.

`parent` can be used to retrieve the original argument.

See `methods(ADgradient)`. Note that **some methods are defined conditionally on the
relevant package being loaded.**
"""
ADgradient(kind::Symbol, P; kwargs...) = ADgradient(Val{kind}(), P; kwargs...)

function ADgradient(v::Val{kind}, P; kwargs...) where kind
    @info "Don't know how to AD with $(kind), consider `import $(kind)` if there is such a package."
    throw(MethodError(ADgradient, (v, P)))
end

####
#### AD wrappers - specific
####

function __init__()
    @require DiffResults="163ba53b-c6d8-5494-b064-1a9d43ac40c5" include("DiffResults_helpers.jl")
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("AD_ForwardDiff.jl")
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("AD_Flux.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("AD_ReverseDiff.jl")
    if VERSION ≥ v"1.1.0"
        # workaround for https://github.com/FluxML/Zygote.jl/issues/104
        @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("AD_Zygote.jl")
    end
end

####
#### stress testing
####

"""
$(SIGNATURES)

Test `ℓ` with random values.

`N` random vectors are drawn from a standard multivariate Cauchy distribution, scaled with
`scale` (which can be a scalar or a conformable vector).

Each random vector is then used as an argument in `f(ℓ, ...)`. `logdensity` and
`logdensity_and_gradient` are  recommended for `f`.

In case the call produces an error, the value is recorded as a failure, which are returned
by the function.

Not exported, but part of the API.
"""
function stresstest(f, ℓ; N = 1000, rng::Random.AbstractRNG = Random.GLOBAL_RNG, scale = 1)
    failures = Vector{Float64}[]
    d = dimension(ℓ)
    for _ in 1:N
        x = TransformVariables.random_reals(d; scale = scale, cauchy = true, rng = rng)
        try
            f(ℓ, x)
        catch e
            push!(failures, x)
        end
    end
    failures
end

end # module
