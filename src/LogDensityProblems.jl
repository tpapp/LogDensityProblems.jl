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
using DensityInterface
using DocStringExtensions: SIGNATURES, TYPEDEF
import Random
using Requires: @require
using StaticNumbers
import TransformVariables
using UnPack: @unpack

####
#### interface for problems
####


"""
$(SIGNATURES)

Test if the type (or a value, for convenience) supports the log density interface.

When `nothing` is returned, it doesn't support this interface.  When `LogDensityOrder`
is `K >= 1`), derivatives up to order `K` are supported.
*All other return values are invalid*.

# Interface description

The following methods need to be implemented for the interface:

1. [`dimension`](@ref) returns the *dimension* of the domain,

2. [`logdensityof`](@ref) evaluates the log density at a given point.

3. [`logdensity_and_gradient_of`](@ref) when `K ≥ 1`.

See also [`LogDensityProblems.stresstest`](@ref) for stress testing.
"""
LogDensityOrder(x) = nothing
    
"""
    dimension(ℓ)

Dimension of the input vectors `x` for log density `ℓ`. See [`logdensityof`](@ref),
[`logdensity_and_gradient_of`](@ref).

!!! note
    This function is *distinct* from `TransformedVariables.dimension`.
"""
function dimension end

Base.@deprecate logdensity(ℓ, x) DensityInterface.logdensityof(ℓ, x)

"""
    logdensity_and_gradient_of(ℓ, x)

Evaluate the log density `ℓ` and its gradient at `x`, which has length compatible with its
[`dimension`](@ref).

Return two values:

- the log density as real number, which equivalent to `logdensityof(ℓ, x)`

- *if* the log density is finite, the gradient, an `::AbstractVector` of real numbers,
   otherwise this value is arbitrary and should be ignored.

!!! note
    Caller may assume ownership of results, ie that the gradient vector will not be
    overwritten or reused for a different purpose.

The first argument (the log density) can be shifted by a constant, see the note for
[`logdensityof`](@ref).
"""
function logdensity_and_gradient_of end

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
arguments of `ℓ::TransformedLogDensity`, these are part of the public API.

# Usage note

This is the most convenient way to define log densities, as `LogDensityOrder`, `logdensityof`,
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

LogDensityOrder(::TransformedLogDensity) = static(0)

dimension(p::TransformedLogDensity) = TransformVariables.dimension(p.transformation)

DensityInterface.DensityKind(ℓ::TransformedLogDensity) = DensityInterface.DensityKind(ℓ.log_density_function)

function DensityInterface.logdensityof(p::TransformedLogDensity, x::AbstractVector)
    @unpack transformation, log_density_function = p
    TransformVariables.transform_logdensity(transformation, logdensityof(log_density_function), x)
end

#####
##### AD wrappers --- interface and generic code
#####

"""
An abstract type that wraps another log density for calculating the gradient via AD.

Automatically defines the methods `LogDensityOrder`, `dimension`, and `logdensityof` forwarding
to the field `ℓ`, subtypes should define a [`logdensity_and_gradient_of`](@ref).

This is an implementation helper, not part of the API.
"""
abstract type ADGradientWrapper end

DensityInterface.logdensityof(ℓ::ADGradientWrapper, x::AbstractVector) = logdensityof(ℓ.ℓ, x)

DensityInterface.DensityKind(ℓ::ADGradientWrapper) = DensityInterface.DensityKind(ℓ.ℓ)

LogDensityOrder(::ADGradientWrapper) = static(1)

dimension(ℓ::ADGradientWrapper) = dimension(ℓ.ℓ)

Base.parent(ℓ::ADGradientWrapper) = ℓ.ℓ

"""
$(SIGNATURES)

Wrap `P` using automatic differentiation to obtain a gradient.

`kind` is usually a `Val` type with a symbol that refers to a package, for example
```julia
ADgradient(Val(:ForwardDiff), P)
ADgradient(Val(:ReverseDiff), P)
ADgradient(Val(:Zygote), P)
```
Some methods may be loaded only conditionally after the relevant package is loaded (eg
`using Zygote`).

The symbol can also be used directly as eg

```julia
ADgradient(:ForwardDiff, P)
```

and should mostly be equivalent if the compiler manages to fold the constant.

The function `parent` can be used to retrieve the original argument.
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
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
        include("AD_ForwardDiff.jl")
        @require BenchmarkTools="6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf" begin
            include("ForwardDiff_benchmarking.jl")
        end

    end
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("AD_Tracker.jl")
    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("AD_Zygote.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("AD_ReverseDiff.jl")
    @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" include("AD_Enzyme.jl")
end

####
#### stress testing
####

"""
$(SIGNATURES)

Test `ℓ` with random values.

`N` random vectors are drawn from a standard multivariate Cauchy distribution, scaled with
`scale` (which can be a scalar or a conformable vector).

Each random vector is then used as an argument in `f(ℓ, ...)`. `logdensityof` and
`logdensity_and_gradient_of` are  recommended for `f`.

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
