#####
##### Wrappers for automatic differentiation.
#####

export ADgradient

"""
An abstract type that wraps another log density for calculating the gradient via AD.

Automatically defines a `logdensity(Value, ...)` method, subtypes should define a
`logdensity(ValueGradient, ...)` one.
"""
abstract type ADGradientWrapper <: LogDensityWrapper end

function logdensity(::Type{Real}, fℓ::ADGradientWrapper, x::AbstractVector)
    logdensity(Real, fℓ.ℓ, x)
end

"""
$(SIGNATURES)

Return a closure that evaluetes the log density. Call this function to ensure stable tags
for `ForwardDiff`.
"""
_logdensity_closure(ℓ) = x -> logdensity(Real, ℓ, x)

"""
$(SIGNATURES)

Wrap `P` using automatic differentiation to obtain a gradient.

`kind` is usually a `Val` type, containing a symbol that refers to a package. The symbol can
also be used directly as eg

```julia
ADgradient(:ForwardDiff, P)
```

See `methods(ADgradient)`. Note that some methods are defined conditionally on the relevant
package being loaded.
"""
ADgradient(kind::Symbol, P; kwargs...) =
    ADgradient(Val{kind}(), P; kwargs...)

function ADgradient(v::Val{kind}, P; kwargs...) where kind
    @info "Don't know how to AD with $(kind), consider `import $(kind)` if there is such a package."
    throw(MethodError(ADgradient, (v, P)))
end

####
#### wrappers - specific
####

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("AD_ForwardDiff.jl")
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("AD_Flux.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("AD_ReverseDiff.jl")
    if VERSION ≥ v"1.1.0"
        # workaround for https://github.com/FluxML/Zygote.jl/issues/104
        @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("AD_Zygote.jl")
    end
end
