#####
##### utilities
#####

####
#### random reals
####

function _random_reals_scale(rng::AbstractRNG, scale::Real, cauchy::Bool)
    cauchy ? scale / abs2(randn(rng)) : scale * 1.0
end

"""
$(SIGNATURES)

Random vector in ``ℝⁿ`` of length `n`.

A standard multivaritate normal or Cauchy is used, depending on `cauchy`, then scaled with
`scale`. `rng` is the random number generator used.

Not exported, but part of the API.
"""
function random_reals(n::Integer; scale::Real = 1, cauchy::Bool = false,
                      rng::AbstractRNG = default_rng())
    randn(rng, n) .* _random_reals_scale(rng, scale, cauchy)
end

####
#### stress testing
####

"""
$(SIGNATURES)

Test `ℓ` with random values.

`N` random vectors are drawn from a standard multivariate Cauchy distribution, scaled with
`scale` (which can be a scalar or a conformable vector).

Each random vector `x` is then used as an argument in `f(ℓ, x)`. [`logdensity`](@ref),
[`logdensity_and_gradient`](@ref), and [`logdensity_gradient_and_hessian`](@ref) are
recommended for `f`.

In case the call produces an error, the value is recorded as a failure, which are returned
by the function.

Not exported, but part of the API.
"""
function stresstest(f, ℓ; N = 1000, rng::AbstractRNG = default_rng(), scale = 1)
    failures = Vector{Float64}[]
    d = dimension(ℓ)
    for _ in 1:N
        x = random_reals(d; scale = scale, cauchy = true, rng = rng)
        try
            f(ℓ, x)
        catch e
            push!(failures, x)
        end
    end
    failures
end

####
#### converting input and output
####

"""
Type implementating [`converting_logdensity`](@ref).

Not part of the API *per se*, use the eponymous function to construct.
"""
struct ConvertingLogDensity{I,  # input type
                            L,  # logdensity output type
                            G,  # gradient output type
                            H,  # Hessian output type
                            P}
    parent::P
end

"""
$(SIGNATURES)

Return an object implementing the same [`capabilities`](@ref) as the first argument,
converting inputs and outputs are specified.

All conversions are implemented via `convert(T, …)::T`. Typical use cases include fixing
type instability introduced by automatic differentiation.

The original logdensity can be retrieved with `parent`.

# Keyword arguments (with defaults)

- `input = Any`: convert *inputs* to the given type before evaluating log densities, gradients, ….

- `logdensity = Any`: convert the *log density* to that type.

- `gradient = Any`: convert the *gradient* to that type.

- `hessian = Any`: convert the *Hessian* to that type.

# Note

The types are not checked for validity.

# Examples

```julia
converting_logdensity(ℓ; input = Vector{Float32}, logdensity = Float64,
                         gradient = Vector{Float64})
```
will convert to a vector of `Float32`s, then enforce Float64 elements for the logdensity
and its gradient, leaving the Hessian alone.
"""
function converting_logdensity(ℓ::P; input::Type = Any, logdensity::Type = Any, gradient::Type = Any,
                               hessian::Type = Any) where P
    @argcheck(capabilities(ℓ) ≥ LogDensityOrder(0),
              "Input does not implement the log density interface.")
    ConvertingLogDensity{input,logdensity,gradient,hessian,P}(ℓ)
end

Base.parent(ℓ::ConvertingLogDensity) = ℓ.parent

capabilities(ℓ::ConvertingLogDensity) = capabilities(ℓ.parent)

dimension(ℓ::ConvertingLogDensity) = dimension(ℓ.parent)

function logdensity(ℓ::ConvertingLogDensity{I,L}, x) where {I,L}
    convert(L, logdensity(ℓ.parent, convert(I, x)))::L
end

function logdensity_and_gradient(ℓ::ConvertingLogDensity{I,L,G}, x) where {I,L,G}
    l, g = logdensity_and_gradient(ℓ.parent, convert(I, x))
    convert(L, l)::L, convert(G, g)::G
end

function logdensity_gradient_and_hessian(ℓ::ConvertingLogDensity{I,L,G,H}, x) where {I,L,G,H}
    l, g, h = logdensity_gradient_and_hessian(ℓ.parent, convert(I, x))
    convert(L, l)::L, convert(G, g)::G, convert(H, h)::H
end
