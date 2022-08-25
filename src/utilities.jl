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

$(_RANDOM_REALS_KWARGS_DOC)
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

Each random vector is then used as an argument in `f(ℓ, ...)`. `logdensity` and
`logdensity_and_gradient` are  recommended for `f`.

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
