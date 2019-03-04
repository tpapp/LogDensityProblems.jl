####
#### stress testing
####

TransformVariables.random_arg(ℓ::AbstractLogDensityProblem; kwargs...) = random_reals(dimension(ℓ); kwargs...)

"""
$(SIGNATURES)

Test `ℓ` with random values.

`N` random vectors are drawn from a standard multivariate Cauchy distribution, scaled with
`scale` (which can be a scalar or a conformable vector). In case the call produces an error,
the value is recorded as a failure, which are returned by the function.

Not exported, but part of the API.
"""
function stresstest(ℓ; N = 1000, rng::AbstractRNG = GLOBAL_RNG, scale = 1, resulttype = Value)
    failures = Vector{Float64}[]
    for _ in 1:N
        x = random_arg(ℓ; scale = scale, cauchy = true, rng = rng)
        try
            logdensity(resulttype, ℓ, x)
        catch e
            push!(failures, x)
        end
    end
    failures
end
