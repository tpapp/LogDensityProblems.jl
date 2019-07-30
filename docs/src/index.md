# Introduction

This package serves two purposes:

1. Introduce a common API for packages that operate on *log densities*, which for these purposes are black box ``\mathbb{R}^n \to \mathbb{R}`` mappings. Using the interface of introduced in this package, you can query ``n``, evaluate the log density and optionally its gradient, and determine if a particular object supports these methods using traits. **This usage is relevant primarily for package developers** who write generic algorithms that use (log) densities that correspond to posteriors and likelihoods, eg [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation), [ML](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation). An example is [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl). This is documented in [the API section](@ref log-density-api).

2. Make it easier for **users who want to perform inference** using the above methods (and packages) to
    - *define their own log densities*, either taking a vector of real numbers as input, or extracting and
      transforming parameters using the [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl)
      package,
    - *obtain gradients* of these log densities using one of the supported *automatic differentiation* packages
      of Julia.
This is documented in the next section, with a worked example.

For the purposes of this package, *log densities* are still valid when shifted by a constant that may be unknown, but is consistent within calls. This is necessary for Bayesian inference, where log posteriors are usually calculated up to a constant. See [`LogDensityProblems.logdensity`](@ref) for details.

# Working with log density problems

Consider an inference problem where IID draws are obtained from a normal distribution,
```math
x_i \sim N(\mu, \sigma)
```
for ``i = 1, \dots, N``. It can be shown that the *log likelihood* conditional on ``\mu`` and ``\sigma`` is
```math
\ell = -N\log \sigma - \sum_{i = 1}^N \frac{(x-\mu)^2}{2\sigma^2} =
-N\left( \log \sigma + \frac{S + (\bar{x} - \mu)^2}{2\sigma^2} \right)
```
where we have dropped constant terms, and defined the sufficient statistics
```math
\bar{x} = \frac{1}{N} \sum_{i = 1}^N x_i
```
and
```math
S = \frac{1}{N} \sum_{i = 1}^N (x_i - \bar{x})^2
```

Finally, we use priors
```math
\mu \sim N(0, 5), \sigma \sim N(0, 2)
```
which yield the log prior
```math
-\sigma^2/8 - \mu^2/50
```
which is added to the log likelihood to obtain the log posterior.

It is useful to define a *callable* that implements this, taking some vector `x` as an input and calculating the summary statistics, then, when called with a `NamedTuple` containing the parameters, evaluating to the log posterior.

```@example 1
using Random; Random.seed!(1) # hide
using Statistics, Parameters # imported for our implementation

struct NormalPosterior{T} # contains the summary statistics
    N::Int
    x̄::T
    S::T
end

# calculate summary statistics from a data vector
function NormalPosterior(x::AbstractVector)
    NormalPosterior(length(x), mean(x), var(x; corrected = false))
end

# define a callable that unpacks parameters, and evaluates the log likelihood
function (problem::NormalPosterior)(θ)
    @unpack μ, σ = θ
    @unpack N, x̄, S = problem
    loglikelihood = -N * (log(σ) + (S + abs2(μ - x̄)) / (2 * abs2(σ)))
    logprior = - abs2(σ)/8 - abs2(μ)/50
    loglikelihood + logprior
end

problem = NormalPosterior(randn(100))
nothing # hide
```

Let's try out the posterior calculation:

```@repl 1
problem((μ = 0.0, σ = 1.0))
```

!!! note
    Just evaluating your log density function like above is a great way to test and benchmark your implementation. See the “Performance Tips” section of the Julia manual for optimization advice.

## Using the TransformVariables package

In our example, we require ``\sigma > 0``, otherwise the problem is meaningless. However, many MCMC samplers prefer to operate on *unconstrained* spaces ``\mathbb{R}^n``. The TransformVariables package was written to transform unconstrained to constrained spaces, and help with the log Jacobian correction (more on that later). That package has detailed documentation, now we just define a transformation from a length 2 vector to a `NamedTuple` with fields `μ` (unconstrained) and `σ > 0`.

```@repl 1
using LogDensityProblems, TransformVariables
ℓ = TransformedLogDensity(as((μ = asℝ, σ = asℝ₊)), problem)
```

Then we can query the dimension of this problem, and evaluate the log density:
```@repl 1
LogDensityProblems.dimension(ℓ)
LogDensityProblems.logdensity(ℓ, zeros(2))
```

!!! note
    Before running time-consuming algorithms like MCMC, it is advisable to test and benchmark your log density evaluations separately. The same applies to [`LogDensityProblems.logdensity_and_gradient`](@ref).

```@docs
TransformedLogDensity
```

## Manual unpacking and transformation

If you prefer to implement the transformation yourself, you just have to define the following three methods for your problem: declare that it can evaluate log densities (but not their gradient, hence the `0` order), allow the dimension of the problem to be queried, and then finally code the density calculation with the transformation. (Note that using [`TransformedLogDensity`](@ref) takes care of all of these for you, as shown above).

```@example 1
function LogDensityProblems.capabilities(::Type{<:NormalPosterior})
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(::NormalPosterior) = 2

function LogDensityProblems.logdensity(problem::NormalPosterior, x)
    μ, logσ = x
    σ = exp(logσ)
    problem((μ = μ, σ = σ)) + logσ
end
nothing # hide
```

```@repl 1
LogDensityProblems.logdensity(problem, zeros(2))
```

Here we use the exponential function to transform from ``\mathbb{R}`` to the positive reals, but this requires that we correct the log density with the *logarithm* of the Jacobian, which here happens to be ``\log(\sigma)``.

## Automatic differentiation

Using either definition, you can now transform to another object which is capable of evaluating the *gradient*, using automatic differentiation. The wrapper for this is
```@docs
ADgradient
```
Note that support for Zygote is experimental. At the moment, I would recommend that you use `Flux`.

Now observe that we can obtain gradients, too:
```@repl 1
import ForwardDiff
∇ℓ = ADgradient(:ForwardDiff, ℓ)
LogDensityProblems.capabilities(∇ℓ)
LogDensityProblems.logdensity_and_gradient(∇ℓ, zeros(2))
```

## Manually calculated derivatives

If you prefer not to use automatic differentiation, you can wrap your own derivatives following the template
```julia
function LogDensityProblems.capabilities(::Type{<:NormalPosterior})
    LogDensityProblems.LogDensityOrder{1}() # can do gradient
end

LogDensityProblems.dimension(::NormalPosterior) = 2 # for this problem

function LogDensityProblems.logdensity_and_gradient(problem::NormalPosterior, x)
    logdens = ...
    grad = ...
    logdens, grad
end
```

# Various utilities

You may find these utilities useful for debugging and optimization.

```@docs
LogDensityProblems.stresstest
LogDensityProblems.benchmark_ForwardDiff_chunks
```

# [Log densities API](@id log-density-api)

Use the functions below for evaluating gradients and querying their dimension and other information. These symbols are not exported, as they are mostly used by package developers and in any case would need to be `import`ed or qualified to add methods to.

```@docs
LogDensityProblems.capabilities
LogDensityProblems.LogDensityOrder
LogDensityProblems.dimension
LogDensityProblems.logdensity
LogDensityProblems.logdensity_and_gradient
```
