# Introduction

This package provides the following functionality:

1. It defines the [`logdensity`](@ref) method with corresponding interface, which can be used by other packages that operate on (log) densities and need to evaluate the log densities or the gradients (eg [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation), [ML](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) or similar methods).

2. It defines the [`ADgradient`](@ref) which makes objects that support `logdensity` to calculate log density *values* calculate log density *gradients* using various automatic differentiation packages.

3. It defines the wrapper [`TransformedLogDensity`](@ref) using the [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) package, allowing callables that take a set of parameters transformed from a flat vector of real numbers to support the `logdensity` interface.

4. Various utility functions for debugging and testing log densities.

## Inference

```@docs
logdensity
dimension
LogDensityProblems.Value
LogDensityProblems.ValueGradient
LogDensityProblems.ValueGradientBuffer
```

## Gradient via automatic differentiation

```@docs
ADgradient
```

## Transformed problem definition

```@docs
TransformedLogDensity
```

## Benchmarking, diagnostics, and utilities

```@docs
LogDensityProblems.stresstest
LogDensityProblems.benchmark_ForwardDiff_chunks
LogDensityProblems.@iffinite
LogDensityRejectErrors
```