# Introduction

!!! note

    Currently this is just a placeholder that renders the docstrings.

# Problem definition

```@docs
TransformedLogDensity
ForwardDiffLogDensity
```

# Inference

```@docs
logdensity
dimension
LogDensityProblems.Value
LogDensityProblems.ValueGradient
```

# Benchmarking and diagnostics

```@docs
LogDensityProblems.stresstest
LogDensityProblems.benchmark_ForwardDiff_chunks
```

# Internals

```@docs
LogDensityProblems.AbstractLogDensityProblem
LogDensityProblems.LogDensityWrapper
LogDensityProblems.ADGradientWrapper
LogDensityProblems.heuristic_chunks
```
