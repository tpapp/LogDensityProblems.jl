# Unreleased

- clarify constant shifts in log densities

# 0.9.0

**Major, breaking API changes.**

1. Wrapper types `Value` and `ValueGradient` are removed. Interface functions return real numbers (`logdensity`), or a real number and a vector (`logdensity_and_gradient`). The calling convention of these types is simplified.

2. Capabilities of log density objects are described by traits, see `capabilities`.

3. `dimension` is now distinct from `TransformVariables.dimension`, as it is not really the same thing.

4. Condition-based wrappers removed, as it was interfering with AD (mostly Zygote).

5. Documentation significantly improved.

6. Code organized into a single file (except for conditional loads), tests greatly simplified.

# 0.8.3

- update to work with new Zygote interface

# 0.8.2

- no code change, just a version bump to fix registoy problems

# 0.8.1

- minor package and version fixes

# 0.8.0

- add `ValueGradientBuffer` for pre-allocated gradient storage ([#34](https://github.com/tpapp/LogDensityProblems.jl/pull/34))
- remove taped ReverseDiff-based AD ([#36](https://github.com/tpapp/LogDensityProblems.jl/pull/36))
- remove support for rejection-based unwindind of the stack ([#39](https://github.com/tpapp/LogDensityProblems.jl/pull/39))
- add support for Zygote-based AD ([#40](https://github.com/tpapp/LogDensityProblems.jl/pull/40), experimental)

# 0.7.0 and prior

Sorry, there is no changelog available before this version.
