#####
##### Gradient AD implementation using ForwardDiff
#####

import .ForwardDiff
using BenchmarkTools: @belapsed

struct ForwardDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

function Base.show(io::IO, ℓ::ForwardDiffLogDensity)
    print(io, "ForwardDiff AD wrapper for ", ℓ.ℓ,
          ", w/ chunk size ", length(ℓ.gradientconfig.seeds))
end

_default_chunk(ℓ) = ForwardDiff.Chunk(dimension(ℓ))

# defined to make the tag match
_logdensity_closure(ℓ) = x -> logdensity(ℓ, x)

function _default_gradientconfig(ℓ, chunk::ForwardDiff.Chunk)
    ForwardDiff.GradientConfig(_logdensity_closure(ℓ), zeros(dimension(ℓ)), chunk)
end

function _default_gradientconfig(ℓ, chunk::Integer)
    _default_gradientconfig(ℓ, ForwardDiff.Chunk(chunk))
end

"""
$(SIGNATURES)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keywords are passed on to `ForwardDiff.GradientConfig` to customize the setup. In
particular, chunk size can be set with a `chunk` keyword argument (accepting an integer or a
`ForwardDiff.Chunk`).
"""
function ADgradient(::Val{:ForwardDiff}, ℓ;
                    chunk = _default_chunk(ℓ),
                    gradientconfig = _default_gradientconfig(ℓ, chunk))
    ForwardDiffLogDensity(ℓ, gradientconfig)
end

function logdensity_and_gradient(fℓ::ForwardDiffLogDensity, x::AbstractVector)
    @unpack ℓ, gradientconfig = fℓ
    buffer = _diffresults_buffer(ℓ, x)
    result = ForwardDiff.gradient!(buffer, _logdensity_closure(ℓ), x, gradientconfig)
    _diffresults_extract(result)
end

"""
$(SIGNATURES)

Default chunk sizes to try for benchmarking. Fewer than `M`, always contains `1` and `N`.
"""
function heuristic_chunks(N, M = 20)
    step = max(N ÷ M, 1)
    Ns = 1:step:N
    if N ∉ Ns
        Ns = vcat(Ns, N)
    end
    Ns
end

"""
$(SIGNATURES)

Benchmark a log density problem with various chunk sizes using ForwardDiff.

`chunks`, which defaults to all possible chunk sizes, determines the chunks that are tried.

The function returns `chunk => time` pairs, where `time` is the benchmarked runtime in
seconds, as determined by `BenchmarkTools.@belapsed`. The gradient is evaluated at `x`
(defaults to zeros).

*Runtime may be long* because of tuned benchmarks, so when `markprogress == true` (the
default), dots are printed to mark progress.

This function is not exported, but part of the API when `ForwardDiff` is imported.
"""
function benchmark_ForwardDiff_chunks(ℓ;
                                      chunks = heuristic_chunks(dimension(ℓ), 20),
                                      markprogress = true,
                                      x = zeros(dimension(ℓ)))
    map(chunks) do chunk
        ∇ℓ = ADgradient(Val(:ForwardDiff), ℓ; chunk = ForwardDiff.Chunk(chunk))
        markprogress && print(".")
        chunk => @belapsed logdensity_and_gradient($(∇ℓ), $(x))
    end
end
