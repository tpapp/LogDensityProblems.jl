import .ForwardDiff


# AD using ForwardDiff

struct ForwardDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

function show(io::IO, ℓ::ForwardDiffLogDensity)
    print(io, "ForwardDiff AD wrapper for ", ℓ.ℓ,
          ", w/ chunk size ", length(ℓ.gradientconfig.seeds))
end

_default_chunk(ℓ) = ForwardDiff.Chunk(dimension(ℓ))

_default_gradientconfig(ℓ, chunk::ForwardDiff.Chunk) =
    ForwardDiff.GradientConfig(_value_closure(ℓ), _vectorargument(ℓ), chunk)

_default_gradientconfig(ℓ, chunk::Integer) =
    _default_gradientconfig(ℓ, ForwardDiff.Chunk(chunk))

"""
$(SIGNATURES)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keywords are passed on to `ForwardDiff.GradientConfig` to customize the setup. In
particular, chunk size can be set with a `chunk` keyword argument (accepting an integer or a
`ForwardDiff.Chunk`).
"""
function ADgradient(::Val{:ForwardDiff},
                    ℓ::AbstractLogDensityProblem;
                    chunk = _default_chunk(ℓ),
                    gradientconfig = _default_gradientconfig(ℓ, chunk))
    ForwardDiffLogDensity(ℓ, gradientconfig)
end

function logdensity(::Type{ValueGradient}, fℓ::ForwardDiffLogDensity, x::RealVector)
    @unpack ℓ, gradientconfig = fℓ
    result = DiffResults.GradientResult(_vectorargument(ℓ)) # allocate a new result
    result = ForwardDiff.gradient!(result, _value_closure(ℓ), x, gradientconfig)
    ValueGradient(DiffResults.value(result), DiffResults.gradient(result))
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
seconds, as determined by `BenchmarkTools.@belapsed`.

*Runtime may be long* because of tuned benchmarks, so when `markprogress == true` (the
 default), dots are printed to mark progress.

This function is not exported, but part of the API.
"""
function benchmark_ForwardDiff_chunks(ℓ::AbstractLogDensityProblem;
                                      chunks = heuristic_chunks(dimension(ℓ), 20),
                                      resulttype = ValueGradient,
                                      markprogress = true)
    map(chunks) do chunk
        ∇ℓ = ADgradient(Val(:ForwardDiff), ℓ; chunk = ForwardDiff.Chunk(chunk))
        x = zeros(dimension(ℓ))
        markprogress && print(".")
        chunk => @belapsed logdensity($(resulttype), $(∇ℓ), $(x))
    end
end
