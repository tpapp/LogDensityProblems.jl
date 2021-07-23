using .BenchmarkTools: @belapsed
using .ForwardDiff

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

This function is not exported, but part of the API.

*It is loaded conditionally when both `ForwardDiff` and `BenchmarkTools` are loaded.*
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
