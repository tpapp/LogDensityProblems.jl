#####
##### Gradient AD implementation using ReverseDiff
#####

import .ReverseDiff

import .ReverseDiff.DiffResults # should load DiffResults_helpers.jl

struct ReverseDiffLogDensity{L,C} <: ADGradientWrapper
    ℓ::L
    compiledtape::C
end

"""
    ADgradient(:ReverseDiff, ℓ; compile=Val(false), x=nothing)
    ADgradient(Val(:ReverseDiff), ℓ; compile=Val(false), x=nothing)

Gradient using algorithmic/automatic differentiation via ReverseDiff.

If `compile isa Val{true}`, a tape of the log density computation is created upon construction of the gradient function and used in every evaluation of the gradient.
One may provide an example input `x::AbstractVector` of the log density function.
If `x` is `nothing` (the default), the tape is created with input `zeros(dimension(ℓ))`.

By default, no tape is created.

!!! note
    Using a compiled tape can lead to significant performance improvements when the gradient of the log density
    is evaluated multiple times (possibly for different inputs).
    However, if the log density contains branches, use of a compiled tape can lead to silently incorrect results.
"""
function ADgradient(::Val{:ReverseDiff}, ℓ;
                    compile::Union{Val{true},Val{false}}=Val(false), x::Union{Nothing,AbstractVector}=nothing)
    ReverseDiffLogDensity(ℓ, _compiledtape(ℓ, compile, x))
end

_compiledtape(ℓ, compile, x) = nothing
_compiledtape(ℓ, ::Val{true}, ::Nothing) = _compiledtape(ℓ, Val(true), zeros(dimension(ℓ)))
function _compiledtape(ℓ, ::Val{true}, x)
    tape = ReverseDiff.GradientTape(Base.Fix1(logdensity, ℓ), x)
    return ReverseDiff.compile(tape)
end

function Base.show(io::IO, ∇ℓ::ReverseDiffLogDensity)
    print(io, "ReverseDiff AD wrapper for ", ∇ℓ.ℓ, " (")
    if ∇ℓ.compiledtape === nothing
        print(io, "no ")
    end
    print(io, "compiled tape)")
end

function logdensity_and_gradient(∇ℓ::ReverseDiffLogDensity, x::AbstractVector)
    @unpack ℓ, compiledtape = ∇ℓ
    buffer = _diffresults_buffer(x)
    if compiledtape === nothing
        result = ReverseDiff.gradient!(buffer, Base.Fix1(logdensity, ℓ), x)
    else
        result = ReverseDiff.gradient!(buffer, compiledtape, x)
    end
    _diffresults_extract(result)
end
