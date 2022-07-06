#####
##### Gradient AD implementation using ReverseDiff
#####

import .ReverseDiff

struct ReverseDiffLogDensity{L,C} <: ADGradientWrapper
    ℓ::L
    compiledtape::C
end

"""
    ADgradient(:ReverseDiff, ℓ)
    ADgradient(Val(:ReverseDiff), ℓ)

Gradient using algorithmic/automatic differentiation via ReverseDiff.
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
    buffer = _diffresults_buffer(ℓ, x)
    if compiledtape === nothing
        result = ReverseDiff.gradient!(buffer, Base.Fix1(logdensity, ℓ), x)
    else
        result = ReverseDiff.gradient!(buffer, compiledtape, x)
    end
    _diffresults_extract(result)
end
