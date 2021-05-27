#####
##### Gradient AD implementation using ReverseDiff
#####

import .ReverseDiff

struct ReverseDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
	compiledtape::C
end

"""
    ADgradient(:ReverseDiff, ℓ)
    ADgradient(Val(:ReverseDiff), ℓ)

Gradient using algorithmic/automatic differentiation via ReverseDiff.
"""
ADgradient(::Val{:ReverseDiff}, ℓ) = begin
	f = _logdensity_closure(ℓ)
	x = rand(dimension(ℓ)) #init random parameters
	tape = ReverseDiff.GradientTape(f, x)
	compiledtape = ReverseDiff.compile(tape)
	ReverseDiffLogDensity(ℓ, compiledtape)
end

Base.show(io::IO, ∇ℓ::ReverseDiffLogDensity) = print(io, "ReverseDiff AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient(∇ℓ::ReverseDiffLogDensity, x::AbstractVector{T}) where {T}
    @unpack ℓ, compiledtape = ∇ℓ

	buffer = _diffresults_buffer(ℓ, x)
    result = ReverseDiff.gradient!(buffer, compiledtape, x)
    _diffresults_extract(result)
end
