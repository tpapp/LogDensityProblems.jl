#####
##### Gradient AD implementation using ForwardDiff
#####

import .ReverseDiff

struct ReverseDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

"""
$(SIGNATURES)

AD via ReverseDiff.
"""
function ADgradient(::Val{:ReverseDiff}, ℓ)
    z = zeros(dimension(ℓ))
    cfg = ReverseDiff.GradientConfig(z)
    ReverseDiffLogDensity(ℓ, cfg)
end

Base.show(io::IO, ℓ::ReverseDiffLogDensity) = print(io, "ReverseDiff AD wrapper for ", ℓ.ℓ)

function logdensity_and_gradient(fℓ::ReverseDiffLogDensity, x::AbstractVector)
    @unpack ℓ, gradientconfig = fℓ
    buffer = _diffresults_buffer(ℓ, x)
    result = ReverseDiff.gradient!(buffer, x -> logdensity(ℓ, x), x, gradientconfig)
    _diffresults_extract(result)
end
