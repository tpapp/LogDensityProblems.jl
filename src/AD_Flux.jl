#####
##### Gradient AD implementation using Flux
#####

import .Flux

struct FluxGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
$(SIGNATURES)

Gradient using algorithmic/automatic differentiation via Flux.
"""
ADgradient(::Val{:Flux}, ℓ) = FluxGradientLogDensity(ℓ)

Base.show(io::IO, ∇ℓ::FluxGradientLogDensity) = print(io, "Flux AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient(∇ℓ::FluxGradientLogDensity, x::AbstractVector)
    @unpack ℓ = ∇ℓ
    y, back = Flux.Tracker.forward(x -> logdensity(ℓ, x), x)
    yval = Flux.Tracker.data(y)
    yval, first(Flux.Tracker.data.(back(1)))
end
