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

function logdensity(::Type{ValueGradient}, ∇ℓ::FluxGradientLogDensity, x::AbstractVector)
    @unpack ℓ = ∇ℓ
    y, back = Flux.Tracker.forward(_logdensity_closure(ℓ), x)
    yval = Flux.Tracker.data(y)
    ValueGradient(yval, isfinite(yval) ? first(Flux.Tracker.data.(back(1))) : similar(x))
end

function logdensity(::ValueGradientBuffer, ∇ℓ::FluxGradientLogDensity, x::AbstractVector)
    # NOTE this implementation ignores the buffer
    logdensity(ValueGradient, ∇ℓ, x)
end
