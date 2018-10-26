export FluxGradientLogDensity

import Flux

struct FluxGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

show(io::IO, ∇ℓ::FluxGradientLogDensity) = print(io, "Flux AD wrapper for ", ∇ℓ.ℓ)

function logdensity(::Type{ValueGradient}, ∇ℓ::FluxGradientLogDensity, x::RealVector)
    @unpack ℓ = ∇ℓ
    y, back = Flux.Tracker.forward(_value_closure(ℓ), x)
    yval = Flux.Tracker.data(y)
    ValueGradient(yval, isfinite(yval) ? first(Flux.Tracker.data.(back(1))) : similar(x))
end
