export FluxGradientLogDensity

import Flux

struct FluxGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

show(io::IO, ℓ::FluxGradientLogDensity) = print(io, "Flux AD wrapper for ", ℓ.ℓ)

function logdensity(::Type{ValueGradient}, ∇ℓ::FluxGradientLogDensity, x::RealVector)
    @unpack ℓ = ∇ℓ
    f = x -> logdensity(Value, ℓ, x).value
    @debug f = f(Flux.Tracker.TrackedReal.(x))
    y, back = Flux.Tracker.forward(f, x)
    @debug y = y
    yval = Flux.Tracker.data(y)
    ValueGradient(yval, isfinite(yval) ? first(Flux.Tracker.data.(back(1))) : similar(x))
end
