import .Zygote

struct ZygoteGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
$(SIGNATURES)

Gradient using algorithmic/automatic differentiation via Zygote.

!!! NOTE
    Experimental, use latest `Zygote#master` and `IRTools#master`.
"""
ADgradient(::Val{:Zygote}, ℓ) = ZygoteGradientLogDensity(ℓ)

Base.show(io::IO, ∇ℓ::ZygoteGradientLogDensity) = print(io, "Zygote AD wrapper for ", ∇ℓ.ℓ)

function logdensity(::Type{ValueGradient}, ∇ℓ::ZygoteGradientLogDensity, x::AbstractVector)
    @unpack ℓ = ∇ℓ
    y, back = Zygote.forward(_logdensity_closure(ℓ), x)
    gradient = isfinite(y) ? back(Int8(1))[1] : zeros(typeof(y), length(y))
    ValueGradient(y, gradient)
end

function logdensity(::ValueGradientBuffer, ∇ℓ::ZygoteGradientLogDensity, x::AbstractVector)
    # NOTE this implementation ignores the buffer
    logdensity(ValueGradient, ∇ℓ, x)
end
