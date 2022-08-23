import .Zygote

struct ZygoteGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
    ADgradient(:Zygote, ℓ)
    ADgradient(Val(:Zygote), ℓ)

Gradient using algorithmic/automatic differentiation via Zygote.
"""
ADgradient(::Val{:Zygote}, ℓ) = ZygoteGradientLogDensity(ℓ)

Base.show(io::IO, ∇ℓ::ZygoteGradientLogDensity) = print(io, "Zygote AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient_of(∇ℓ::ZygoteGradientLogDensity, x::AbstractVector)
    @unpack ℓ = ∇ℓ
    y, back = Zygote.pullback(logdensityof(ℓ), x)
    y, first(back(Zygote.sensitivity(y)))
end
