import .Zygote

struct ZygoteGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
$(SIGNATURES)

Gradient using algorithmic/automatic differentiation via Zygote.

!!! NOTE
    Experimental, bug reports welcome.
"""
ADgradient(::Val{:Zygote}, ℓ) = ZygoteGradientLogDensity(ℓ)

Base.show(io::IO, ∇ℓ::ZygoteGradientLogDensity) = print(io, "Zygote AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient(∇ℓ::ZygoteGradientLogDensity, x::AbstractVector)
    @unpack ℓ = ∇ℓ
    y, back = Zygote.forward(_logdensity_closure(ℓ), x)
    y, first(back(Zygote.sensitivity(y)))
end
