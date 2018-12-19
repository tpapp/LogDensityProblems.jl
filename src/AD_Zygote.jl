import .Zygote

struct ZygoteGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
$(SIGNATURES)

Gradient using algorithmic/automatic differentiation via Zygote.
"""
ADgradient(::Val{:Zygote}, ℓ::AbstractLogDensityProblem) = ZygoteGradientLogDensity(ℓ)

show(io::IO, ∇ℓ::ZygoteGradientLogDensity) = print(io, "Zygote AD wrapper for ", ∇ℓ.ℓ)

function logdensity(::Type{ValueGradient}, ∇ℓ::ZygoteGradientLogDensity, x::RealVector)
    @unpack ℓ = ∇ℓ
    y, back = Zygote.forward(_value_closure(ℓ), x)
    isfinite(y) || return ValueGradient(y, zeros(typeof(y), length(x)))
    grad = back(Int8(1))
    ValueGradient(y, grad)
end
