import .ReverseDiff, .DiffResults

struct ReverseDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

Base.show(io::IO, ℓ::ReverseDiffLogDensity) = print(io, "ReverseDiff AD wrapper for ", ℓ.ℓ)

function logdensity(vgb::ValueGradientBuffer, fℓ::ReverseDiffLogDensity, x::AbstractVector)
    @unpack ℓ, gradientconfig = fℓ
    result = ReverseDiff.gradient!(DiffResults.MutableDiffResult(vgb),
                                   x -> logdensity(Real, ℓ, x), x, gradientconfig)
    ValueGradient(result)
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
