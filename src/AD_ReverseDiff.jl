import .ReverseDiff

struct ReverseDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

show(io::IO, ℓ::ReverseDiffLogDensity) = print(io, "ReverseDiff AD wrapper for ", ℓ.ℓ)

function ADgradient(::Val{:ReverseDiff}, ℓ::AbstractLogDensityProblem)
    cfg = ReverseDiff.GradientConfig(zeros(dimension(ℓ)))
    ReverseDiffLogDensity(ℓ, cfg)
end

function logdensity(::Type{ValueGradient}, fℓ::ReverseDiffLogDensity, x::RealVector)
    @unpack ℓ, gradientconfig = fℓ
    result = DiffResults.GradientResult(_vectorargument(ℓ)) # allocate a new result
    result = ReverseDiff.gradient!(result, _value_closure(ℓ), x, gradientconfig)
    ValueGradient(DiffResults.value(result), DiffResults.gradient(result))
end
