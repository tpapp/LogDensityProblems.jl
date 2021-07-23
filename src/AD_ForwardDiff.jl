#####
##### Gradient AD implementation using ForwardDiff
#####

import .ForwardDiff

struct ForwardDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

function Base.show(io::IO, ℓ::ForwardDiffLogDensity)
    print(io, "ForwardDiff AD wrapper for ", ℓ.ℓ,
          ", w/ chunk size ", length(ℓ.gradientconfig.seeds))
end

_default_chunk(ℓ) = ForwardDiff.Chunk(dimension(ℓ))

# defined to make the tag match
_logdensity_closure(ℓ) = x -> logdensity(ℓ, x)

function _default_gradientconfig(ℓ, chunk::ForwardDiff.Chunk)
    ForwardDiff.GradientConfig(_logdensity_closure(ℓ), zeros(dimension(ℓ)), chunk)
end

function _default_gradientconfig(ℓ, chunk::Integer)
    _default_gradientconfig(ℓ, ForwardDiff.Chunk(chunk))
end

"""
    ADgradient(:ForwardDiff, ℓ; chunk, gradientconfig)
    ADgradient(Val(:ForwardDiff), ℓ; chunk, gradientconfig)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keywords are passed on to `ForwardDiff.GradientConfig` to customize the setup. In
particular, chunk size can be set with a `chunk` keyword argument (accepting an integer or a
`ForwardDiff.Chunk`).
"""
function ADgradient(::Val{:ForwardDiff}, ℓ;
                    chunk = _default_chunk(ℓ),
                    gradientconfig = _default_gradientconfig(ℓ, chunk))
    ForwardDiffLogDensity(ℓ, gradientconfig)
end

function logdensity_and_gradient(fℓ::ForwardDiffLogDensity, x::AbstractVector)
    @unpack ℓ, gradientconfig = fℓ
    buffer = _diffresults_buffer(ℓ, x)
    result = ForwardDiff.gradient!(buffer, _logdensity_closure(ℓ), x, gradientconfig)
    _diffresults_extract(result)
end
