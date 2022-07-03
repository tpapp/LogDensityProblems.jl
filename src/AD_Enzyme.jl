import .Enzyme

struct EnzymeGradientLogDensity{L,M<:Union{Enzyme.ForwardMode,Enzyme.ReverseMode},S} <: ADGradientWrapper
    ℓ::L
    mode::M
    shadow::S # only used in forward mode
end

"""
    ADgradient(:Enzyme, ℓ; kwargs...)
    ADgradient(Val(:Enzyme), ℓ; kwargs...)

Gradient using algorithmic/automatic differentiation via Enzyme.

# Keyword arguments

- `mode::Enzyme.Mode`: Differentiation mode (default: `Enzyme.Reverse`).
  Currently only `Enzyme.Reverse` and `Enzyme.Forward` are supported.

- `shadow`: Collection of one-hot vectors for each entry of the inputs `x` to the log density
  `ℓ`, or `nothing` (default: `nothing`). This keyword argument is only used in forward
  mode. By default, it will be recomputed in every call of `logdensity_and_gradient(ℓ, x)`.
  For performance reasons it is recommended to compute it only once when calling `ADgradient`.
  The one-hot vectors can be constructed, e.g., with `Enzyme.onehot(x)`.
"""
function ADgradient(::Val{:Enzyme}, ℓ; mode::Enzyme.Mode = Enzyme.Reverse, shadow = nothing)
    mode isa Union{Enzyme.ForwardMode,Enzyme.ReverseMode} ||
        throw(ArgumentError("currently automatic differentiation via Enzyme only supports " *
                            "`Enzyme.Forward` and `Enzyme.Reverse` modes"))
    if mode isa Enzyme.ReverseMode && shadow !== nothing
        @info "keyword argument `shadow` is ignored in reverse mode"
        shadow = nothing
    end
    return EnzymeGradientLogDensity(ℓ, mode, shadow)
end

function Base.show(io::IO, ∇ℓ::EnzymeGradientLogDensity)
    print(io, "Enzyme AD wrapper for ", ∇ℓ.ℓ, " with ",
          ∇ℓ.mode isa Enzyme.ForwardMode ? "forward" : "reverse", " mode")
end

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ForwardMode},
                                 x::AbstractVector)
    @unpack ℓ, mode, shadow = ∇ℓ
    _shadow = shadow === nothing ? Enzyme.onehot(x) : shadow
    y, ∂ℓ_∂x = Enzyme.autodiff(mode, Base.Fix1(logdensity, ℓ), Enzyme.BatchDuplicated,
                               Enzyme.BatchDuplicated(x, _shadow))
    return y, collect(∂ℓ_∂x)
end

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ReverseMode},
                                 x::AbstractVector)
    @unpack ℓ, mode = ∇ℓ
    # Currently it is not possible to retrieve the primal together with the derivatives.
    # Ref: https://github.com/EnzymeAD/Enzyme.jl/issues/107
    y = logdensity(ℓ, x)
    ∂ℓ_∂x = zero(x)
    Enzyme.autodiff(mode, Base.Fix1(logdensity, ℓ), Enzyme.Active,
                    Enzyme.Duplicated(x, ∂ℓ_∂x))
    y, ∂ℓ_∂x
end
