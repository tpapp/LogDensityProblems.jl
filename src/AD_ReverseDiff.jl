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

struct ReverseDiffTapeLogDensity{L, T} <: ADGradientWrapper
    ℓ::L
    compiled_tape::T
end

function Base.show(io::IO, ℓ::ReverseDiffTapeLogDensity)
    print(io, "ReverseDiff AD wrapper (compiled tape) for ", ℓ.ℓ)
end

function logdensity(vgb::ValueGradientBuffer, fℓ::ReverseDiffTapeLogDensity,
                    x::AbstractVector)
    @unpack compiled_tape = fℓ
    result = ReverseDiff.gradient!(DiffResults.MutableDiffResult(vgb), compiled_tape, x)
    ValueGradient(result)
end

"""
$(SIGNATURES)

AD via ReverseDiff. When `tape`, record and compile a tape; usual caveats apply, see the
ReverseDiff documentation.
"""
function ADgradient(::Val{:ReverseDiff}, ℓ; tape = false)
    z = zeros(dimension(ℓ))
    if tape
        f = _logdensity_closure(ℓ)
        tape = ReverseDiff.GradientTape(f, z)
        compiled_tape = ReverseDiff.compile(tape)
        ReverseDiffTapeLogDensity(ℓ, compiled_tape)
    else
        cfg = ReverseDiff.GradientConfig(z)
        ReverseDiffLogDensity(ℓ, cfg)
    end
end
