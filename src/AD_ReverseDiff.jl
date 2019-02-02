import .ReverseDiff, .DiffResults

struct ReverseDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

show(io::IO, ℓ::ReverseDiffLogDensity) = print(io, "ReverseDiff AD wrapper for ", ℓ.ℓ)

function logdensity(::Type{ValueGradient}, fℓ::ReverseDiffLogDensity, x::RealVector)
    @unpack ℓ, gradientconfig = fℓ
    result = DiffResults.GradientResult(_vectorargument(ℓ)) # allocate a new result
    result = ReverseDiff.gradient!(result, _value_closure(ℓ), x, gradientconfig)
    ValueGradient(DiffResults.value(result), DiffResults.gradient(result))
end

struct ReverseDiffTapeLogDensity{L, R, T} <: ADGradientWrapper
    ℓ::L
    result_buffer::R
    compiled_tape::T
end

show(io::IO, ℓ::ReverseDiffTapeLogDensity) = print(io, "ReverseDiff AD wrapper (compiled tape) for ", ℓ.ℓ)

function logdensity(::Type{ValueGradient}, fℓ::ReverseDiffTapeLogDensity, x::RealVector)
    @unpack result_buffer, compiled_tape = fℓ
    result = ReverseDiff.gradient!(result_buffer, compiled_tape, x)
    v = DiffResults.value(result)
    ValueGradient(isfinite(v) ? v : oftype(v, -Inf), DiffResults.gradient(result))
end

"""
$(SIGNATURES)

AD via ReverseDiff. When `tape`, record and compile a tape; usual caveats apply, see the
ReverseDiff documentation.
"""
function ADgradient(::Val{:ReverseDiff}, ℓ; tape = false)
    z = zeros(dimension(ℓ))
    if tape
        result_buffer = DiffResults.GradientResult(similar(z))
        f = _value_closure(ℓ)
        tape = ReverseDiff.GradientTape(f, z)
        compiled_tape = ReverseDiff.compile(tape)
        ReverseDiffTapeLogDensity(ℓ, result_buffer, compiled_tape)
    else
        cfg = ReverseDiff.GradientConfig(z)
        ReverseDiffLogDensity(ℓ, cfg)
    end
end
