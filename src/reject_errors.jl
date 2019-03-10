#####
##### Catching errors and treating them as a -∞ log density.
#####

export reject_logdensity, LogDensityRejectErrors

struct LogDensityRejectErrors{E, L} <: LogDensityWrapper
    ℓ::L
end

"""
$(SIGNATURES)

Wrap a logdensity `ℓ` so that errors `<: E` are caught and replaced with a ``-∞`` value.

`E` defaults to `InvalidLogDensityExceptions`.

# Note

Use cautiously, as catching errors can mask errors in your code. The recommended use case is
for catching quirks and corner cases of AD. See also [`stresstest`](@ref) as an alternative
to using this wrapper.

Also, some AD packages don't handle `try` - `catch` blocks, so it is advised to use this as
the outermost wrapper.
"""
LogDensityRejectErrors{E}(ℓ::L) where {E,L} = LogDensityRejectErrors{E,L}(ℓ)

LogDensityRejectErrors(ℓ) = LogDensityRejectErrors{InvalidLogDensityException}(ℓ)

minus_inf_like(::Type{Real}, x) = convert(eltype(x), -Inf)

minus_inf_like(::Type{Value}, x) = Value(minus_inf_like(Real, x))

minus_inf_like(::Type{ValueGradient}, x) = ValueGradient(minus_inf_like(Real, x), similar(x))

minus_inf_like(vgb::ValueGradientBuffer, x) = ValueGradient(minus_inf_like(Real, x), vgb.buffer)

function _logdensity_reject_errors(kind, w::LogDensityRejectErrors{E}, x) where {E}
    try
        logdensity(kind, parent(w), x)
    catch e
        if e isa E
            minus_inf_like(kind, x)
        else
            rethrow(e)
        end
    end
end

logdensity(::Type{Real}, w::LogDensityRejectErrors, x::AbstractVector) =
    _logdensity_reject_errors(Real, w, x)

logdensity(::Type{Value}, w::LogDensityRejectErrors, x::AbstractVector) =
    _logdensity_reject_errors(Value, w, x)

logdensity(::Type{ValueGradient}, w::LogDensityRejectErrors, x::AbstractVector) =
    _logdensity_reject_errors(ValueGradient, w, x)

function logdensity(vgb::ValueGradientBuffer, w::LogDensityRejectErrors, x::AbstractVector)
    _logdensity_reject_errors(vgb, w, x)
end
