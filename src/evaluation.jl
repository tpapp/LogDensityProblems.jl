export LogDensityEvaluated, NoDerivative, ForwardDiffAD

struct LogDensityEvaluated{T <: Real,
                           S <: Union{Nothing,AbstractVector{T}}}
    value::T
    gradient::S
    function LogDensityEvaluated(value::T, gradient::S) where
            {T <: Real, S <: AbstractVector{T}}
        @argcheck isfinite(value)
        @argcheck all(isfinite, gradient)
        new{T,S}(value, gradient)
    end
    function LogDensityEvaluated(value::T) where {T <: Real}
        @argcheck isfinite(value)
        new{T,Nothing}(value, nothing)
    end
end

abstract type EvaluationMethod end

struct NoDerivative <: EvaluationMethod end

evaluation_environment(::NoDerivative, f, x) = NoDerivative()

function evaluate(::NoDerivative, f, x)
    v = f(x)::Real
    isfinite(v) ? LogDensityEvaluated(v) : nothing
end

struct ForwardDiffAD{C} <: EvaluationMethod
    chunk::C
end

ForwardDiffAD() = ForwardDiffAD(nothing)

struct ForwardDiffEvalEnv{C,R}
    cfg::C
    result::R
end

function evaluation_environment(evalmethod::ForwardDiffAD{<: ForwardDiff.Chunk}, f, x)
    cfg = ForwardDiff.GradientConfig(f, x, evalmethod.chunk)
    result = DiffResults.GradientResult(x)
    ForwardDiffEvalEnv(cfg, result)
end

evaluation_environment(::ForwardDiffAD{<: Nothing}, f, x) =
    evaluation_environment(ForwardDiffAD(ForwardDiff.Chunk(x)), f, x)

function evaluate(evalenv::ForwardDiffEvalEnv, f, x)
    @unpack cfg, result = evalenv
    result = ForwardDiff.gradient!(result, f, x, cfg)
    v = DiffResults.value(result)::Real
    g = DiffResults.gradient(result)
    isfinite(v) || return nothing
    all(isfinite, g) || return nothing
    LogDensityEvaluated(v, g)
end
