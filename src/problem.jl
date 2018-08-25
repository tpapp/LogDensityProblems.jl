export Problem, LogDensityProblem, Trajectory, logdensity


# problem interface

module Problem

using Random: AbstractRNG
using DocStringExtensions: SIGNATURES

"""
    transformation(problem)

Return a `TransformVariables.TransformReals` that encodes the transformation of
variables.
"""
function transformation end

"""
    $SIGNATURES

If `true`, the problem

1. *may* use common random numbers and have a `make_common_random` method,
2. has `logdensity` method with `rng` and `common_random`.

If `false`, the problem has a `logdensity` method with two arguments.
"""
isstochastic(problem) = false

"""
    $SIGNATURES
"""
common_random(problem, rng::AbstractRNG) = nothing


"""
```julia
logdensity(problem, parameters)
logdensity(problem, parameters, rng::AbstractRNG, common_random)
```

Log density function for `problem` at `parameters`.

Returns `nothing` for `0` density or infeasible parameters, a finite real number
otherwise.
"""
function logdensity end

end



# log density

"""
    $(TYPEDEF)


"""
struct LogDensityProblem{S, P, T <: TransformReals, E <: EvaluationMethod}
    problem::P
    transformation::T
    evalmethod::E
    function LogDensityProblem(problem::P, evalmethod::E) where {P,E}
        S = Problem.isstochastic(problem)::Bool
        transformation = Problem.transformation(problem)
        new{S,P,typeof(transformation),E}(problem, transformation, evalmethod)
    end
end

length(ℓ::LogDensityProblem) = length(ℓ.transformation)

function local_logdensity(ℓ::LogDensityProblem{false})
    @unpack problem, transformation = ℓ
    x -> transform_logdensity(transformation,
                              y -> Problem.logdensity(problem, y), x)
end

function local_logdensity(ℓ::LogDensityProblem{true}, rng::AbstractRNG,
                          common_random)
    @unpack problem, transformation = ℓ
    x -> transform_logdensity(transformation,
                              y -> Problem.logdensity(problem, y, rng, common_random),
                              x)
end

struct Trajectory{L, E, F, R <: Union{Nothing, AbstractRNG}, C}
    ℓ::L
    evalenv::E
    f::F
    rng::R
    common_random::C
end

function Trajectory(ℓ::LogDensityProblem{false})
    f = local_logdensity(ℓ)
    evalenv = evaluation_environment(ℓ.evalmethod, f, zeros(length(ℓ)))
    Trajectory(ℓ, evalenv, f, nothing, nothing)
end

function Trajectory(ℓ::LogDensityProblem{true}, rng::AbstractRNG)
    common_random = Problem.common_random(ℓ.problem, rng)
    f = local_logdensity(ℓ, rng, common_random)
    evalenv = evaluation_environment(ℓ.evalmethod, f, zeros(length(ℓ)))
    Trajectory(ℓ, evalenv, f, rng, common_random)
end

length(τ::Trajectory) = length(τ.ℓ)

update_trajectory(τ::Trajectory{<:LogDensityProblem{false},Nothing}) = τ

"""
    update_trajectory(trajectory)

Update a trajectory with new common random numbers.
"""
function update_trajectory(τ::Trajectory{<:LogDensityProblem{true}, <:AbstractRNG})
    @unpack ℓ, evalenv, f, rng = trajectory
    Trajectory(ℓ, evalenv, f, rng, Problem.common_random(rng, problem))
end

function logdensity(τ::Trajectory, x::AbstractVector{<: Real})
    @unpack evalenv, f = τ
    evaluate(evalenv, f, x)
end
