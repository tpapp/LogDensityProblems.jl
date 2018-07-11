using LogDensityFramework
using Test
using Parameters: @unpack
# using StatsBase: mean_and_var
# using TransformVariables
# using Random: GLOBAL_RNG

using LogDensityFramework
import LogDensityFramework: logdensity
using LogDensityFramework: evaluation_environment, evaluate


# test utilities

import Base: isapprox, ==

function isapprox(a::LogDensityEvaluated{T}, b::LogDensityEvaluated{S};
                  atol = 0, rtol = Base.rtoldefault(promote_type(T,S))) where {T,S}
    isapprox(a.value, b.value; atol = atol, rtol = rtol) || return false
    a.gradient ≡ b.gradient ≡ nothing && return true
    isapprox(a.gradient, b.gradient; atol = atol, rtol = rtol)
end

(==)(a::LogDensityEvaluated, b::LogDensityEvaluated) =
    isapprox(a, b; atol = 0, rtol = 0)


# evaluation

@testset "log density evaluated" begin
    @test_throws ArgumentError LogDensityEvaluated(-Inf)
    @test_throws ArgumentError LogDensityEvaluated(2.0, [NaN])
    @test_throws MethodError LogDensityEvaluated(2.0, 1.0)
    @test_throws MethodError LogDensityEvaluated("a fish")
    d = LogDensityEvaluated(2.0)
    @test d.value ≡ 2.0
    @test d.gradient ≡ nothing
    d = LogDensityEvaluated(2.0, [3.0, 4.0])
    @test d.value == 2.0
    @test d.gradient == [3.0, 4.0]
end

@testset "evaluation" begin
    fexp = x -> exp(3*x[1])
    env1 = evaluation_environment(NoDerivative(), fexp, [0.0])
    @test env1 ≡ NoDerivative()
    @test evaluate(env1, fexp, [1.0]) ≈ LogDensityEvaluated(exp(3))
    env2 = evaluation_environment(ForwardDiffAD(), fexp, [0.0])
    @test evaluation_environment(NoDerivative(), fexp, [0.0]) ≡ NoDerivative()
    @test evaluate(env2, fexp, [1.0]) ≈ LogDensityEvaluated(exp(3), [3*exp(3)])
end
