using LogDensityProblems: Value, ValueGradient, finite_or_nothing
using Test
using Parameters: @unpack
using DocStringExtensions: SIGNATURES

"""
    a ≅ b

Compare fields and types (strictly), for unit testing.
"""
≅(::Any, ::Any) = false
≅(::Nothing, ::Nothing) = true
≅(a::Value{T}, b::Value{T}) where {T} = a.value == b.value
≅(a::ValueGradient{T,V}, b::ValueGradient{T,V}) where {T,V} =
    a.value == b.value && a.gradient == b.gradient

@testset "Value constructor" begin
    @test eltype(Value(1.0)) ≡ Float64
    @test Value(1) ≅ Value(1.0)
    @test_throws ArgumentError Value(-Inf)
    @test_throws ArgumentError Value(NaN)
    @test_throws MethodError Value(:something)
    @test finite_or_nothing(Value, 1.0) ≅ Value(1)
    @test finite_or_nothing(Value, -Inf) ≅ nothing
    @test_throws ArgumentError finite_or_nothing(Value, Inf)
end

@testset "ValueGradient constructor" begin
    @test eltype(ValueGradient(1.0, [2.0])) ≡ Float64
    @test ValueGradient(1, [2]) ≅ ValueGradient(1.0, [2.0])
    @test_throws ArgumentError ValueGradient(-Inf, [1.0])
    @test_throws ArgumentError ValueGradient(2.0, [Inf])
end

# 
# # test utilities

# import Base: isapprox, ==

# function isapprox(a::LogDensityEvaluated{T}, b::LogDensityEvaluated{S};
#                   atol = 0, rtol = Base.rtoldefault(promote_type(T,S))) where {T,S}
#     isapprox(a.value, b.value; atol = atol, rtol = rtol) || return false
#     a.gradient ≡ b.gradient ≡ nothing && return true
#     isapprox(a.gradient, b.gradient; atol = atol, rtol = rtol)
# end

# (==)(a::LogDensityEvaluated, b::LogDensityEvaluated) =
#     isapprox(a, b; atol = 0, rtol = 0)

# 
# # evaluation

# @testset "log density evaluated" begin
#     @test_throws ArgumentError LogDensityEvaluated(-Inf)
#     @test_throws ArgumentError LogDensityEvaluated(2.0, [NaN])
#     @test_throws MethodError LogDensityEvaluated(2.0, 1.0)
#     @test_throws MethodError LogDensityEvaluated("a fish")
#     d = LogDensityEvaluated(2.0)
#     @test d.value ≡ 2.0
#     @test d.gradient ≡ nothing
#     d = LogDensityEvaluated(2.0, [3.0, 4.0])
#     @test d.value == 2.0
#     @test d.gradient == [3.0, 4.0]
# end

# @testset "evaluation" begin
#     fexp = x -> exp(3*x[1])
#     env1 = evaluation_environment(NoDerivative(), fexp, [0.0])
#     @test env1 ≡ NoDerivative()
#     @test evaluate(env1, fexp, [1.0]) ≈ LogDensityEvaluated(exp(3))
#     env2 = evaluation_environment(ForwardAD(), fexp, [0.0])
#     @test evaluation_environment(NoDerivative(), fexp, [0.0]) ≡ NoDerivative()
#     @test evaluate(env2, fexp, [1.0]) ≈ LogDensityEvaluated(exp(3), [3*exp(3)])
# end


# 
# # simple normal

# struct IIDNormal{T}
#     # "sample size"
#     N::Int
#     # "sample mean"
#     x̄::T
#     # "sample variance "
#     σ²::T
# end

# IIDNormal(x::AbstractVector) = IIDNormal(length(x),
#                                          mean_and_var(x; corrected = false)...)

# function Problem.logdensity(problem::IIDNormal, parameters::NamedTuple{(:μ,:v)})
#     @unpack N, x̄, σ² = problem
#     @unpack μ, v = parameters
#     -N/2*(log(v) + (abs2(x̄-μ) + σ²)/v)
# end

# Problem.transformation(::IIDNormal) = to_tuple((μ = to_ℝ, v = to_ℝ₊))

# @testset "simple normal problem and trajectory" begin
#     x = randn(20) * 3 .+ 1
#     p = IIDNormal(x)

#     # no derivatives
#     ℓ = LogDensityProblem(p, NoDerivative())
#     τ = Trajectory(ℓ)
#     @test length(τ) == length(ℓ) == 2
#     n = length(ℓ)
#     x = randn(n)
#     d = logdensity(τ, x)
#     t = Problem.transformation(p)
#     f = x -> transform_logdensity(t, y -> Problem.logdensity(p, y), x)
#     @test d.value == f(x)
#     @test d.gradient ≡ nothing

#     # Forward AD
#     ℓ2 = LogDensityProblem(p, ForwardAD())
#     τ2 = Trajectory(ℓ2)
#     @test length(τ2) == length(ℓ2) == 2
#     d2 = logdensity(τ2, x)
#     @test d2.value == f(x)
#     @test d2.gradient == ForwardDiff.gradient(f, x)
# end
