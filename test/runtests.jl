using LogDensityProblems, Test, Random
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient,
    logdensity_gradient_and_hessian
using LogDensityProblems: logdensity_and_gradient, LogDensityOrder

####
#### test setup and utilities
####

###
### reproducible randomness
###

Random.seed!(1)

###
### comparisons (for testing)
###

"""
    a ≅ b

Compare log denfields and types, for unit testing.
"""
≅(::Any, ::Any, atol = 0) = false

function ≅(a::Real, b::Real, atol = 0)
    if isnan(a)
        isnan(b)
    elseif isinf(a)
        a == b
    else
        abs(a - b) ≤ atol
    end
end

function ≅(a::Tuple{Real,Any}, b::Tuple{Real,Any}, atol = 0)
    ≅(first(a), first(b), atol) || return false
    !isfinite(first(a)) || isapprox(last(a), last(b); atol = atol, rtol = 0)
end

@testset "comparisons for unit testing" begin
    @test 1 ≅ 1
    @test !(1 ≅ 2)
    @test Inf ≅ Inf
    @test (1, [1, 2]) ≅ (1, [1, 2])
    @test !((1, [1, 2]) ≅ (1, [1, 3]))
    @test !((3, [1, 2]) ≅ (1, [1, 2]))
    @test (-Inf, [1, 2]) ≅ (-Inf, [1, 2])
    @test (-Inf, [1, 2]) ≅ (-Inf, [1, 3])
    @test (-Inf, [1, 2]) ≅ (-Inf, nothing)
end

###
### a simple log density for testing
###

struct TestLogDensity{F}
    ℓ::F
end
logdensity(ℓ::TestLogDensity, x) = ℓ.ℓ(x)
dimension(::TestLogDensity) = 3
test_logdensity1(x) = -2*abs2(x[1]) - 3*abs2(x[2]) - 5*abs2(x[3])
test_logdensity(x) = any(x .< 0) ? -Inf : test_logdensity1(x)
test_gradient(x) = x .* [-4, -6, -10]
TestLogDensity() = TestLogDensity(test_logdensity) # default: -Inf for negative input
Base.show(io::IO, ::TestLogDensity) = print(io, "TestLogDensity")

####
#### traits
####

@test capabilities("a fish") ≡ nothing

@testset "LogDensityOrder" begin
    @test LogDensityOrder(1) == LogDensityOrder(1)
    @test_throws ArgumentError LogDensityOrder(-1)
    @test LogDensityOrder(2) ≥ LogDensityOrder(1)
    @test !(LogDensityOrder(1) > LogDensityOrder(1))
end

####
#### utilities
####


@testset "stresstest" begin
    @info "stress testing"
    ℓ = TestLogDensity(x -> all(x .< 0) ? error("invalid") : -sum(abs2, x))
    failures = LogDensityProblems.stresstest(logdensity, ℓ; N = 500)
    @test 50 ≤ length(failures) ≤ 100
    @test all(x -> all(x .< 0), failures)
end

####
#### public API
####

@testset "public API" begin
    # NOTE remove this once we require Julia v1.11 and use public
    if isdefined(Base, :ispublic)
        @test Base.ispublic(LogDensityProblems, :capabilities)
        @test Base.ispublic(LogDensityProblems, :LogDensityOrder)
        @test Base.ispublic(LogDensityProblems, :dimension)
        @test Base.ispublic(LogDensityProblems, :logdensity)
        @test Base.ispublic(LogDensityProblems, :logdensity_and_gradient)
        @test Base.ispublic(LogDensityProblems, :logdensity_gradient_and_hessian)
        @test Base.ispublic(LogDensityProblems, :stresstest)
        @test Base.ispublic(LogDensityProblems, :converting_logdensity)
    end
end

####
#### converting logdensity
####

struct BadLogDensity end
dimension(::BadLogDensity) = 1
capabilities(::BadLogDensity) = LogDensityOrder(2)
_bad_x(x) = (_x = only(x); _x > 0 ? Float64(_x) : _x) # introduce type instability
function logdensity(::BadLogDensity, x::Vector{Float32}) # deliberate restriction
     -_bad_x(x)^2 / 2
end
function logdensity_and_gradient(::BadLogDensity, x::Vector{Float32})
    _x = _bad_x(x)
     -_x^2 / 2, [-_x]
end
function logdensity_gradient_and_hessian(::BadLogDensity, x::Vector{Float32})
    _x = _bad_x(x)
     -_x^2 / 2, [-_x], [-one(_x)]
end

@testset "converting logdensity" begin
    bad = BadLogDensity()
    ℓ = LogDensityProblems.converting_logdensity(bad;
                                                 input = Vector{Float32},
                                                 logdensity = Float64,
                                                 gradient = Vector{Float64})
    @test dimension(ℓ) == dimension(bad)
    @test capabilities(ℓ) == capabilities(bad)
    x = [0.9]                   # no such method for the parent
    xF32 = Float32.(x)
    @test @inferred(logdensity(ℓ, x)) == logdensity(bad, xF32)
    @test @inferred(logdensity_and_gradient(ℓ, x)) == logdensity_and_gradient(bad, xF32)
    @test eltype(logdensity_gradient_and_hessian(ℓ, .-x)[3]) ≡ Float32 # we do not touch this
end
