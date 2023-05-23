using LogDensityProblems, Test, Random
import LogDensityProblems: capabilities, dimension, logdensity
using LogDensityProblems: logdensity_and_gradient, LogDensityOrder, is_valid_result

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
#### valid results
####

@testset "valid results" begin
    @test is_valid_result(1.0)
    @test is_valid_result(-Inf)
    @test !is_valid_result(Inf)
    @test !is_valid_result(NaN)
    @test !is_valid_result(missing)
    @test !is_valid_result("a fish")

    @test is_valid_result(1.0, [2.0, 3.0])              # all finite
    @test !is_valid_result(Inf, [2.0, 3.0])              # invalid
    @test !is_valid_result(NaN, [2.0, 3.0])              # invalid
    @test is_valid_result(-Inf, [NaN, Inf])             # gradient ignored
    @test is_valid_result(-Inf, "wrong type")          # wrong type but ignored
    @test is_valid_result(-Inf, ["wrong element", 1.0]) # gradient ignored

    @test is_valid_result(1.0, [2.0, 3], [4.0 5; 6 7]) # non-symmetric but OK
    @test !is_valid_result(Inf, [2.0, 3], [4.0 5; 6 7]) # invalid
    @test !is_valid_result(NaN, [2.0, 3], [4.0 5; 6 7]) # invalid
    @test !is_valid_result(:a_fish, [2.0, 3], [4.0 5; 6 7]) # invalid
    @test is_valid_result(-Inf, [2.0, 3], [NaN 5; 6 7]) # Hessian ignored
    @test is_valid_result(-Inf, "bad to the", :bone)
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
