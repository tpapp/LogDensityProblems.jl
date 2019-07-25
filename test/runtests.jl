using LogDensityProblems, Test, Distributions, TransformVariables
import LogDensityProblems: capabilities, dimension, logdensity
using LogDensityProblems: logdensity_and_gradient, LogDensityOrder

import ForwardDiff, Flux, ReverseDiff, TransformVariables, Random
using Parameters: @unpack

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
### simple log density for testing
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
end

####
#### AD backends
####

@testset "AD via ForwardDiff" begin
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:ForwardDiff, ℓ)
    @test repr(∇ℓ) == "ForwardDiff AD wrapper for " * repr(ℓ) * ", w/ chunk size 3"
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
    for _ in 1:100
        x = randn(3)
        @test logdensity(∇ℓ, x) ≅ test_logdensity(x)
        @test logdensity_and_gradient(∇ℓ, x) ≅ (test_logdensity(x), test_gradient(x))
    end
end

@testset "chunk heuristics for ForwardDiff" begin
    @test LogDensityProblems.heuristic_chunks(82) == vcat(1:4:81, [82])
end

@testset "benchmark ForwardDiff chunk size" begin
    ℓ = TransformedLogDensity(as(Array, 20), x -> -sum(abs2, x))
    b = LogDensityProblems.benchmark_ForwardDiff_chunks(ℓ)
    @test b isa Vector{Pair{Int,Float64}}
    @test length(b) ≤ 20
end

@testset "AD via ReverseDiff" begin
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:ReverseDiff, ℓ)
    @test repr(∇ℓ) == "ReverseDiff AD wrapper for " * repr(ℓ)
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
    for _ in 1:100
        x = randn(3)
        @test logdensity(∇ℓ, x) ≅ test_logdensity(x)
        @test logdensity_and_gradient(∇ℓ, x) ≅ (test_logdensity(x), test_gradient(x))
    end
end

@testset "AD via Flux" begin
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:Flux, ℓ)
    @test repr(∇ℓ) == "Flux AD wrapper for " * repr(ℓ)
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
    for _ in 1:100
        x = randn(3)
        @test logdensity(∇ℓ, x) ≅ test_logdensity(x)
        @test logdensity_and_gradient(∇ℓ, x) ≅ (test_logdensity(x), test_gradient(x))
    end
end

if VERSION ≥ v"1.1.0"
    # cf https://github.com/FluxML/Zygote.jl/issues/104
    import Zygote

    @testset "AD via Zygote" begin
        # cf https://github.com/FluxML/Zygote.jl/issues/271
        ℓ = TestLogDensity(test_logdensity1)
        ∇ℓ = ADgradient(:Zygote, ℓ)
        @test repr(∇ℓ) == "Zygote AD wrapper for " * repr(ℓ)
        @test dimension(∇ℓ) == 3
        @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
        for _ in 1:100
            x = randn(3)
            @test logdensity(∇ℓ, x) ≅ test_logdensity1(x)
            @test logdensity_and_gradient(∇ℓ, x) ≅ (test_logdensity1(x), test_gradient(x))
        end
    end
end

@testset "ADgradient missing method" begin
    msg = "Don't know how to AD with Foo, consider `import Foo` if there is such a package."
    P = TransformedLogDensity(as(Array, 1), x -> sum(abs2, x))
    @test_logs((:info, msg), @test_throws(MethodError, ADgradient(:Foo, P)))
end

####
#### transformed Bayesian problem
####

@testset "transformed Bayesian problem" begin
    t = as((y = asℝ₊, ))
    d = LogNormal(1.0, 2.0)
    logposterior = ((x, ), ) -> logpdf(d, x)

    # a Bayesian problem
    p = TransformedLogDensity(t, logposterior)
    @test repr(p) == "TransformedLogDensity of dimension 1"
    @test dimension(p) == 1
    @test p.transformation ≡ t
    @test capabilities(p) == LogDensityOrder(0)

    # gradient of a problem
    ∇p = ADgradient(:ForwardDiff, p)
    @test dimension(∇p) == 1
    @test parent(∇p).transformation ≡ t

    for _ in 1:100
        x = random_arg(t)
        θ, lj = transform_and_logjac(t, x)
        px = logdensity(p, x)
        @test logpdf(d, θ.y) + lj ≈ (px::Real)
        px2, ∇px = logdensity_and_gradient(∇p, x)
        @test px2 == px
        @test ∇px ≈ [ForwardDiff.derivative(x -> logpdf(d, exp(x)) + x, x[1])]
    end
end

@testset "-∞ log densities" begin
    t = as(Array, 2)
    validx = x -> all(x .> 0)
    p = TransformedLogDensity(t, x -> validx(x) ?  sum(abs2, x)/2 : -Inf)
    ∇p = ADgradient(:ForwardDiff, p)

    @test dimension(p) == dimension(∇p) == TransformVariables.dimension(t)
    @test p.transformation ≡ parent(∇p).transformation ≡ t

    for _ in 1:100
        x = random_arg(t)
        px = logdensity(∇p, x)
        px_∇px = logdensity_and_gradient(∇p, x)
        @test px isa Real
        @test px_∇px isa Tuple{Real,Any}
        @test first(px_∇px) ≡ px
        if validx(x)
            @test px ≈ sum(abs2, x)/2
            @test last(px_∇px) ≈ x
        else
            @test isinf(px)
        end
    end
end

@testset "stresstest" begin
    @info "stress testing"
    ℓ = TestLogDensity(x -> all(x .< 0) ? error("invalid") : -sum(abs2, x))
    failures = LogDensityProblems.stresstest(logdensity, ℓ; N = 500)
    @test 50 ≤ length(failures) ≤ 100
    @test all(x -> all(x .< 0), failures)
end
