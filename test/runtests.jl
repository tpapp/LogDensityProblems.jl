using LogDensityProblems, Test, Distributions, TransformVariables
using LogDensityProblems: Value, ValueGradient, ValueGradientBuffer

import ForwardDiff, Flux, ReverseDiff
using Parameters: @unpack
using TransformVariables
import Random

####
#### test setup and utilities
####

Random.seed!(1)

"""
    a ≅ b

Compare fields and types (strictly), for unit testing. For less strict comparison, use `≈`.
"""
≅(::Any, ::Any) = false
≅(a::Value{T}, b::Value{T}) where {T} = a.value == b.value
≅(a::ValueGradient{T,V}, b::ValueGradient{T,V}) where {T,V} =
    a.value == b.value && (a.value == -Inf || a.gradient == b.gradient)

_wide_tol(a, b) = max(√eps(a.value), √eps(b.value))

function Base.isapprox(a::Value, b::Value; atol = _wide_tol(a, b), rtol = atol)
    isapprox(a.value, b.value; atol = atol, rtol = rtol)
end

function Base.isapprox(a::ValueGradient, b::ValueGradient; atol = _wide_tol(a, b), rtol = atol)
    isapprox(a.value, b.value; atol = atol, rtol = rtol) &&
        (a.value == -Inf || isapprox(a.gradient, b.gradient; atol = √eps(), rtol = atol))
end

####
#### result types
####

@testset "Value constructor" begin
    @test eltype(Value(1.0)) ≡ Float64
    @test_throws InvalidLogDensityException(0, Inf) Value(Inf)
    @test_throws InvalidLogDensityException Value(NaN) # FIXME more specific test
    @test isfinite(Value(1.0))
    @test !isinf(Value(1.0))
    @test !isfinite(Value(-Inf))
    @test isinf(Value(-Inf))
    @test_throws MethodError Value(:something)
end

@testset "ValueGradient constructor" begin
    @test eltype(ValueGradient(1.0, [2.0])) ≡ Float64
    @test_throws InvalidLogDensityException(0, Inf) ValueGradient(Inf, [1.0])
    @test_throws InvalidLogDensityException(1, Inf) ValueGradient(2.0, [Inf])
    @test !isfinite(ValueGradient(-Inf, [12.0]))
    @test isinf(ValueGradient(-Inf, [12.0]))
    @test isfinite(ValueGradient(1.0, [12.0]))
    @test !isinf(ValueGradient(1.0, [12.0]))
    @test ValueGradient(1, [2.0]) ≅ ValueGradient(1.0, [2.0]) # conversion
end

@testset "error printing" begin
    @test sprint(showerror, InvalidLogDensityException(0, NaN)) ==
        "InvalidLogDensityException: value is NaN"
    @test sprint(showerror, InvalidLogDensityException(1, Inf)) ==
        "InvalidLogDensityException: gradient[1] is Inf"
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
    @test dimension(p) == 1
    @test p.transformation ≡ t

    # gradient of a problem
    ∇p = ADgradient(:ForwardDiff, p)
    @test dimension(∇p) == 1
    @test parent(∇p).transformation ≡ t

    for _ in 1:100
        x = random_arg(p)
        θ, lj = transform_and_logjac(t, x)
        px = logdensity(Value, p, x)
        @test logpdf(d, θ.y) + lj ≈ (px::Value).value
        @test (logdensity(Value, ∇p, x)::Value).value ≈ px.value
        ∇px = logdensity(ValueGradient, ∇p, x)
        @test (∇px::ValueGradient).value ≈ px.value
        @test ∇px.gradient ≈ [ForwardDiff.derivative(x -> logpdf(d, exp(x)) + x, x[1])]
    end
end

@testset "-∞ log densities" begin
    t = as(Array, 2)
    validx = x -> all(x .> 0)
    p = TransformedLogDensity(t, x -> validx(x) ?  sum(abs2, x)/2 : -Inf)
    ∇p = ADgradient(:ForwardDiff, p)

    @test dimension(p) == dimension(∇p) == dimension(t)
    @test p.transformation ≡ parent(∇p).transformation ≡ t

    for _ in 1:100
        x = random_arg(p)
        px = logdensity(Value, ∇p, x)
        ∇px = logdensity(ValueGradient, ∇p, x)
        @test px isa Value
        @test ∇px isa ValueGradient
        @test px.value ≈ ∇px.value
        if validx(x)
            @test isfinite(px)
            @test isfinite(∇px)
            @test ∇px.value ≈ sum(abs2, x)/2
            @test ∇px.gradient ≈ x
        else
            @test isinf(px)
            @test isinf(∇px)
        end
    end
end

@testset "benchmark ForwardDiff problems" begin
    ℓ = TransformedLogDensity(as(Array, 20), x -> -sum(abs2, x))
    b = LogDensityProblems.benchmark_ForwardDiff_chunks(ℓ)
    @test b isa Vector{Pair{Int,Float64}}
    @test length(b) ≤ 20
end

@testset "stresstest" begin
    f = x -> all(x .< 0) ? NaN : -sum(abs2, x)
    ℓ = TransformedLogDensity(as(Array, 2), f)
    failures = LogDensityProblems.stresstest(ℓ; N = 1000)
    @test 230 ≤ length(failures) ≤ 270
    @test all(x -> all(x .< 0), failures)
end

@testset "show" begin
    t = as(Array, 5)
    p = TransformedLogDensity(t, x -> -sum(abs2, x))
    @test repr(p) == "TransformedLogDensity of dimension 5"
    @test repr(ADgradient(:ForwardDiff, p; chunk = 2)) ==
        ("ForwardDiff AD wrapper for " * repr(p) * ", w/ chunk size 2")
    @test repr(ADgradient(:ReverseDiff, p)) == ("ReverseDiff AD wrapper for " * repr(p))
    @test repr(ADgradient(:Flux, p)) == ("Flux AD wrapper for " * repr(p))
end

@testset "@iffinite" begin
    flag = [0]
    f(x) = (y = LogDensityProblems.@iffinite x; flag[1] += 1; y)
    @test f(NaN) ≡ NaN
    @test flag == [0]
    @test f(1) ≡ 1
    @test flag == [1]
end

@testset "reject wrapper" begin

    # function that throws various errors
    function f(x)
        y = first(x)
        if y > 1
            throw(DomainError(x))
        elseif y > 0
            throw(ArgumentError("bad"))
        elseif y > -1
            y
        else
            Inf
        end
    end
    x_dom = [1.5]
    x_arg = [0.5]
    x_inf = [-1.5]
    x_ok = [-0.5]

    # test unwrapped (for consistency)
    P = TransformedLogDensity(as(Array, 1), f)
    ∇P = ADgradient(:ForwardDiff, P)
    vgb = ValueGradientBuffer(randn(1))

    # -∞, not a valid value
    @test logdensity(Real, P, x_inf) == Inf
    @test_throws InvalidLogDensityException(0, Inf) logdensity(Value, P, x_inf)
    @test_throws InvalidLogDensityException logdensity(ValueGradient, ∇P, x_inf)
    @test_throws InvalidLogDensityException logdensity(vgb, ∇P, x_inf)

    # ArgumentError
    @test_throws ArgumentError logdensity(Real, P, x_arg)
    @test_throws ArgumentError logdensity(Value, P, x_arg)
    @test_throws ArgumentError logdensity(ValueGradient, ∇P, x_arg)
    @test_throws ArgumentError logdensity(vgb, ∇P, x_arg)

    # DomainError
    @test_throws DomainError logdensity(Real, P, x_dom)
    @test_throws DomainError logdensity(Value, P, x_dom)
    @test_throws DomainError logdensity(ValueGradient, ∇P, x_dom)
    @test_throws DomainError logdensity(vgb, ∇P, x_dom)

    # valid values
    @test logdensity(Real, P, x_ok) == -0.5
    @test logdensity(Value, P, x_ok) ≅ Value(-0.5)
    @test logdensity(ValueGradient, ∇P, x_ok) ≅ ValueGradient(-0.5, [1.0])
    @test logdensity(vgb, ∇P, x_ok) ≅ ValueGradient(-0.5, [1.0])

    # test wrapped -- we catch domain and invalid log density errors
    R = LogDensityRejectErrors{Union{DomainError,InvalidLogDensityException}}(∇P)

    # InvalidLogDensityException and DomainError converted to -∞
    @test logdensity(Real, R, x_inf) == Inf # no error
    @test logdensity(Real, R, x_dom) == -Inf # converted
    @test logdensity(Value, R, x_inf) ≅ logdensity(Value, R, x_dom) ≅ Value(-Inf)
    @test logdensity(ValueGradient, R, x_inf) ≅ logdensity(ValueGradient, R, x_dom) ≅
        ValueGradient(-Inf, x_inf)
    @test logdensity(vgb, R, x_inf) ≅ logdensity(vgb, R, x_dom) ≅ ValueGradient(-Inf, x_inf)

    # ArgumentError passes through
    @test_throws ArgumentError logdensity(Real, R, x_arg)
    @test_throws ArgumentError logdensity(Value, R, x_arg)
    @test_throws ArgumentError logdensity(ValueGradient, R, x_arg)
    @test_throws ArgumentError logdensity(vgb, R, x_arg)

    # valid values pass through

    @test logdensity(Real, R, x_ok) == -0.5
    @test logdensity(Value, R, x_ok) ≅ Value(-0.5)
    @test logdensity(ValueGradient, R, x_ok) ≅ ValueGradient(-0.5, [1.0])
    @test logdensity(vgb, R, x_ok) ≅ ValueGradient(-0.5, [1.0])

    # test constructor
    @test LogDensityRejectErrors{InvalidLogDensityException}(∇P) ≡
        LogDensityRejectErrors(∇P)
end

####
#### various AD tests
####

struct TestLogDensity end
TransformVariables.dimension(::TestLogDensity) = 3
test_logdensity(x) = any(x .< 0) ? -Inf : -2*abs2(x[1]) - 3*abs2(x[2]) - 5*abs2(x[3])
test_gradient(x) = x .* [-4, -6, -10]
LogDensityProblems.logdensity(::Type{Real}, ::TestLogDensity, x) = test_logdensity(x)

@testset "AD via ForwardDiff" begin
    ∇ℓ = ADgradient(:ForwardDiff, TestLogDensity())
    @test dimension(∇ℓ) == 3
    buffer = randn(3)
    vb = ValueGradientBuffer(buffer)
    buffer32 = randn(Float32, 3) # test non-matching buffer type
    vb32 = ValueGradientBuffer(buffer32)
    for _ in 1:100
        x = randn(3)
        @test logdensity(Real, ∇ℓ, x) ≈ test_logdensity(x)
        @test logdensity(Value, ∇ℓ, x) ≅ Value(test_logdensity(x))
        vg = ValueGradient(test_logdensity(x), test_gradient(x))
        @test logdensity(ValueGradient, ∇ℓ, x) ≅ vg
        vg2 = logdensity(vb, ∇ℓ, x)
        @test vg2.gradient ≡ buffer
        @test vg2 ≅ vg
        vg3 = logdensity(vb32, ∇ℓ, x)
        @test vg3.gradient ≡ buffer32
        @test vg3 ≈ vg
        @test vg3 isa ValueGradient{Float32, Vector{Float32}}
    end
end

@testset "AD via Flux" begin
    ∇ℓ = ADgradient(:Flux, TestLogDensity())
    @test dimension(∇ℓ) == 3
    buffer = randn(3)
    vb = ValueGradientBuffer(buffer)
    for _ in 1:100
        x = randn(3)
        @test logdensity(Real, ∇ℓ, x) ≈ test_logdensity(x)
        @test logdensity(Value, ∇ℓ, x) ≅ Value(test_logdensity(x))
        vg = ValueGradient(test_logdensity(x), test_gradient(x))
        @test logdensity(ValueGradient, ∇ℓ, x) ≅ vg
        # NOTE don't test buffer ≡, as that is not implemented for Flux
        @test logdensity(vb, ∇ℓ, x) ≅ vg
    end
end

@testset "AD via ReverseDiff" begin
    ∇ℓ = ADgradient(:ReverseDiff, TestLogDensity())
    @test dimension(∇ℓ) == 3
    buffer = randn(3)
    vb = ValueGradientBuffer(buffer)
    buffer32 = randn(Float32, 3) # test non-matching buffer type
    vb32 = ValueGradientBuffer(buffer32)
    for _ in 1:100
        x = randn(3)
        @test logdensity(Real, ∇ℓ, x) ≈ test_logdensity(x)
        @test logdensity(Value, ∇ℓ, x) ≅ Value(test_logdensity(x))
        vg = ValueGradient(test_logdensity(x), test_gradient(x))
        @test logdensity(ValueGradient, ∇ℓ, x) ≅ vg
        vg2 = logdensity(vb, ∇ℓ, x)
        @test vg2.gradient ≡ buffer
        @test vg2 ≅ vg
        vg3 = logdensity(vb32, ∇ℓ, x)
        @test vg3.gradient ≡ buffer32
        @test vg3 ≈ vg
        @test vg3 isa ValueGradient{Float32, Vector{Float32}}
    end
end

if VERSION ≥ v"1.1.0"
    # cf https://github.com/FluxML/Zygote.jl/issues/104
    import Pkg # use latest versions until tagged
    Pkg.add(Pkg.PackageSpec(name = "IRTools", rev = "master"))
    Pkg.add(Pkg.PackageSpec(name = "Zygote", rev = "master"))
    import Zygote

    @testset "AD via Zygote" begin
        ℓ = TestLogDensity()
        ∇ℓ = ADgradient(:Zygote, ℓ)
        @test repr(∇ℓ) == ("Zygote AD wrapper for " * repr(ℓ))
        @test dimension(∇ℓ) == 3
        buffer = randn(3)
        vb = ValueGradientBuffer(buffer)
        for _ in 1:100
            x = randn(3)
            @test logdensity(Real, ∇ℓ, x) ≈ test_logdensity(x)
            @test logdensity(Value, ∇ℓ, x) ≅ Value(test_logdensity(x))
            vg = ValueGradient(test_logdensity(x), test_gradient(x))
            @test logdensity(ValueGradient, ∇ℓ, x) ≅ vg
            # NOTE don't test buffer ≡, as that is not implemented for Zygote
            @test logdensity(vb, ∇ℓ, x) ≅ vg
        end
    end
end

@testset "ADgradient missing method" begin
    msg = "Don't know how to AD with Foo, consider `import Foo` if there is such a package."
    P = TransformedLogDensity(as(Array, 1), x -> sum(abs2, x))
    @test_logs((:info, msg), @test_throws(MethodError, ADgradient(:Foo, P)))
end

@testset "chunk heuristics for ForwardDiff" begin
    @test LogDensityProblems.heuristic_chunks(82) == vcat(1:4:81, [82])
end
