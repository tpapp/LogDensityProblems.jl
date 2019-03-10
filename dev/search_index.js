var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "This package provides the following functionality:It defines the logdensity method with corresponding interface, which can be used by other packages that operate on (log) densities and need to evaluate the log densities or the gradients (eg MCMC, MAP, ML or similar methods).\nIt defines the ADgradient which makes objects that support logdensity to calculate log density values calculate log density gradients using various automatic differentiation packages.\nIt defines the wrapper TransformedLogDensity using the TransformVariables.jl package, allowing callables that take a set of parameters transformed from a flat vector of real numbers to support the logdensity interface.\nVarious utility functions for debugging and testing log densities."
},

{
    "location": "#LogDensityProblems.logdensity",
    "page": "Home",
    "title": "LogDensityProblems.logdensity",
    "category": "function",
    "text": "logdensity(resulttype, ℓ, x)\n\nEvaluate the AbstractLogDensityProblem ℓ at x, which has length compatible with its dimension.\n\nThe argument resulttype determines the type of the result:\n\nReal for an unchecked evaluation of the log density which should return a ::Real\n\nnumber (that could be NaN, Inf, etc),\n\nValue for a checked log density, returning a Value,\nValueGradient also calculates the gradient, returning a ValueGradient,\nValueGradientBuffer calculates a ValueGradient potentially (but always\n\nconsistently for argument types) using the provided buffer for the gradient. In this case, the element type of the array may determine the result element type.\n\nImplementation note\n\nMost types should just define the methods for Real and ValueGradientBuffer (when applicable), as Value and ValueGradient fall back to these, respectively.\n\n\n\n\n\n"
},

{
    "location": "#TransformVariables.dimension",
    "page": "Home",
    "title": "TransformVariables.dimension",
    "category": "function",
    "text": "dimension(p)\n\n\nThe dimension of the problem, ie the length of the vectors in its domain.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.Value",
    "page": "Home",
    "title": "LogDensityProblems.Value",
    "category": "type",
    "text": "Value(value)\n\n\nHolds the value of a logdensity at a given point.\n\nConstructor ensures that the value is either finite, or -.\n\nAll other values (eg NaN or Inf for the value) lead to an error.\n\nSee also logdensity.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.ValueGradient",
    "page": "Home",
    "title": "LogDensityProblems.ValueGradient",
    "category": "type",
    "text": "ValueGradient(value, gradient)\n\n\nHolds the value and gradient of a logdensity at a given point.\n\nConstructor ensures that either\n\nboth the value and the gradient are finite,\nthe value is - (then gradient is not checked).\n\nAll other values (eg NaN or Inf for the value) lead to an error.\n\nSee also logdensity.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.ValueGradientBuffer",
    "page": "Home",
    "title": "LogDensityProblems.ValueGradientBuffer",
    "category": "type",
    "text": "struct ValueGradientBuffer{T<:Real, V<:AbstractArray{T<:Real,1}}\n\nA wrapper for a vector that indicates that the vector may be used for the gradient in a ValueGradient. Consequences are undefined if it is modified later, implicitly the caller guarantees that it will not be used for anything else while the gradient is retrieved.\n\nSee logdensity.\n\n\n\n\n\n"
},

{
    "location": "#Inference-1",
    "page": "Home",
    "title": "Inference",
    "category": "section",
    "text": "logdensity\ndimension\nLogDensityProblems.Value\nLogDensityProblems.ValueGradient\nLogDensityProblems.ValueGradientBuffer"
},

{
    "location": "#LogDensityProblems.ADgradient",
    "page": "Home",
    "title": "LogDensityProblems.ADgradient",
    "category": "function",
    "text": "ADgradient(kind, P; kwargs...)\n\n\nWrap P using automatic differentiation to obtain a gradient.\n\nkind is usually a Val type, containing a symbol that refers to a package. The symbol can also be used directly as eg\n\nADgradient(:ForwardDiff, P)\n\nSee methods(ADgradient). Note that some methods are defined conditionally on the relevant package being loaded.\n\n\n\n\n\nADgradient(#temp#, ℓ; chunk, gradientconfig)\n\n\nWrap a log density that supports evaluation of Value to handle ValueGradient, using ForwardDiff.\n\nKeywords are passed on to ForwardDiff.GradientConfig to customize the setup. In particular, chunk size can be set with a chunk keyword argument (accepting an integer or a ForwardDiff.Chunk).\n\n\n\n\n\nADgradient(?, ℓ)\n\n\nGradient using algorithmic/automatic differentiation via Flux.\n\n\n\n\n\n"
},

{
    "location": "#Gradient-via-automatic-differentiation-1",
    "page": "Home",
    "title": "Gradient via automatic differentiation",
    "category": "section",
    "text": "ADgradient"
},

{
    "location": "#LogDensityProblems.TransformedLogDensity",
    "page": "Home",
    "title": "LogDensityProblems.TransformedLogDensity",
    "category": "type",
    "text": "TransformedLogDensity(transformation, log_density_function)\n\nA problem in Bayesian inference. Vectors of length dimension(transformation) are transformed into a general object θ (unrestricted type, but a named tuple is recommended for clean code), correcting for the log Jacobian determinant of the transformation.\n\nIt is recommended that log_density_function is a callable object that also encapsulates the data for the problem.\n\nlog_density_function(θ) is expected to return real numbers. For zero densities or infeasible θs, -Inf or similar should be returned, but for efficiency of inference most methods recommend using transformation to avoid this.\n\nUse the property accessors ℓ.transformation and ℓ.log_density_function to access the arguments of ℓ::TransformedLogDensity, these are part of the API.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.reject_logdensity",
    "page": "Home",
    "title": "LogDensityProblems.reject_logdensity",
    "category": "function",
    "text": "reject_logdensity()\n\n\nMake wrappers return a -Inf log density (of the appropriate type).\n\nnote: Note\nThis is done by throwing an exception that is caught by the wrappers, unwinding the stack. Using this function or returning -Inf is an implementation choice, do whatever is most convenient.\n\n\n\n\n\n"
},

{
    "location": "#Transformed-problem-definition-1",
    "page": "Home",
    "title": "Transformed problem definition",
    "category": "section",
    "text": "TransformedLogDensity\nreject_logdensity"
},

{
    "location": "#LogDensityProblems.stresstest",
    "page": "Home",
    "title": "LogDensityProblems.stresstest",
    "category": "function",
    "text": "stresstest(ℓ; N, rng, scale, resulttype)\n\n\nTest ℓ with random values.\n\nN random vectors are drawn from a standard multivariate Cauchy distribution, scaled with scale (which can be a scalar or a conformable vector). In case the call produces an error, the value is recorded as a failure, which are returned by the function.\n\nNot exported, but part of the API.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.benchmark_ForwardDiff_chunks",
    "page": "Home",
    "title": "LogDensityProblems.benchmark_ForwardDiff_chunks",
    "category": "function",
    "text": "benchmark_ForwardDiff_chunks(ℓ; chunks, resulttype, markprogress)\n\n\nBenchmark a log density problem with various chunk sizes using ForwardDiff.\n\nchunks, which defaults to all possible chunk sizes, determines the chunks that are tried.\n\nThe function returns chunk => time pairs, where time is the benchmarked runtime in seconds, as determined by BenchmarkTools.@belapsed.\n\nRuntime may be long because of tuned benchmarks, so when markprogress == true (the  default), dots are printed to mark progress.\n\nThis function is not exported, but part of the API.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.@iffinite",
    "page": "Home",
    "title": "LogDensityProblems.@iffinite",
    "category": "macro",
    "text": "If expr evaluates to a non-finite value, return with that, otherwise evaluate to that value. Useful for returning early from non-finite likelihoods.\n\nPart of the API, but not exported.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.LogDensityRejectErrors",
    "page": "Home",
    "title": "LogDensityProblems.LogDensityRejectErrors",
    "category": "type",
    "text": "LogDensityRejectErrors(ℓ)\n\n\nWrap a logdensity ℓ so that errors <: E are caught and replaced with a - value.\n\nE defaults to InvalidLogDensityExceptions.\n\nNote\n\nUse cautiously, as catching errors can mask errors in your code. The recommended use case is for catching quirks and corner cases of AD. See also stresstest as an alternative to using this wrapper.\n\n\n\n\n\n"
},

{
    "location": "#Benchmarking,-diagnostics,-and-utilities-1",
    "page": "Home",
    "title": "Benchmarking, diagnostics, and utilities",
    "category": "section",
    "text": "LogDensityProblems.stresstest\nLogDensityProblems.benchmark_ForwardDiff_chunks\nLogDensityProblems.@iffinite\nLogDensityRejectErrors"
},

{
    "location": "internals/#",
    "page": "Internals",
    "title": "Internals",
    "category": "page",
    "text": ""
},

{
    "location": "internals/#LogDensityProblems.AbstractLogDensityProblem",
    "page": "Internals",
    "title": "LogDensityProblems.AbstractLogDensityProblem",
    "category": "type",
    "text": "Abstract type for log density representations, which support the following interface for ℓ::AbstractLogDensityProblem:\n\ndimension returns the dimension of the domain of ℓ,\nlogdensity evaluates the log density ℓ at a given point.\n\nSee also LogDensityProblems.stresstest for stress testing.\n\n\n\n\n\n"
},

{
    "location": "internals/#LogDensityProblems.LogDensityWrapper",
    "page": "Internals",
    "title": "LogDensityProblems.LogDensityWrapper",
    "category": "type",
    "text": "An abstract type that wraps another log density in its field ℓ.\n\nNotes\n\nImplementation detail, not exported.\n\n\n\n\n\n"
},

{
    "location": "internals/#LogDensityProblems.ADGradientWrapper",
    "page": "Internals",
    "title": "LogDensityProblems.ADGradientWrapper",
    "category": "type",
    "text": "An abstract type that wraps another log density for calculating the gradient via AD.\n\nAutomatically defines a logdensity(Value, ...) method, subtypes should define a logdensity(ValueGradient, ...) one.\n\n\n\n\n\n"
},

{
    "location": "internals/#LogDensityProblems.heuristic_chunks",
    "page": "Internals",
    "title": "LogDensityProblems.heuristic_chunks",
    "category": "function",
    "text": "heuristic_chunks(N)\nheuristic_chunks(N, M)\n\n\nDefault chunk sizes to try for benchmarking. Fewer than M, always contains 1 and N.\n\n\n\n\n\n"
},

{
    "location": "internals/#LogDensityProblems.RejectLogDensity",
    "page": "Internals",
    "title": "LogDensityProblems.RejectLogDensity",
    "category": "type",
    "text": "struct RejectLogDensity <: Exception\n\nException for unwinding the stack early for infeasible values. Use reject_logdensity().\n\n\n\n\n\n"
},

{
    "location": "internals/#Internals-1",
    "page": "Internals",
    "title": "Internals",
    "category": "section",
    "text": "LogDensityProblems.AbstractLogDensityProblem\nLogDensityProblems.LogDensityWrapper\nLogDensityProblems.ADGradientWrapper\nLogDensityProblems.heuristic_chunks\nLogDensityProblems.RejectLogDensity"
},

]}
