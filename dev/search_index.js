var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Overview",
    "title": "Overview",
    "category": "page",
    "text": ""
},

{
    "location": "#Introduction-1",
    "page": "Overview",
    "title": "Introduction",
    "category": "section",
    "text": "note: Note\nCurrently this is just a placeholder that renders the docstrings."
},

{
    "location": "#LogDensityProblems.TransformedLogDensity",
    "page": "Overview",
    "title": "LogDensityProblems.TransformedLogDensity",
    "category": "type",
    "text": "TransformedLogDensity(transformation, log_density_function)\n\nA problem in Bayesian inference. Vectors of length dimension(transformation) are transformed into a general object θ (unrestricted type, but a named tuple is recommended for clean code), correcting for the log Jacobian determinant of the transformation.\n\nIt is recommended that log_density_function is a callable object that also encapsulates the data for the problem.\n\nlog_density_function(θ) is expected to return real numbers. For zero densities or infeasible θs, -Inf or similar should be returned, but for efficiency of inference most methods recommend using transformation to avoid this.\n\nUse the property accessors ℓ.transformation and ℓ.log_density_function to access the arguments of ℓ::TransformedLogDensity, these are part of the API.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.reject_logdensity",
    "page": "Overview",
    "title": "LogDensityProblems.reject_logdensity",
    "category": "function",
    "text": "reject_logdensity()\n\n\nMake wrappers return a -Inf log density (of the appropriate type).\n\nnote: Note\nThis is done by throwing an exception that is caught by the wrappers, unwinding the stack. Using this function or returning -Inf is an implementation choice, do whatever is most convenient.\n\n\n\n\n\n"
},

{
    "location": "#Problem-definition-1",
    "page": "Overview",
    "title": "Problem definition",
    "category": "section",
    "text": "TransformedLogDensity\nreject_logdensity"
},

{
    "location": "#LogDensityProblems.ADgradient",
    "page": "Overview",
    "title": "LogDensityProblems.ADgradient",
    "category": "function",
    "text": "ADgradient(kind, P; kwargs...)\n\n\nWrap P using automatic differentiation to obtain a gradient.\n\nkind is usually a Val type, containing a symbol that refers to a package. The symbol can also be used directly as eg\n\nADgradient(:ForwardDiff, P)\n\nSee methods(ADgradient). Note that some methods are defined conditionally on the relevant package being loaded.\n\n\n\n\n\nADgradient(#temp#, ℓ; chunk, gradientconfig)\n\n\nWrap a log density that supports evaluation of Value to handle ValueGradient, using ForwardDiff.\n\nKeywords are passed on to ForwardDiff.GradientConfig to customize the setup. In particular, chunk size can be set with a chunk keyword argument (accepting an integer or a ForwardDiff.Chunk).\n\n\n\n\n\nADgradient(?, ℓ)\n\n\nGradient using algorithmic/automatic differentiation via Flux.\n\n\n\n\n\n"
},

{
    "location": "#Gradient-via-automatic-differentiation-1",
    "page": "Overview",
    "title": "Gradient via automatic differentiation",
    "category": "section",
    "text": "ADgradient"
},

{
    "location": "#LogDensityProblems.logdensity",
    "page": "Overview",
    "title": "LogDensityProblems.logdensity",
    "category": "function",
    "text": "logdensity(resulttype, ℓ, x)\n\nEvaluate the AbstractLogDensityProblem ℓ at x, which has length compatible with its dimension.\n\nThe argument resulttype determines the type of the result. [Value]@(ref) results in the log density, while ValueGradient also calculates the gradient, both returning eponymous types.\n\n\n\n\n\n"
},

{
    "location": "#TransformVariables.dimension",
    "page": "Overview",
    "title": "TransformVariables.dimension",
    "category": "function",
    "text": "dimension(p)\n\n\nThe dimension of the problem, ie the length of the vectors in its domain.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.Value",
    "page": "Overview",
    "title": "LogDensityProblems.Value",
    "category": "type",
    "text": "Value(value)\n\n\nHolds the value of a logdensity at a given point.\n\nConstructor ensures that the value is either finite, or -.\n\nAll other values (eg NaN or Inf for the value) lead to an error.\n\nSee also logdensity.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.ValueGradient",
    "page": "Overview",
    "title": "LogDensityProblems.ValueGradient",
    "category": "type",
    "text": "ValueGradient(value, gradient)\n\n\nHolds the value and gradient of a logdensity at a given point.\n\nConstructor ensures that either\n\nboth the value and the gradient are finite,\nthe value is - (then gradient is not checked).\n\nAll other values (eg NaN or Inf for the value) lead to an error.\n\nSee also logdensity.\n\n\n\n\n\n"
},

{
    "location": "#Inference-1",
    "page": "Overview",
    "title": "Inference",
    "category": "section",
    "text": "logdensity\ndimension\nLogDensityProblems.Value\nLogDensityProblems.ValueGradient"
},

{
    "location": "#LogDensityProblems.stresstest",
    "page": "Overview",
    "title": "LogDensityProblems.stresstest",
    "category": "function",
    "text": "stresstest(ℓ; N, rng, scale, resulttype)\n\n\nTest ℓ with random values.\n\nN random vectors are drawn from a standard multivariate Cauchy distribution, scaled with scale (which can be a scalar or a conformable vector). In case the call produces an error, the value is recorded as a failure, which are returned by the function.\n\nNot exported, but part of the API.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.benchmark_ForwardDiff_chunks",
    "page": "Overview",
    "title": "LogDensityProblems.benchmark_ForwardDiff_chunks",
    "category": "function",
    "text": "benchmark_ForwardDiff_chunks(ℓ; chunks, resulttype, markprogress)\n\n\nBenchmark a log density problem with various chunk sizes using ForwardDiff.\n\nchunks, which defaults to all possible chunk sizes, determines the chunks that are tried.\n\nThe function returns chunk => time pairs, where time is the benchmarked runtime in seconds, as determined by BenchmarkTools.@belapsed.\n\nRuntime may be long because of tuned benchmarks, so when markprogress == true (the  default), dots are printed to mark progress.\n\nThis function is not exported, but part of the API.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.@iffinite",
    "page": "Overview",
    "title": "LogDensityProblems.@iffinite",
    "category": "macro",
    "text": "If expr evaluates to a non-finite value, return with that, otherwise evaluate to that value. Useful for returning early from non-finite likelihoods.\n\nPart of the API, but not exported.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.LogDensityRejectErrors",
    "page": "Overview",
    "title": "LogDensityProblems.LogDensityRejectErrors",
    "category": "type",
    "text": "LogDensityRejectErrors(ℓ)\n\n\nWrap a logdensity ℓ so that errors <: E are caught and replaced with a - value.\n\nE defaults to InvalidLogDensityExceptions.\n\nNote\n\nUse cautiously, as catching errors can mask errors in your code. The recommended use case is for catching quirks and corner cases of AD. See also stresstest as an alternative to using this wrapper.\n\n\n\n\n\n"
},

{
    "location": "#Benchmarking,-diagnostics,-and-utilities-1",
    "page": "Overview",
    "title": "Benchmarking, diagnostics, and utilities",
    "category": "section",
    "text": "LogDensityProblems.stresstest\nLogDensityProblems.benchmark_ForwardDiff_chunks\nLogDensityProblems.@iffinite\nLogDensityRejectErrors"
},

{
    "location": "#LogDensityProblems.AbstractLogDensityProblem",
    "page": "Overview",
    "title": "LogDensityProblems.AbstractLogDensityProblem",
    "category": "type",
    "text": "Abstract type for log density representations, which support the following interface for ℓ::AbstractLogDensityProblem:\n\ndimension returns the dimension of the domain of ℓ,\nlogdensity evaluates the log density ℓ at a given point.\n\nSee also LogDensityProblems.stresstest for stress testing.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.LogDensityWrapper",
    "page": "Overview",
    "title": "LogDensityProblems.LogDensityWrapper",
    "category": "type",
    "text": "An abstract type that wraps another log density in its field ℓ.\n\nNotes\n\nImplementation detail, not exported.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.ADGradientWrapper",
    "page": "Overview",
    "title": "LogDensityProblems.ADGradientWrapper",
    "category": "type",
    "text": "An abstract type that wraps another log density for calculating the gradient via AD.\n\nAutomatically defines a logdensity(Value, ...) method, subtypes should define a logdensity(ValueGradient, ...) one.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.heuristic_chunks",
    "page": "Overview",
    "title": "LogDensityProblems.heuristic_chunks",
    "category": "function",
    "text": "heuristic_chunks(N)\nheuristic_chunks(N, M)\n\n\nDefault chunk sizes to try for benchmarking. Fewer than M, always contains 1 and N.\n\n\n\n\n\n"
},

{
    "location": "#LogDensityProblems.RejectLogDensity",
    "page": "Overview",
    "title": "LogDensityProblems.RejectLogDensity",
    "category": "type",
    "text": "struct RejectLogDensity <: Exception\n\nException for unwinding the stack early for infeasible values. Use reject_logdensity().\n\n\n\n\n\n"
},

{
    "location": "#Internals-1",
    "page": "Overview",
    "title": "Internals",
    "category": "section",
    "text": "LogDensityProblems.AbstractLogDensityProblem\nLogDensityProblems.LogDensityWrapper\nLogDensityProblems.ADGradientWrapper\nLogDensityProblems.heuristic_chunks\nLogDensityProblems.RejectLogDensity"
},

]}
