#####
##### Helper functions for working with DiffResults
#####
##### Only included when required by AD wrappers.

import .DiffResults

"""
$(SIGNATURES)

Allocate a DiffResults buffer for a gradient, taking the element type of `x` into account
(heuristically).
"""
function _diffresults_buffer(ℓ, x)
    T = eltype(x)
    S = T <: Real ? float(Real) : Float64 # heuristic
    DiffResults.MutableDiffResult(zero(S), (Vector{S}(undef, dimension(ℓ)), ))
end

"""
$(SIGNATURES)

Extract a return value for [`logdensity_and_gradient`](@ref) from a DiffResults buffer,
constructed with [`diffresults_buffer`](@ref). Gradient is not copied as caller created the
vector.
"""
function _diffresults_extract(diffresult::DiffResults.DiffResult)
    DiffResults.value(diffresult)::Real, DiffResults.gradient(diffresult)
end
