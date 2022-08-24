using .TransformVariables: TransformVariables

"""
    TransformedLogDensity(t::TransformVariables.AbstractTransform, log_density_function)

Construct a `TransformedLogDensity` with transform `TransformVariables.transform(t)` and
log density function `log_density_function`.

For the resulting log density function, `capabilities`, `logdensity`, and `dimension` are
automatically defined.
"""
function TransformedLogDensity(t::TransformVariables.AbstractTransform, log_density_function)
    return TransformedLogDensity(TransformVariables.transform(t), log_density_function)
end

function dimension(p::TransformedLogDensity{<:TransformVariables.CallableTransform})
    return TransformVariables.dimension(p.transform.t)
end
