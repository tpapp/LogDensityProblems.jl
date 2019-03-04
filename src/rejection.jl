#####
##### Rejection via exceptions: shortcut to a -∞ log density.
#####

"""
$(TYPEDEF)

Exception for unwinding the stack early for infeasible values. Use `reject_logdensity()`.
"""
struct RejectLogDensity <: Exception end

"""
$(SIGNATURES)

Make wrappers return a `-Inf` log density (of the appropriate type).

!!! note

    This is done by throwing an exception that is caught by the wrappers, unwinding the
    stack. Using this function or returning `-Inf` is an implementation choice, do whatever
    is most convenient.
"""
reject_logdensity() = throw(RejectLogDensity())
