export has_bounds, bound_constrained, unconstrained
export linearly_constrained, equality_constrained, inequality_constrained
export has_equalities, has_inequalities

"""
    has_bounds(nlp)
    has_bounds(meta)

Returns whether the problem has bounds on the variables.
"""
has_bounds(meta::NLPModelMeta) = length(meta.ifree) < meta.nvar

"""
    bound_constrained(nlp)
    bound_constrained(meta)

Returns whether the problem has bounds on the variables and no other constraints.
"""
bound_constrained(meta::NLPModelMeta) = meta.ncon == 0 && has_bounds(meta)

"""
    unconstrained(nlp)
    unconstrained(meta)

Returns whether the problem in unconstrained.
"""
unconstrained(meta::NLPModelMeta) = meta.ncon == 0 && !has_bounds(meta)

"""
    linearly_constrained(nlp)
    linearly_constrained(meta)

Returns whether the problem's constraints are known to be all linear.
"""
linearly_constrained(meta::NLPModelMeta) = meta.nlin == meta.ncon > 0

"""
    equality_constrained(nlp)
    equality_constrained(meta)

Returns whether the problem's constraints are all equalities.
Unconstrained problems return false.
"""
equality_constrained(meta::NLPModelMeta) = length(meta.jfix) == meta.ncon > 0

"""
    inequality_constrained(nlp)
    inequality_constrained(meta)

Returns whether the problem's constraints are all inequalities.
Unconstrained problems return true.
"""
inequality_constrained(meta::NLPModelMeta) = meta.ncon > 0 && length(meta.jfix) == 0

"""
    has_equalities(nlp)

Returns whether the problem has constraints and at least one of them is an equality.
Unconstrained problems return false.
"""
has_equalities(meta::NLPModelMeta) = meta.ncon ≥ length(meta.jfix) > 0

"""
    has_inequalities(nlp)

Returns whether the problem has constraints and at least one of them is an inequality.
Unconstrained problems return false.
"""
has_inequalities(meta::NLPModelMeta) = meta.ncon > 0 && meta.ncon > length(meta.jfix)

for meth in [
  :has_bounds,
  :bound_constrained,
  :unconstrained,
  :linearly_constrained,
  :equality_constrained,
  :inequality_constrained,
  :has_equalities,
  :has_inequalities,
]
  @eval $meth(nlp::AbstractNLPModel) = $meth(nlp.meta)
end
