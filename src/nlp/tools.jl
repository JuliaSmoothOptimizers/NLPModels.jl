export has_bounds, bound_constrained, unconstrained
export linearly_constrained, equality_constrained, inequality_constrained
export has_equalities, has_inequalities

for field in fieldnames(NLPModelMeta)
  meth = Symbol("get_", field)
  @eval begin
    @doc """
        $($meth)(nlp)
        $($meth)(meta)
    Return the value $($(QuoteNode(field))) from meta or nlp.meta.
    """
    $meth(meta::AbstractNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
  end
  @eval $meth(nlp::AbstractNLPModel) = $meth(nlp.meta)
  @eval export $meth
end

"""
    has_bounds(nlp)
    has_bounds(meta)

Returns whether the problem has bounds on the variables.
"""
function has_bounds(meta::AbstractNLPModelMeta)
  if get_variable_bounds_analysis(meta)
    return length(get_ifree(meta)) < get_nvar(meta)
  else
    return !all(lv -> isinf(lv), get_lvar(meta)) || !all(uv -> isinf(uv), get_uvar(meta))
  end
end

"""
    bound_constrained(nlp)
    bound_constrained(meta)

Returns whether the problem has bounds on the variables and no other constraints.
"""
bound_constrained(meta::AbstractNLPModelMeta) = (get_ncon(meta) == 0) && has_bounds(meta)

"""
    unconstrained(nlp)
    unconstrained(meta)

Returns whether the problem in unconstrained.
"""
unconstrained(meta::AbstractNLPModelMeta) = (get_ncon(meta) == 0) && !has_bounds(meta)

"""
    linearly_constrained(nlp)
    linearly_constrained(meta)

Returns whether the problem's constraints are known to be all linear.
"""
linearly_constrained(meta::AbstractNLPModelMeta) =
  (get_ncon(meta) > 0) && (get_nlin(meta) == get_ncon(meta))

"""
    equality_constrained(nlp)
    equality_constrained(meta)

Returns whether the problem's constraints are all equalities.
Unconstrained problems return false.
"""
function equality_constrained(meta::AbstractNLPModelMeta)
  if get_constraint_bounds_analysis(meta)
    return (get_ncon(meta) > 0) && (length(get_jfix(meta)) == get_ncon(meta))
  else
    return (get_ncon(meta) > 0) && all(x -> x[1] == x[2], zip(get_lcon(meta), get_ucon(meta)))
  end
end

"""
    inequality_constrained(nlp)
    inequality_constrained(meta)

Returns whether the problem's constraints are all inequalities.
Unconstrained problems return true.
"""
function inequality_constrained(meta::AbstractNLPModelMeta)
  if get_constraint_bounds_analysis(meta)
    return (get_ncon(meta) > 0) && (length(get_jfix(meta)) == 0)
  else
    return (get_ncon(meta) > 0) && all(x -> x[1] != x[2], zip(get_lcon(meta), get_ucon(meta)))
  end
end

"""
    has_equalities(nlp)

Returns whether the problem has constraints and at least one of them is an equality.
Unconstrained problems return false.
"""
function has_equalities(meta::AbstractNLPModelMeta)
  if get_constraint_bounds_analysis(meta)
    return (get_ncon(meta) > 0) && (length(get_jfix(meta)) > 0)
  else
    return (get_ncon(meta) > 0) && !all(x -> x[1] != x[2], zip(get_lcon(meta), get_ucon(meta)))
  end
end

"""
    has_inequalities(nlp)

Returns whether the problem has constraints and at least one of them is an inequality.
Unconstrained problems return false.
"""
function has_inequalities(meta::AbstractNLPModelMeta)
  if get_constraint_bounds_analysis(meta)
    return (get_ncon(meta) > 0) && (get_ncon(meta) > length(get_jfix(meta)))
  else
    return (get_ncon(meta) > 0) && !all(x -> x[1] == x[2], zip(get_lcon(meta), get_ucon(meta)))
  end
end

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
