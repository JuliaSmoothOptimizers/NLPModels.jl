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
    if meta.variable_bounds_analysis
        return length(meta.ifree) < meta.nvar
    else
        return !all(lv -> isinf(lv), meta.lvar) || !all(uv -> isinf(uv), meta.uvar)
    end
end

"""
    bound_constrained(nlp)
    bound_constrained(meta)

Returns whether the problem has bounds on the variables and no other constraints.
"""
bound_constrained(meta::AbstractNLPModelMeta) = (meta.ncon == 0) && has_bounds(meta)

"""
    unconstrained(nlp)
    unconstrained(meta)

Returns whether the problem in unconstrained.
"""
unconstrained(meta::AbstractNLPModelMeta) = (meta.ncon == 0) && !has_bounds(meta)

"""
    linearly_constrained(nlp)
    linearly_constrained(meta)

Returns whether the problem's constraints are known to be all linear.
"""
linearly_constrained(meta::AbstractNLPModelMeta) = (meta.ncon > 0) && (meta.nlin == meta.ncon)

"""
    equality_constrained(nlp)
    equality_constrained(meta)

Returns whether the problem's constraints are all equalities.
Unconstrained problems return false.
"""
function equality_constrained(meta::AbstractNLPModelMeta)
    if meta.constraint_bounds_analysis
        return (meta.ncon > 0) && (length(meta.jfix) == meta.ncon)
    else
        return (meta.ncon > 0) && all(x -> x[1] == x[2], zip(meta.lcon, meta.ucon))
    end
end

"""
    inequality_constrained(nlp)
    inequality_constrained(meta)

Returns whether the problem's constraints are all inequalities.
Unconstrained problems return true.
"""
function inequality_constrained(meta::AbstractNLPModelMeta)
    if meta.constraint_bounds_analysis
        return (meta.ncon > 0) && (length(meta.jfix) == 0)
    else
        return (meta.ncon > 0) && all(x -> x[1] != x[2], zip(meta.lcon, meta.ucon))
    end
end

"""
    has_equalities(nlp)

Returns whether the problem has constraints and at least one of them is an equality.
Unconstrained problems return false.
"""
function has_equalities(meta::AbstractNLPModelMeta)
    if meta.constraint_bounds_analysis
        return (meta.ncon > 0) && (length(meta.jfix) > 0)
    else
        return (meta.ncon > 0) && !all(x -> x[1] != x[2], zip(meta.lcon, meta.ucon))
    end
end

"""
    has_inequalities(nlp)

Returns whether the problem has constraints and at least one of them is an inequality.
Unconstrained problems return false.
"""
function has_inequalities(meta::AbstractNLPModelMeta)
    if meta.constraint_bounds_analysis
        return (meta.ncon > 0) && (meta.ncon > length(meta.jfix))
    else
        return (meta.ncon > 0) && !all(x -> x[1] == x[2], zip(meta.lcon, meta.ucon))
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
