export Counters, sum_counters, increment!, decrement!, reset!

"""
    Counters

Struct for storing the number of function evaluations.

---

    Counters()

Creates an empty Counters struct.
"""
mutable struct Counters
  neval_obj::Int  # Number of objective evaluations.
  neval_grad::Int  # Number of objective gradient evaluations.
  neval_cons::Int  # Number of constraint vector evaluations.
  neval_jcon::Int  # Number of individual constraint evaluations.
  neval_jgrad::Int  # Number of individual constraint gradient evaluations.
  neval_jac::Int  # Number of constraint Jacobian evaluations.
  neval_jprod::Int  # Number of Jacobian-vector products.
  neval_jtprod::Int  # Number of transposed Jacobian-vector products.
  neval_hess::Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hprod::Int  # Number of Lagrangian/objective Hessian-vector products.
  neval_jhess::Int  # Number of individual Lagrangian Hessian evaluations.
  neval_jhprod::Int  # Number of individual constraint Hessian-vector products.

  function Counters()
    return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  end
end

# simple default API for retrieving counters
for counter in fieldnames(Counters)
  @eval begin
    """
        $($counter)(nlp)

    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nlp::AbstractNLPModel) = nlp.counters.$counter
    export $counter
  end
end

"""
    increment!(nlp, s)

Increment counter `s` of problem `nlp`.
"""
function increment!(nlp::AbstractNLPModel, s::Symbol)
  setproperty!(nlp.counters, s, getproperty(nlp.counters, s) + 1)
end

"""
    decrement!(nlp, s)

Decrement counter `s` of problem `nlp`.
"""
function decrement!(nlp::AbstractNLPModel, s::Symbol)
  setproperty!(nlp.counters, s, getproperty(nlp.counters, s) - 1)
end

"""
    sum_counters(counters)

Sum all counters of `counters`.
"""
sum_counters(c::Counters) = sum(getproperty(c, x) for x in fieldnames(Counters))

"""
    sum_counters(nlp)

Sum all counters of problem `nlp`.
"""
sum_counters(nlp::AbstractNLPModel) = sum_counters(nlp.counters)

"""
    reset!(counters)

Reset evaluation counters
"""
function LinearOperators.reset!(counters::Counters)
  for f in fieldnames(Counters)
    setfield!(counters, f, 0)
  end
  return counters
end

"""
    reset!(nlp)

Reset evaluation count in `nlp`
"""
function LinearOperators.reset!(nlp::AbstractNLPModel)
  reset!(nlp.counters)
  reset_data!(nlp)
  return nlp
end
