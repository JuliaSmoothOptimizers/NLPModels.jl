export NLSCounters

"""
    NLSCounters

Struct for storing the number of functions evaluations for nonlinear least-squares models.
NLSCounters also stores a `Counters` instance named `counters`.

---

    NLSCounters()

Creates an empty NLSCounters struct.
"""
mutable struct NLSCounters
  counters::Counters
  neval_residual::Int
  neval_jac_residual::Int
  neval_jprod_residual::Int
  neval_jtprod_residual::Int
  neval_hess_residual::Int
  neval_jhess_residual::Int
  neval_hprod_residual::Int

  function NLSCounters()
    return new(Counters(), 0, 0, 0, 0, 0, 0, 0)
  end
end

function Base.getproperty(c::NLSCounters, f::Symbol)
  if f in fieldnames(Counters)
    getfield(c.counters, f)
  else
    getfield(c, f)
  end
end

function Base.setproperty!(c::NLSCounters, f::Symbol, x)
  if f in fieldnames(Counters)
    setfield!(c.counters, f, x)
  else
    setfield!(c, f, x)
  end
end

function sum_counters(c::NLSCounters)
  s = sum_counters(c.counters)
  for field in fieldnames(NLSCounters)
    field == :counters && continue
    s += getfield(c, field)
  end
  return s
end
sum_counters(nls::AbstractNLSModel) = sum_counters(nls.counters)

for counter in fieldnames(NLSCounters)
  counter == :counters && continue
  @eval begin
    """
    $($counter)(nlp)

    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nls::AbstractNLSModel) = nls.counters.$counter
    export $counter
  end
end

for counter in fieldnames(Counters)
  @eval begin
    $counter(nls::AbstractNLSModel) = nls.counters.counters.$counter
    export $counter
  end
end

"""
    increment!(nls, s)

Increment counter `s` of problem `nls`.
"""
@inline function increment!(nls::AbstractNLSModel, s::Symbol)
  increment!(nls, Val(s))
end

for fun in fieldnames(NLSCounters)
  fun == :counters && continue
  @eval increment!(nls::AbstractNLSModel, ::Val{$(Meta.quot(fun))}) = nls.counters.$fun += 1
end

for fun in fieldnames(Counters)
  @eval $NLPModels.increment!(nls::AbstractNLSModel, ::Val{$(Meta.quot(fun))}) =
    nls.counters.counters.$fun += 1
end

function LinearOperators.reset!(nls::AbstractNLSModel)
  reset!(nls.counters)
  return nls
end

function LinearOperators.reset!(nls_counters::NLSCounters)
  for f in fieldnames(NLSCounters)
    f == :counters && continue
    setfield!(nls_counters, f, 0)
  end
  reset!(nls_counters.counters)
  return nls_counters
end
