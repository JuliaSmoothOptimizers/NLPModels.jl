export coo_prod!, coo_sym_prod!
export @default_counters
export DimensionError, @lencheck, @rangecheck

"""
    DimensionError <: Exception
    DimensionError(name, dim_expected, dim_found)

Error for unexpected dimension.
Output: "DimensionError: Input `name` should have length `dim_expected` not `dim_found`"
"""
struct DimensionError <: Exception
  name::Union{Symbol, String}
  dim_expected::Int
  dim_found::Int
end

function Base.showerror(io::IO, e::DimensionError)
  print(
    io,
    "DimensionError: Input $(e.name) should have length $(e.dim_expected) not $(e.dim_found)",
  )
end

# https://groups.google.com/forum/?fromgroups=#!topic/julia-users/b6RbQ2amKzg
"""
    @lencheck n x y z …

Check that arrays `x`, `y`, `z`, etc. have a prescribed length `n`.
"""
macro lencheck(l, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs, :(
      if length($(esc(var))) != $(esc(l))
        throw(DimensionError($varname, $(esc(l)), length($(esc(var)))))
      end
    ))
  end
  Expr(:block, exprs...)
end

"""
    @rangecheck ℓ u i j k …

Check that values `i`, `j`, `k`, etc. are in the range `[ℓ,u]`.
"""
macro rangecheck(lo, hi, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(
      exprs,
      :(
        if (
          length($(esc(var))) > 0 && (
            any(broadcast(<, $(esc(var)), $(esc(lo)))) ||
            any(broadcast(>, $(esc(var)), $(esc(hi))))
          )
        )
          error(string($varname, " elements must be between ", $(esc(lo)), " and ", $(esc(hi))))
        end
      ),
    )
  end
  Expr(:block, exprs...)
end

"""
    coo_prod!(rows, cols, vals, v, Av)

Compute the product of a matrix `A` given by `(rows, cols, vals)` and the vector `v`.
The result is stored in `Av`, which should have length equals to the number of rows of `A`.
"""
function coo_prod!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  fill!(Av, zero(eltype(v)))
  nnz = length(rows)
  @inbounds for k = 1:nnz
    i, j = rows[k], cols[k]
    Av[i] += vals[k] * v[j]
  end
  return Av
end

"""
    coo_sym_prod!(rows, cols, vals, v, Av)

Compute the product of a symmetric matrix `A` given by `(rows, cols, vals)` and the vector `v`.
The result is stored in `Av`, which should have length equals to the number of rows of `A`.
Only one triangle of `A` should be passed.
"""
function coo_sym_prod!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  fill!(Av, zero(eltype(v)))
  nnz = length(rows)
  @inbounds for k = 1:nnz
    i, j, a = rows[k], cols[k], vals[k]
    Av[i] += a * v[j]
    if i != j
      Av[j] += a * v[i]
    end
  end
  return Av
end

"""
    @default_counters Model inner [excluded]

Define functions relating counters of `Model` to counters of `Model.inner`.
Any function listed in `excluded` (which is an empty list by default), will
not be forwarded.

Examples:

    @default_counters MyModel inner (sum_counters, neval_hprod,)
    @default_counters MyModel inner (neval_hprod,)

Excluding a method from forwarding allows the user to redefine it without
overwriting an existing method. Note that a generic method will still be
defined as, e.g.,

    neval_hprod(model) = model.inner.counters.neval_hprod

because the `counters` attribute itself is forwarded by `@default_counters`.
"""
macro default_counters(Model, inner, excluded = :())

  # Normalize excluded to a set of symbols
  excluded_set = if excluded == :()
    Set{Symbol}()
  elseif excluded isa Expr && excluded.head == :tuple
    Set{Symbol}([excluded.args...])
  else
    throw(ArgumentError("`@default_counters`: third argument must be a tuple of functions"))
  end

  ex = Expr(:block)
  for foo in fieldnames(Counters) ∪ [:sum_counters]
    Symbol(foo) in excluded_set && continue
    push!(ex.args, :(NLPModels.$foo(nlp::$(esc(Model))) = $foo(nlp.$inner)))
  end
  push!(ex.args, :(NLPModels.reset!(nlp::$(esc(Model))) = begin
    reset!(nlp.$inner)
    reset_data!(nlp)
  end))
  push!(ex.args, :(NLPModels.increment!(nlp::$(esc(Model)), s::Symbol) = increment!(nlp.$inner, s)))
  push!(ex.args, :(NLPModels.decrement!(nlp::$(esc(Model)), s::Symbol) = decrement!(nlp.$inner, s)))

  push!(
    ex.args,
    :(
      Base.getproperty(nlp::$(esc(Model)), s::Symbol) =
        (s == :counters ? nlp.$inner.counters : getfield(nlp, s))
    ),
  )
  ex
end

"""
    eltype(nlp::AbstractNLPModel{T, S})

Element type of `nlp.meta.x0`.
"""
Base.eltype(nlp::AbstractNLPModel{T, S}) where {T, S} = eltype(get_x0(nlp))
