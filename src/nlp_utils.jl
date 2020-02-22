export coo_prod!, coo_sym_prod!

# Check that arrays have a prescribed size.
# https://groups.google.com/forum/?fromgroups=#!topic/julia-users/b6RbQ2amKzg
macro lencheck(l, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs,
          :(if length($(esc(var))) != $(esc(l))
                error(string($varname, " must have length ", $(esc(l))))
            end))
  end
  Expr(:block, exprs...)
end

macro rangecheck(lo, hi, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs,
          :(if (length($(esc(var))) > 0 && (any(broadcast(<, $(esc(var)), $(esc(lo)))) || any(broadcast(>, $(esc(var)), $(esc(hi))))))
            error(string($varname, " elements must be between ", $(esc(lo)), " and ", $(esc(hi))))
            end))
  end
  Expr(:block, exprs...)
end

"""
    coo_prod!(rows, cols, vals, v, Av)

Compute the product of a matrix `A` given by `(rows, cols, vals)` and the vector `v`.
The result is stored in `Av`, which should have length equals to the number of rows of `A`.
"""
function coo_prod!(rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector, v :: AbstractVector, Av :: AbstractVector)
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
function coo_sym_prod!(rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector, v :: AbstractVector, Av :: AbstractVector)
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
