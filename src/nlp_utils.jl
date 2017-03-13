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
