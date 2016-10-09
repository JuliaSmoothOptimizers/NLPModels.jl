# Check that arrays have a prescribed size.
# https://groups.google.com/forum/?fromgroups=#!topic/julia-users/b6RbQ2amKzg
macro lencheck(l, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs,
          :(if length($var) != $l
              error(string($varname, " must have length ", $l))
            end))
  end
  Expr(:block, exprs...)
end

macro rangecheck(lo, hi, vars...)
  exprs = Expr[]
  for var in vars
    varname = string(var)
    push!(exprs,
          :(if (length($var) > 0 && (any($var .< $lo) || any($var .> $hi)))
              error(string($varname, " elements must be between ", $lo, " and ", $hi))
            end))
  end
  Expr(:block, exprs...)
end
