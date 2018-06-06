# MGH problem 35 - Chebyquad function
#
#   Source:
#   J. J. Moré, B. S. Garbow and K. E. Hillstrom
#   Testing Unconstrained Optimization Software
#   ACM Transactions on Mathematical Software, 7(1):17-41, 1981
#
# A. Montoison, Montreal, 05/2018.

export mgh35

"Chebyquad function"
function mgh35(m :: Int = 10, n :: Int = 10)
  if m < n
    warn(": number of function must be ≥ number of variables. Adjusting to m = n")
    m = n
  end

  nls  = Model()
  x0   = collect(1:n)/(n+1)
  @variable(nls, x[i=1:n], start = x0[i])

  function Tsim(x, n)
    if n == 0
      return 1
    elseif n == 1
      return x
    else
      return 2*x*Tsim(x,n-1) - Tsim(x,n-2)
    end
  end

  Ts = Vector{Function}(n)
  Tnames = Vector{Symbol}(n)
  for i = 1:n
    Ts[i] = x -> Tsim(2*x-1, i)
    Tnames[i] = gensym()
    JuMP.register(nls, Tnames[i], 1, Ts[i], autodiff=true)
  end

  I = [i%2 == 0 ? -1/(i^2-1) : 0 for i = 1:n]
  @NLexpression(nls, F[i=1:n], sum($(Tnames[i])(x[j]) for j = 1:n)/n - I[i])

  return MathProgNLSModel(nls, F, name="mgh35")
end