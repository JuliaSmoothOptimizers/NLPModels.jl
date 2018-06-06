# MGH problem 7 - Helical valley function
#
#  Source:
#  J. J. Moré, B. S. Garbow and K. E. Hillstrom
#  Testing Unconstrained Optimization Software
#  ACM Transactions on Mathematical Software, 7(1):17-41, 1981
#
# A. Montoison, Montreal, 05/2018.

export mgh07

"Helical valley function"
function mgh07()

  nls  = Model()
  x0   = [-1.0,  0.0,  0.0]
  @variable(nls, x[i=1:3], start = x0[i])

  θ_aux(t) = (t > 0 ? 0.0 : 0.5)
  JuMP.register(nls, :θ_aux, 1, θ_aux, autodiff=true)

  @NLexpression(nls, F1, 10*(x[3] - 10*(atan(x[2]/x[1])/(2*π) + θ_aux(x[1]))))
  @NLexpression(nls, F2, 10*(sqrt(x[1]^2 + x[2]^2) - 1.0))
  @NLexpression(nls, F3, 1*x[3])
  
  return MathProgNLSModel(nls, [F1, F2, F3], name="mgh07")
end