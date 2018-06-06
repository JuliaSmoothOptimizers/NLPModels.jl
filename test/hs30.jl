# HS problem 30 in NLS format
#
#   Source:
#   W. Hock and K. Schittkowski,
#   Test examples for nonlinear programming codes,
#   Lecture Notes in Economics and Mathematical Systems 187,
#   Springer Verlag Berlin Heidelberg, 1981
#   10.1007/978-3-642-48320-2
#
# A. S. Siqueira, Curitiba/BR, 04/2018.

export hs30

"Hock-Schittkowski problem 30 in NLS format"
function hs30()

  model = Model()
  lvar = [1.0; -10.0; -10.0]
  @variable(model, lvar[i] <= x[i=1:3] <= 10, start=1.0)
  @NLexpression(model, F[i=1:3], x[i] + 0.0)
  @NLconstraint(model, x[1]^2 + x[2]^2 >= 1.0)

  return MathProgNLSModel(model, F, name="hs30")
end
