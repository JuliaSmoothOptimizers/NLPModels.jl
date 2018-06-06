# HS problem 43 in NLS format without constants in the objective
#
#   Source:
#   W. Hock and K. Schittkowski,
#   Test examples for nonlinear programming codes,
#   Lecture Notes in Economics and Mathematical Systems 187,
#   Springer Verlag Berlin Heidelberg, 1981
#   10.1007/978-3-642-48320-2
#
# A. S. Siqueira, Curitiba/BR, 04/2018.

export hs43

"Hock-Schittkowski problem 43 in NLS format without constants in the objective"
function hs43()

  model = Model()
  @variable(model, x[1:4], start=0.0)
  @NLexpression(model, F1, x[1] - 5/2)
  @NLexpression(model, F2, x[2] - 5/2)
  @NLexpression(model, F3, sqrt(2) * (x[3] - 21 / 4))
  @NLexpression(model, F4, x[4] + 7/2)
  @NLconstraint(model, 8 - x[1]^2 - x[2]^2 - x[3]^2 - x[4]^2 - x[1] +
                x[2] - x[3] + x[4] >= 0.0)
  @NLconstraint(model, 10 - x[1]^2 - 2 * x[2]^2 - x[3]^2 - 2 * x[4]^2 +
                x[1] + x[4] >= 0.0)
  @NLconstraint(model, 5 - 2 * x[1]^2 - x[2]^2 - x[3]^2 - 2 * x[1] +
                x[2] + x[4] >= 0.0)

  return MathProgNLSModel(model, [F1; F2; F3; F4], name="hs43")
end
