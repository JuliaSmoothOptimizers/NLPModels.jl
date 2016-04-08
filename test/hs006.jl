"Problem 6 in the Hock-Schittkowski suite"
function hs006()

  nlp = Model()

  @defVar(nlp, x[1:2])
  setValue(x[1], -1.2)
  setValue(x[2],  1.0)

  @setNLObjective(
    nlp,
    Min,
    (1 - x[1])^2
  )

  @addNLConstraint(
    nlp,
    10 * (x[2] - x[1]^2) == 0
  )

  return nlp
end

function hs006_simple()

  x0 = [-1.2; 1.0]
  f(x) = (1 - x[1])^2
  g(x) = [2 * x[1] - 2; 0.0]
  c(x) = [10 * (x[2] - x[1]^2)]
  J(x) = [-20 * x[1]  10]
  H(x,y) = [2.0-20*y[1] 0.0; 0.0 0.0]

  return SimpleNLPModel(x0, f, grad=g, cons=c, jac=J, hess=H, ncon=1)
end
