"Problem 14 in the Hock-Schittkowski suite"
function hs014()

  nlp = Model()

  @variable(nlp, x[i=1:2])
  setvalue(x[1], 2)
  setvalue(x[2], 2)

  @NLobjective(
    nlp,
    Min,
    (x[1] - 2)^2 + (x[2] - 1)^2
  )

  @NLconstraint(
    nlp,
    x[1]^2/4 + x[2]^2 ≤ 1
  )

  @constraint(
    nlp,
    x[1] - 2 * x[2] == -1
  )

  return nlp
end

function hs014_simple()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  g(x) = [2 * (x[1] - 2); 2 * (x[2] - 1)]
  c(x) = [x[1]^2/4 + x[2]^2 - 1; x[1] - 2 * x[2] + 1]
  J(x) = [x[1]/2  2*x[2]; 1.0  -2.0]
  H(x,y) = [2.0 + y[1]/2 0.0; 0.0 2.0 + 2 * y[1]]
  lcon = [-Inf; 0.0]
  ucon = [0.0; 0.0]

  return SimpleNLPModel(x0, f, grad=g, cons=c, jac=J, hess=H, lcon=lcon,
    ucon=ucon)
end
