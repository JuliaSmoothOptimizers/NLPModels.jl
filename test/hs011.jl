"Problem 11 in the Hock-Schittkowski suite"
function hs011()

  nlp = Model()

  @variable(nlp, x[i=1:2])
  setvalue(x[1], 4.9)
  setvalue(x[2], 0.1)

  @NLobjective(
    nlp,
    Min,
    (x[1] - 5)^2 + x[2]^2 - 25
  )

  @NLconstraint(
    nlp,
    x[1]^2 <= x[2]
  )

  return nlp
end

function hs011_simple()

  x0 = [4.9; 0.1]
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  g(x) = [2 * (x[1] - 5); 2 * x[2]]
  c(x) = [x[1]^2 - x[2]]
  J(x) = [2*x[1]  -1]
  H(x,y) = [2+2*y[1] 0.0; 0.0 2.0]
  lcon = [-Inf]
  ucon = [0.0]

  return SimpleNLPModel(x0, f, grad=g, cons=c, jac=J, hess=H, lcon=lcon,
    ucon=ucon)

end
