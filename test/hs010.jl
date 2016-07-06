"Problem 10 in the Hock-Schittkowski suite"
function hs010()

  nlp = Model()

  @variable(nlp, x[i=1:2])
  setvalue(x[1], -10)
  setvalue(x[2],  10)

  @NLobjective(
    nlp,
    Min,
    x[1] - x[2]
  )

  @NLconstraint(
    nlp,
    -3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 ≥ -1
  )

  return nlp
end

function hs010_simple()

  x0 = [-10.0; 10.0]
  f(x) = x[1] - x[2]
  c(x) = [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2]
  lcon = [-1.0]
  ucon = [Inf]

  return SimpleNLPModel(x0, f, c=c, lcon=lcon, ucon=ucon)
end
