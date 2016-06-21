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
    -x[1]^2/4 - x[2]^2 + 1 >= 0
  )

  @constraint(
    nlp,
    x[1] - 2 * x[2] + 1 == 0
  )

  return nlp
end

function hs014_simple()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [-x[1]^2/4 - x[2]^2 + 1; x[1] - 2 * x[2] + 1]
  lcon = [0.0; 0.0]
  ucon = [Inf; 0.0]

  return SimpleNLPModel(x0, f, c=c, lcon=lcon, ucon=ucon)
end
cutest_problem_name = "HS14"
