"Problem 15 in the Hock-Schittkowski suite"
function hs015()

  nlp = Model()

  @variable(nlp, x[i=1:2])
  setvalue(x[1], -2)
  setvalue(x[2],  1)

  @NLobjective(
    nlp,
    Min,
    100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
  )

  @NLconstraint(
    nlp,
    x[1] * x[2] ≥ 1
  )

  @NLconstraint(
    nlp,
    x[1] + x[2]^2 ≥ 0
  )

  @constraint(
    nlp,
    x[1] ≤ 1/2
  )

  return nlp
end
