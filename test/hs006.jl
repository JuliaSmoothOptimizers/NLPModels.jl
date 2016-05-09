"Problem 6 in the Hock-Schittkowski suite"
function hs006()

  nlp = Model()

  @variable(nlp, x[1:2])
  setvalue(x[1], -1.2)
  setvalue(x[2],  1.0)

  @NLobjective(
    nlp,
    Min,
    (1 - x[1])^2
  )

  @NLconstraint(
    nlp,
    10 * (x[2] - x[1]^2) == 0
  )

  return nlp
end
