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
