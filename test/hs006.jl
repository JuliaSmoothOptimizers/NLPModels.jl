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
