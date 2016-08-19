"Problem 5 in the Hock-Schittkowski suite"
function hs5()

  nlp = Model()

  l = [-1.5, -3]
  u = [4, 3]
  @variable(nlp, l[i] ≤ x[i=1:2] ≤ u[i], start=0.0)

  @NLobjective(
    nlp,
    Min,
    sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
  )

  return nlp
end

function hs5_simple()

  x0 = [0.0; 0.0]
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
  l = [-1.5; -3.0]
  u = [4.0; 3.0]

  return SimpleNLPModel(f, x0, lvar=l, uvar=u)

end
