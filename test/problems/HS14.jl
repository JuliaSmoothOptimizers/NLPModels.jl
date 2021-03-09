function HS14_autodiff()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  lcon = [0.0; 0.0]
  ucon = [0.0; Inf]

  return ADNLPModel(f, x0, c, lcon, ucon, name="HS14_autodiff")
end
