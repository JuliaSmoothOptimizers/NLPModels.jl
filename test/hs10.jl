#Problem 10 in the Hock-Schittkowski suite
function hs10_autodiff()

  x0 = [-10.0; 10.0]
  f(x) = x[1] - x[2]
  c(x) = [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1.0]
  lcon = [0.0]
  ucon = [Inf]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
end
