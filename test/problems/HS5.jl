function HS5_autodiff()

  x0 = [0.0; 0.0]
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
  l = [-1.5; -3.0]
  u = [4.0; 3.0]

  return ADNLPModel(f, x0, l, u, name="HS5_autodiff")
end
