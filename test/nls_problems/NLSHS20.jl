function NLSHS20_autodiff()

  x0 = [-2.0; 1.0]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  lvar = [-0.5; -Inf]
  uvar = [0.5; Inf]
  c(x) = [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  lcon = zeros(3)
  ucon = fill(Inf, 3)

  return ADNLSModel(F, x0, 2, lvar, uvar, c, lcon, ucon, name="NLSHS20_autodiff")
end
