function MGH01_autodiff()

  x0 = [-1.2; 1.0]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

  return ADNLSModel(F, x0, 2, name="MGH01_autodiff")
end

MGH01_special() = FeasibilityResidual(MGH01FEAS())
