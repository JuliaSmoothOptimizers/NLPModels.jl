function MGH01FEAS_autodiff()

  x0 = [-1.2; 1.0]
  c(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

  return ADNLPModel(x->zero(eltype(x)), x0, c, zeros(2), zeros(2), name="MGH01FEAS_autodiff")
end