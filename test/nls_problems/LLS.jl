function LLS_autodiff()

  x0 = [0.0; 0.0]
  F(x) = [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  c(x) = [x[1] + x[2]]
  lcon = [0.0]
  ucon = [Inf]

  return ADNLSModel(F, x0, 3, c, lcon, ucon, name="LLS_autodiff")
end

function LLS_special()
  return LLSModel([1.0 -1; 1 1; 0 1], [0.0; 2; 2], x0=zeros(2), C=[1.0 1], lcon=[0.0], ucon=[Inf], name="LLS_LLSModel")
end