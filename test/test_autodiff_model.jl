function test_autodiff_model()
  f1(x) = x[1]^2 + x[2]^2
  f2(x) = x[1] * x[2]
  F(x) = [x[1] - 1; x[2] - x[1]^2]
  A = rand(5,2)
  b = rand(5)
  x0 = zeros(2)
  c(x) = [sum(x) - 1]

  nlp = ADNLPModel([f1, f2], ones(2), F, 2, 1.0, A, b, 1.0, x0,
                   c=c, lcon=[0.0], ucon=[0.0])
  obj(nlp, x0)
  nlp = ADNLPModel(f1, x0)
  obj(nlp, x0)
  nlp = ADNLPModel(F, 2, x0)
  obj(nlp, x0)
  nlp = ADNLPModel(A, b, x0)
  obj(nlp, x0)
end

test_autodiff_model()
