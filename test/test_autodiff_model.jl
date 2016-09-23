function test_autodiff_model()
  x0 = zeros(2)
  f(x) = dot(x,x)
  nlp = ADNLPModel(f, x0)

  c(x) = [sum(x) - 1]
  nlp = ADNLPModel(f, x0, c=c, lcon=[0], ucon=[0])
end

test_autodiff_model()
