function test_simple_model()
  x0 = zeros(2)
  f(x) = dot(x,x)
  nlp = SimpleNLPModel(x0, f)

  c(x) = [sum(x) - 1]
  nlp = SimpleNLPModel(x0, f, cons=c)
end

test_simple_model()
