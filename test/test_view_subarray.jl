function test_view_subarray()
  adnlp = ADNLPModel(x -> sum(x.^4), ones(3), c=x -> [dot(x, x) - 4.0; sum(x) - 1.0])
  x = rand(5)
  nlps = [adnlp]
  functions = [obj, grad, hess, cons, jac]
  for nlp in nlps, foo in functions, I = [1:3, 1:2:5, [2;4;5], [4;1;3]]
    @test foo(nlp, x[I]) == foo(nlp, @view x[I])
  end
end

test_view_subarray()
