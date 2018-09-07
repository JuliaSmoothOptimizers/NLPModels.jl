function test_view_subarray()
  snlp = SimpleNLPModel(x -> sum(x.^4), ones(3),
                        g=x -> 4x.^3,
                        c=x -> [dot(x, x) - 4.0; sum(x) - 1.0],
                        J=x -> [2x[1]  2x[2]  2x[3]; ones(1,3)],
                        H=(x;y=zeros(2),obj_weight=1.0) -> sparse([1, 2, 3], [1, 2, 3],
                                                                  12obj_weight * x.^2 +
                                                                  2 * y[1] * ones(3)),
                        lcon=zeros(2), ucon=zeros(2))
  adnlp = ADNLPModel(x -> sum(x.^4), ones(3), c=x -> [dot(x, x) - 4.0; sum(x) - 1.0])
  x = rand(5)
  nlps = [snlp, adnlp]
  functions = [obj, grad, hess, cons, jac]
  for nlp in nlps, foo in functions, I = [1:3, 1:2:5, [2;4;5], [4;1;3]]
    @test foo(nlp, x[I]) == foo(nlp, @view x[I])
  end
end

test_view_subarray()
