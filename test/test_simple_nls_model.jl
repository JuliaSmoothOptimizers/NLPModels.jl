function simple_nls_test()
  @testset "simple_nls_test" begin
    for A = [rand(10, 3), sprand(10, 3, 0.5)]
      b = rand(10)
      F(x) = A * x - b
      JF(x) = A
      nls = SimpleNLSModel(3, 10, F=F, JF=JF, Hi=(x,i)->zeros(3,3))

      x = rand(3)
      @test isapprox(residual(nls, x), A * x - b, rtol=1e-8)

      I, J, V = hess_coord(nls, x)
      @test sparse(I, J, V) == tril(A' * A)
    end
  end
end

simple_nls_test()
