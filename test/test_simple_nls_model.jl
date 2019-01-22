function simple_nls_test()
  @testset "simple_nls_test" begin
    for A = [Matrix(1.0I, 10, 3) .+ 1,
             sparse(1.0I, 10, 3) .+ 1]
      b = ones(10)
      F(x) = A * x - b
      JF(x) = A
      nls = SimpleNLSModel(3, 10, F=F, JF=JF, H=(x,v)->zeros(3,3), Hi=(x,i)->zeros(3,3))

      x = [1.0; -1.0; 1.0]
      @test isapprox(residual(nls, x), A * x - b, rtol=1e-8)

      I, J, V = hess_coord(nls, x)
      @test sparse(I, J, V) == tril(A' * A)
    end
  end
end

simple_nls_test()
