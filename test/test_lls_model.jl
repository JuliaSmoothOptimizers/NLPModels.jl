function lls_test()
  @testset "lls_test" begin
    for A = [rand(10, 3), sprand(10, 3, 0.5)],
        C = [rand(1, 3), rand(4,3), sprand(1, 3, 1.0)]
      b = rand(10)
      ncon = size(C,1)
      nls = LLSModel(A, b, C=C, lcon=zeros(ncon), ucon=zeros(ncon))
      x = rand(3)

      @test isapprox(A * x - b, residual(nls, x), rtol=1e-8)
      @test A == jac_residual(nls, x)
      for i = 1:10
        @test isapprox(zeros(3, 3), hess_residual(nls, x, i), rtol=1e-8)
      end

      I, J, V = jac_coord(nls, x)
      @test sparse(I, J, V) == C

      I, J, V = hess_coord(nls, x)
      @test sparse(I, J, V) == tril(A' * A)

      @test nls.meta.nlin == length(nls.meta.lin) == ncon
      @test nls.meta.nnln == length(nls.meta.nln) == 0
    end
  end
end

lls_test()
