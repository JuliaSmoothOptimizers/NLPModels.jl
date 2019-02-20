function lls_test()
  @testset "lls_test" begin
    for A = [Matrix(1.0I, 10, 3) .+ 1, sparse(1.0I, 10, 3) .+ 1],
        C = [ones(1, 3), [ones(1,3); -I], sparse(ones(1,3))]
      b = collect(1:10)
      nequ, nvar = size(A)
      ncon = size(C,1)
      nls = LLSModel(A, b, C=C, lcon=zeros(ncon), ucon=zeros(ncon))
      x = [1.0; -1.0; 1.0]

      @test isapprox(A * x - b, residual(nls, x), rtol=1e-8)
      @test A == jac_residual(nls, x)
      @test A == sparse(jac_coord_residual(nls, x)..., nequ, nvar)
      @test sparse(hess_coord_residual(nls, x, ones(nequ))..., nvar, nvar) == zeros(nvar, nvar)
      @test hess_residual(nls, x, ones(10)) == zeros(3,3)
      for i = 1:10
        @test isapprox(zeros(3, 3), jth_hess_residual(nls, x, i), rtol=1e-8)
      end

      I, J, V = jac_coord(nls, x)
      @test sparse(I, J, V, ncon, nvar) == C

      @test nls.meta.nlin == length(nls.meta.lin) == ncon
      @test nls.meta.nnln == length(nls.meta.nln) == 0
    end
  end
end

lls_test()
