function lls_test()
  @testset "lls_test" begin
    A = rand(10, 3)
    b = rand(10)
    nls = LLSModel(A, b)
    x = rand(3)

    @test A * x - b == residual(nls, x)
    @test A == jac_residual(nls, x)
    for i = 1:10
      @test zeros(3, 3) == hess_residual(nls, x, i)
    end
  end
end

lls_test()
