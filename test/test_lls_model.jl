function lls_test()
  @testset "lls_test" begin
    A = rand(10, 3)
    b = rand(10)
    nls = LLSModel(A, b)
    x = rand(3)

    @test isapprox(A * x - b, residual(nls, x), rtol=1e-8)
    @test isapprox(A, jac_residual(nls, x), rtol=1e-8)
    for i = 1:10
      @test isapprox(zeros(3, 3), hess_residual(nls, x, i), rtol=1e-8)
    end
  end
end

lls_test()
