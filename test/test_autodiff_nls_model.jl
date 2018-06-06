function autodiff_nls_test()
  @testset "autodiff_nls_test" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2]
    nls = ADNLSModel(F, 2, 2)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol=1e-8)
  end
end

autodiff_nls_test()
