function autodiff_nls_test()
  @testset "autodiff_nls_test" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2]
    nls = ADNLSModel(F, zeros(2), 2)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol=1e-8)
  end

  @testset "Constructors for ADNLSModel" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2; x[1] * x[2]]
    x0 = ones(2)
    c(x) = [sum(x) - 1]
    lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
    badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
    nlp = ADNLSModel(F, x0, 3)
    nlp = ADNLSModel(F, x0, 3, lvar, uvar)
    nlp = ADNLSModel(F, x0, 3, c, lcon, ucon)
    nlp = ADNLSModel(F, x0, 3, c, lcon, ucon, y0=y0)
    nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon)
    nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, y0=y0)
    @test_throws DimensionError ADNLSModel(F, x0, 3, badlvar, uvar)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, baduvar)
    @test_throws DimensionError ADNLSModel(F, x0, 3, c, badlcon, ucon)
    @test_throws DimensionError ADNLSModel(F, x0, 3, c, lcon, baducon)
    @test_throws DimensionError ADNLSModel(F, x0, 3, c, lcon, ucon, y0=bady0)
    @test_throws DimensionError ADNLSModel(F, x0, 3, badlvar, uvar, c, lcon, ucon)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, baduvar, c, lcon, ucon)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, uvar, c, badlcon, ucon)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, baducon)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, y0=bady0)

  end
end

autodiff_nls_test()
