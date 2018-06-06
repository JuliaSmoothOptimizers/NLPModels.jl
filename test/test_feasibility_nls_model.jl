function feasibility_nls_test()
  @testset "feasibility_nls_test" begin
    nlp = ADNLPModel(x->0, zeros(2), c=x->[x[1] - 1; x[2] - x[1]^2],
                     lcon=zeros(2), ucon=zeros(2))
    nls = FeasibilityResidual(nlp)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol=1e-8)
  end
end

feasibility_nls_test()
