function viability_nls_test()
  @testset "viability_nls_test" begin
    nlp = ADNLPModel(x->0, zeros(2), c=x->[x[1] - 1; x[2] - x[1]^2],
                     lcon=zeros(2), ucon=zeros(2))
    nls = ViabilityModel(nlp)

    @test residual(nls, ones(2)) == zeros(2)
  end
end

viability_nls_test()
