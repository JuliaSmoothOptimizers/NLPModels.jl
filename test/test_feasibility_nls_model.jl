function feasibility_nls_test()
  @testset "feasibility_nls_test" begin
    nlp = ADNLPModel(x->0, zeros(2), c=x->[x[1] - 1; x[2] - x[1]^2],
                     lcon=zeros(2), ucon=zeros(2))
    nls = FeasibilityResidual(nlp)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol=1e-8)

    nlp = ADNLPModel(x->0, zeros(2), c=x->[x[1] - 1; x[2] - x[1]^2],
                     lvar=[-0.3; -0.5], uvar=[1.2; 3.4],
                     lcon=-ones(2), ucon=2*ones(2))
    nls = FeasibilityResidual(nlp)

    @test nls.meta.nvar == 4
    @test nls.nls_meta.nequ == 2
    @test nls.meta.lvar == [-0.3; -0.5; -1.0; -1.0]
    @test nls.meta.uvar == [ 1.2;  3.4;  2.0;  2.0]
    @test isapprox(residual(nls, [1.0; 1.0; 0.0; 0.0]), zeros(2), rtol=1e-8)
    @test isapprox(residual(nls, [0.0; 1.0; 2.0; 3.0]), [-3.0; -2.0], rtol=1e-8)
  end
end

feasibility_nls_test()
