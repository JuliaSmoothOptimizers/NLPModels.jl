function simple_nls_test()
  @testset "simple_nls_test" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2]
    JF(x) = [1.0 0.0; -2*x[1] 1.0]
    nls = SimpleNLSModel(2, 2, F=F, JF=JF)

    @test residual(nls, ones(2)) == zeros(2)
  end
end

simple_nls_test()
