function feasibility_test()
  @testset "feasibility_test" begin
    nlp = ADNLPModel(x->0, zeros(2), c=x->[x[1] - 1; x[2] - x[1]^2],
                     lcon=zeros(2), ucon=zeros(2))
    fnlp = FeasibilityResidual(nlp)
    rnlp = ADNLPModel(x -> [x[1] - 1; x[2] - x[1]^2], 2, zeros(2))

    println("Checking feasibility consistency")
    consistent_nlps([fnlp, rnlp])
  end
end

feasibility_test()
