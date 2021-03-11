for problem in TestUtils.nlp_problems
  @testset "Checking TestUtils tests on problem $problem" begin
    nlp_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
    nlp_man = eval(Meta.parse(problem))()

    show(IOBuffer(), nlp_ad)

    nlps = [nlp_ad, nlp_man]
    @testset "Check Consistency" begin
      consistent_nlps(nlps)
    end
    @testset "Check dimensions" begin
      check_nlp_dimensions(nlp_ad)
    end
    @testset "Check multiple precision" begin
      multiple_precision_nlp(nlp_ad)
    end
    @testset "Check view subarray" begin
      view_subarray_nlp(nlp_ad)
    end
    @testset "Check coordinate memory" begin
      coord_memory_nlp(nlp_ad)
    end
  end
end