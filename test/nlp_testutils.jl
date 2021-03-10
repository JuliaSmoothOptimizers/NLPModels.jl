for problem in problems
  @testset "Checking TestUtils tests on problem $problem" begin
    nlp_ad = eval(Meta.parse("$(problem)_autodiff"))()
    nlp_man = eval(problem)()

    check_nlp_dimensions(nlp_ad)
    multiple_precision_nlp(nlp_ad)
    view_subarray_nlp(nlp_ad)
    coord_memory_nlp(nlp_ad)

    nlps = [nlp_ad, nlp_man]
    reset!.(nlps)
    consistent_nlps(nlps)

    if has_inequalities(nlp_ad)
      check_nlp_dimensions(SlackModel(nlp_ad))
      multiple_precision_nlp(SlackModel(nlp_ad))
      view_subarray_nlp(SlackModel(nlp_ad))
      coord_memory_nlp(SlackModel(nlp_ad))
      reset!.(nlps)
      consistent_nlps(SlackModel.(nlps))
    end
  end
end