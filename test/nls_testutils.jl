for problem in nls_problems
  @testset "Checking TestUtils tests on problem $problem" begin
    nls_ad = eval(Meta.parse("$(problem)_autodiff"))()
    nls_man = eval(problem)()

    nlss = AbstractNLSModel[nls_ad]
    # *_special problems are variant definitions of a model
    spc = "$(problem)_special"
    if isdefined(Main, Symbol(spc))
      push!(nlss, eval(Meta.parse(spc))())
    end

    for nls in nlss
      check_nls_dimensions(nls)
      check_nlp_dimensions(nls, exclude_hess=true)
      if typeof(nls) != LLSModel
        multiple_precision_nls(nls)
      end
      view_subarray_nls(nls)
    end

    if has_inequalities(nls_ad)
      for nls in nlss
        check_nls_dimensions(SlackNLSModel(nls))
        check_nlp_dimensions(SlackNLSModel(nls), exclude_hess=true)
        if typeof(nls) != LLSModel
          multiple_precision_nls(SlackNLSModel(nls))
        end
        view_subarray_nls(SlackNLSModel(nls))
      end
    end

    push!(nlss, nls_man)
    reset!.(nlss)
    consistent_nlss(nlss)
    reset!.(nlss)
    if has_inequalities(nls_ad)
      consistent_nlss(SlackNLSModel.(nlss))
    end
    reset!.(nlss)
    consistent_nlss(FeasibilityFormNLS.(nlss))
  end
end