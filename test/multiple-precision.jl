function multiple_precision(nlp :: AbstractNLPModel;
                            precisions :: Array = [Float16, Float32, Float64, BigFloat])
  for T in precisions
    x = ones(T, nlp.nvar)
    @test typeof(obj(nlp, x)) == T
    @test eltype(grad(nlp, x)) == T
    @test eltype(hess(nlp, x)) == T
    if nlp.ncon > 0
      @test eltype(cons(nlp, x)) == T
      @test eltype(jac(nlp, x)) == T
      @test eltype(hess(nlp, x, ones(T, nlp.ncon))) == T
      @test eltype(hess(nlp, x, ones(T, nlp.ncon), obj_weight=one(T))) == T
    end
  end
end

function multiple_precision(nls :: AbstractNLSModel;
                            precisions :: Array = [Float16, Float32, Float64, BigFloat])
  for T in precisions
    x = ones(T, nls.nvar)
    @test eltype(residual(nls, x)) == T
    @test eltype(jac_residual(nls, x)) == T
    @test eltype(hess_residual(nls, x, ones(T, nls.nls_nequ))) == T
    @test typeof(obj(nls, x)) == T
    @test eltype(grad(nls, x)) == T
    if nls.ncon > 0
      @test eltype(cons(nls, x)) == T
      @test eltype(jac(nls, x)) == T
    end
  end
end
