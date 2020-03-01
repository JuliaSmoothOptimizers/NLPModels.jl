function multiple_precision(nlp :: AbstractNLPModel;
                            precisions :: Array = [Float16, Float32, Float64, BigFloat])
  print("Testing NLP with types: ")
  for T in precisions
    print("$T ")
    x = ones(T, nlp.meta.nvar)
    @test typeof(obj(nlp, x)) == T
    @test eltype(grad(nlp, x)) == T
    @test eltype(hess(nlp, x)) == T
    @test eltype(hess_op(nlp, x)) == T
    rows, cols = hess_structure(nlp)
    vals = hess_coord(nlp, x)
    @test eltype(vals) == T
    Hv = zeros(T, nlp.meta.nvar)
    @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    if nlp.meta.ncon > 0
      y = ones(T, nlp.meta.ncon)
      @test eltype(cons(nlp, x)) == T
      @test eltype(jac(nlp, x)) == T
      @test eltype(jac_op(nlp, x)) == T
      rows, cols = jac_structure(nlp)
      vals = jac_coord(nlp, x)
      @test eltype(vals) == T
      Av = zeros(T, nlp.meta.ncon)
      Atv = zeros(T, nlp.meta.nvar)
      @test eltype(jac_op!(nlp, rows, cols, vals, Av, Atv)) == T
      @test eltype(hess(nlp, x, y)) == T
      @test eltype(hess(nlp, x, y, obj_weight=one(T))) == T
      @test eltype(hess_op(nlp, x, y)) == T
      rows, cols = hess_structure(nlp)
      vals = hess_coord(nlp, x, y)
      @test eltype(vals) == T
      Hv = zeros(T, nlp.meta.nvar)
      @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    end
    print("✓ ")
  end
  println("")
end

function multiple_precision(nls :: AbstractNLSModel;
                            precisions :: Array = [Float16, Float32, Float64, BigFloat])
  print("Testing NLS with types: ")
  for T in precisions
    print("$T ")
    x = ones(T, nls.meta.nvar)
    @test eltype(residual(nls, x)) == T
    @test eltype(jac_residual(nls, x)) == T
    @test eltype(jac_op_residual(nls, x)) == T
    rows, cols = jac_structure_residual(nls)
    vals = jac_coord_residual(nls, x)
    @test eltype(vals) == T
    Av = zeros(T, nls.nls_meta.nequ)
    Atv = zeros(T, nls.meta.nvar)
    @test eltype(jac_op!(nls, rows, cols, vals, Av, Atv)) == T
    @test eltype(hess_residual(nls, x, ones(T, nls.nls_meta.nequ))) == T
    for i = 1:nls.nls_meta.nequ
      @test eltype(hess_op_residual(nls, x, i)) == T
    end
    @test typeof(obj(nls, x)) == T
    @test eltype(grad(nls, x)) == T
    if nls.meta.ncon > 0
      @test eltype(cons(nls, x)) == T
      @test eltype(jac(nls, x)) == T
      @test eltype(jac_op(nls, x)) == T
      rows, cols = jac_structure(nls)
      vals = jac_coord(nls, x)
      @test eltype(vals) == T
      Av = zeros(T, nls.meta.ncon)
      Atv = zeros(T, nls.meta.nvar)
      @test eltype(jac_op!(nls, rows, cols, vals, Av, Atv)) == T
    end
    print("✓ ")
  end
  println("")
end
