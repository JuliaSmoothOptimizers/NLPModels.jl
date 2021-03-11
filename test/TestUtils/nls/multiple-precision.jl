export multiple_precision_nls

"""
    multiple_precision_nls(nls; precisions=[...])

Check that the NLS API functions output type are the same as the input.
In other words, make sure that the model handles multiple precisions.

The array `precisions` are the tested floating point types.
Defaults to `[Float16, Float32, Float64, BigFloat]`.
"""
function multiple_precision_nls(nls :: AbstractNLSModel;
                                precisions :: Array = [Float16, Float32, Float64, BigFloat])
  for T in precisions
    x = ones(T, nls.meta.nvar)
    @test eltype(residual(nls, x)) == T
    @test eltype(jac_residual(nls, x)) == T
    @test eltype(jac_op_residual(nls, x)) == T
    rows, cols = jac_structure_residual(nls)
    vals = jac_coord_residual(nls, x)
    @test eltype(vals) == T
    Av = zeros(T, nls.nls_meta.nequ)
    Atv = zeros(T, nls.meta.nvar)
    @test eltype(jac_op_residual!(nls, rows, cols, vals, Av, Atv)) == T
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
  end
end
