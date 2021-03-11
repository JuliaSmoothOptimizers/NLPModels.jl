export check_nlp_dimensions

"""
    check_nlp_dimensions(nlp; exclude_hess=false)

Make sure NLP API functions will throw DimensionError if the inputs are not the correct dimension.
To make this assertion in your code use

    @lencheck size input [more inputs separated by spaces]
"""
function check_nlp_dimensions(nlp; exclude_hess=false)
  n, m = nlp.meta.nvar, nlp.meta.ncon
  nnzh, nnzj = nlp.meta.nnzh, nlp.meta.nnzj

  x, badx   = nlp.meta.x0, zeros(n + 1)
  v, badv   = ones(n), zeros(n + 1)
  Hv, badHv = zeros(n), zeros(n + 1)
  hrows, badhrows = zeros(Int, nnzh), zeros(Int, nnzh + 1)
  hcols, badhcols = zeros(Int, nnzh), zeros(Int, nnzh + 1)
  hvals, badhvals = zeros(nnzh), zeros(nnzh + 1)
  @test_throws DimensionError obj(nlp, badx)
  @test_throws DimensionError grad(nlp, badx)
  @test_throws DimensionError grad!(nlp, badx, v)
  @test_throws DimensionError grad!(nlp, x, badv)
  @test_throws DimensionError hprod(nlp, badx, v)
  @test_throws DimensionError hprod(nlp, x, badv)
  @test_throws DimensionError hprod!(nlp, badx, v, Hv)
  @test_throws DimensionError hprod!(nlp, x, badv, Hv)
  @test_throws DimensionError hprod!(nlp, x, v, badHv)
  @test_throws DimensionError hess_op(nlp, badx)
  @test_throws DimensionError hess_op!(nlp, badx, Hv)
  @test_throws DimensionError hess_op!(nlp, x, badHv)
  @test_throws DimensionError hess_op!(nlp, badhrows, hcols, hvals, Hv)
  @test_throws DimensionError hess_op!(nlp, hrows, badhcols, hvals, Hv)
  @test_throws DimensionError hess_op!(nlp, hrows, hcols, badhvals, Hv)
  @test_throws DimensionError hess_op!(nlp, hrows, hcols, hvals, badHv)
  if !exclude_hess
    @test_throws DimensionError hess(nlp, badx)
    @test_throws DimensionError hess_structure!(nlp, badhrows, hcols)
    @test_throws DimensionError hess_structure!(nlp, hrows, badhcols)
    @test_throws DimensionError hess_coord!(nlp, badx, hvals)
    @test_throws DimensionError hess_coord!(nlp, x, badhvals)
  end

  if m > 0
    y, bady     = nlp.meta.y0, zeros(m + 1)
    w, badw     = ones(m), zeros(m + 1)
    Jv, badJv   = zeros(m), zeros(m + 1)
    Jtw, badJtw = zeros(n), zeros(n + 1)
    jrows, badjrows = zeros(Int, nnzj), zeros(Int, nnzj + 1)
    jcols, badjcols = zeros(Int, nnzj), zeros(Int, nnzj + 1)
    jvals, badjvals = zeros(nnzj), zeros(nnzj + 1)
    @test_throws DimensionError hprod(nlp, badx, y, v)
    @test_throws DimensionError hprod(nlp, x, bady, v)
    @test_throws DimensionError hprod(nlp, x, y, badv)
    if !exclude_hess
      @test_throws DimensionError hprod!(nlp, badx, y, v, Hv)
      @test_throws DimensionError hprod!(nlp, x, bady, v, Hv)
      @test_throws DimensionError hprod!(nlp, x, y, badv, Hv)
      @test_throws DimensionError hprod!(nlp, x, y, v, badHv)
      @test_throws DimensionError hess(nlp, badx, y)
      @test_throws DimensionError hess(nlp, x, bady)
      @test_throws DimensionError hess_op(nlp, badx, y)
      @test_throws DimensionError hess_op(nlp, x, bady)
      @test_throws DimensionError hess_op!(nlp, badx, y, Hv)
      @test_throws DimensionError hess_op!(nlp, x, bady, Hv)
      @test_throws DimensionError hess_op!(nlp, x, y, badHv)
      @test_throws DimensionError hess_coord!(nlp, badx, y, hvals)
      @test_throws DimensionError hess_coord!(nlp, x, bady, hvals)
      @test_throws DimensionError hess_coord!(nlp, x, y, badhvals)
      @test_throws DimensionError ghjvprod(nlp, badx, v, v)
      @test_throws DimensionError ghjvprod(nlp, x, badv, v)
      @test_throws DimensionError ghjvprod(nlp, x, v, badv)
    end
    @test_throws DimensionError cons(nlp, badx)
    @test_throws DimensionError cons!(nlp, badx, w)
    @test_throws DimensionError cons!(nlp, x, badw)
    @test_throws DimensionError jac(nlp, badx)
    @test_throws DimensionError jprod(nlp, badx, v)
    @test_throws DimensionError jprod(nlp, x, badv)
    @test_throws DimensionError jprod!(nlp, badx, v, Jv)
    @test_throws DimensionError jprod!(nlp, x, badv, Jv)
    @test_throws DimensionError jprod!(nlp, x, v, badJv)
    @test_throws DimensionError jtprod(nlp, badx, w)
    @test_throws DimensionError jtprod(nlp, x, badw)
    @test_throws DimensionError jtprod!(nlp, badx, w, Jtw)
    @test_throws DimensionError jtprod!(nlp, x, badw, Jtw)
    @test_throws DimensionError jtprod!(nlp, x, w, badJtw)
    @test_throws DimensionError jac_structure!(nlp, badjrows, jcols)
    @test_throws DimensionError jac_structure!(nlp, jrows, badjcols)
    @test_throws DimensionError jac_coord(nlp, badx)
    @test_throws DimensionError jac_coord!(nlp, badx, jvals)
    @test_throws DimensionError jac_coord!(nlp, x, badjvals)
    @test_throws DimensionError jac_op(nlp, badx)
    @test_throws DimensionError jac_op!(nlp, badx, Jv, Jtw)
    @test_throws DimensionError jac_op!(nlp, x, badJv, Jtw)
    @test_throws DimensionError jac_op!(nlp, x, Jv, badJtw)
    @test_throws DimensionError jac_op!(nlp, badjrows, jcols, jvals, Jv, Jtw)
    @test_throws DimensionError jac_op!(nlp, jrows, badjcols, jvals, Jv, Jtw)
    @test_throws DimensionError jac_op!(nlp, jrows, jcols, badjvals, Jv, Jtw)
    @test_throws DimensionError jac_op!(nlp, jrows, jcols, jvals, badJv, Jtw)
    @test_throws DimensionError jac_op!(nlp, jrows, jcols, jvals, Jv, badJtw)
  end
end
