function test_memory_of_coord_of_nlp(nlp :: AbstractNLPModel)
  n = nlp.meta.nvar
  m = nlp.meta.ncon

  x = 10 * [-(-1.0)^i for i = 1:n]
  y = [-(-1.0)^i for i = 1:m]

  # Hessian unconstrained test
  rows, cols, vals = hess_coord(nlp, x)
  al1 = @allocated hess_coord(nlp, x)
  V = zeros(nlp.meta.nnzh)
  hess_coord!(nlp, x, rows, cols, V)
  al2 = @allocated hess_coord!(nlp, x, rows, cols, V)
  @test al2 < al1 - 150

  if m > 0
    rows, cols, vals = hess_coord(nlp, x, y=y)
    al1 = @allocated rows, cols, vals = hess_coord(nlp, x, y=y)
    hess_coord!(nlp, x, rows, cols, V, y=y)
    al2 = @allocated hess_coord!(nlp, x, rows, cols, V, y=y)
    @test al2 < al1 - 150

    rows, cols, vals = jac_coord(nlp, x)
    al1 = @allocated rows, cols, vals = jac_coord(nlp, x)
    V = zeros(nlp.meta.nnzj)
    jac_coord!(nlp, x, rows, cols, vals)
    al2 = @allocated jac_coord!(nlp, x, rows, cols, vals)
    @test al2 < al1 - 150
  end
end

function test_memory_of_coord()
  @testset "Memory of coordinate inplace functions" begin
    for p in [:HS5, :HS6, :HS10, :HS11, :HS14]
      @info("Testing $p")
      nlp = eval(p)()
      test_memory_of_coord_of_nlp(nlp)
    end
  end
end
