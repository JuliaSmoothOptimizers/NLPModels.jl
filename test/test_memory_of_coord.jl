function test_memory_of_coord_of_nlp(nlp :: AbstractNLPModel)
  n = nlp.meta.nvar
  m = nlp.meta.ncon

  x = 10 * [-(-1.0)^i for i = 1:n]
  y = [-(-1.0)^i for i = 1:m]

  # Hessian unconstrained test
  vals = hess_coord(nlp, x)
  al1 = @allocated hess_coord(nlp, x)
  V = zeros(nlp.meta.nnzh)
  hess_coord!(nlp, x, V)
  al2 = @allocated hess_coord!(nlp, x, V)
  @test al2 < al1 - 50

  if m > 0
    vals = hess_coord(nlp, x, y)
    al1 = @allocated vals = hess_coord(nlp, x, y)
    hess_coord!(nlp, x, y, V)
    al2 = @allocated hess_coord!(nlp, x, y, V)
    @test al2 < al1 - 50

    vals = jac_coord(nlp, x)
    al1 = @allocated vals = jac_coord(nlp, x)
    V = zeros(nlp.meta.nnzj)
    jac_coord!(nlp, x, vals)
    al2 = @allocated jac_coord!(nlp, x, vals)
    @test al2 < al1 - 50
  end
end

function test_memory_of_coord()
  @testset "Memory of coordinate inplace functions" begin
    for p in [:HS5, :HS6, :HS10, :HS11, :HS14]
      @info("Testing $p")
      nlp = eval(p)()
      test_memory_of_coord_of_nlp(nlp)
    end
    nlp = ADNLPModel(x -> dot(x, x), rand(2), c=x->[x[1] * x[2]], lcon=zeros(1), ucon=zeros(1))
    test_memory_of_coord_of_nlp(nlp)
  end
end
