mutable struct LinearRegression
  x :: Vector
  y :: Vector
end

function (regr::LinearRegression)(beta)
  r = regr.y .- beta[1] - beta[2] * regr.x
  return dot(r, r) / 2
end

function test_autodiff_model()
  x0 = zeros(2)
  f(x) = dot(x,x)
  nlp = ADNLPModel(f, x0)

  c(x) = [sum(x) - 1]
  nlp = ADNLPModel(f, x0, c, [0], [0])
  @test obj(nlp, x0) == f(x0)

  x = range(-1, stop=1, length=100)
  y = 2x .+ 3 + randn(100) * 0.1
  regr = LinearRegression(x, y)
  nlp = ADNLPModel(regr, ones(2))
  β = [ones(100) x] \ y
  @test abs(obj(nlp, β) - norm(y .- β[1] - β[2] * x)^2 / 2) < 1e-12
  @test norm(grad(nlp, β)) < 1e-12

  @testset "Constructors for ADNLPModel" begin
    lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
    badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
    nlp = ADNLPModel(f, x0)
    nlp = ADNLPModel(f, x0, lvar, uvar)
    nlp = ADNLPModel(f, x0, c, lcon, ucon)
    nlp = ADNLPModel(f, x0, c, lcon, ucon, y0=y0)
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, y0=y0)
    @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar)
    @test_throws DimensionError ADNLPModel(f, x0, c, badlcon, ucon)
    @test_throws DimensionError ADNLPModel(f, x0, c, lcon, baducon)
    @test_throws DimensionError ADNLPModel(f, x0, c, lcon, ucon, y0=bady0)
    @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, c, lcon, ucon)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, c, lcon, ucon)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, badlcon, ucon)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, lcon, baducon)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, y0=bady0)
  end
end

test_autodiff_model()
