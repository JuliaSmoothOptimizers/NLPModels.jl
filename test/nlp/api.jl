@testset "NLP API test on a simple model" begin
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  ∇f(x) = [2 * (x[1] - 2); 2 * (x[2] - 1)]
  H(x) = [2.0 0; 0 2.0]
  c(x) = [x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
  J(x) = [1.0 -2.0; -0.5x[1] -2.0x[2]]
  H(x, y) = H(x) + y[2] * [-0.5 0; 0 -2.0]

  nlp = SimpleNLPModel()
  n = nlp.meta.nvar
  m = nlp.meta.ncon

  x = randn(n)
  y = randn(m)
  v = randn(n)
  w = randn(m)
  Jv = zeros(m)
  Jtw = zeros(n)
  Hv = zeros(n)
  Hvals = zeros(nlp.meta.nnzh)

  # Basic methods
  @test obj(nlp, x) ≈ f(x)
  @test grad(nlp, x) ≈ ∇f(x)
  @test hess(nlp, x) ≈ tril(H(x))
  @test hprod(nlp, x, v) ≈ H(x) * v
  @test cons(nlp, x) ≈ c(x)
  @test jac(nlp, x) ≈ J(x)
  @test jprod(nlp, x, v) ≈ J(x) * v
  @test jtprod(nlp, x, w) ≈ J(x)' * w
  @test hess(nlp, x, y) ≈ tril(H(x, y))
  @test hprod(nlp, x, y, v) ≈ H(x, y) * v

  # Increasing coverage
  fx, cx = objcons(nlp, x)
  @test fx ≈ f(x)
  @test cx ≈ c(x)
  fx, _ = objcons!(nlp, x, cx)
  @test fx ≈ f(x)
  @test cx ≈ c(x)
  fx, gx = objgrad(nlp, x)
  @test fx ≈ f(x)
  @test gx ≈ ∇f(x)
  fx, _ = objgrad!(nlp, x, gx)
  @test fx ≈ f(x)
  @test gx ≈ ∇f(x)
  @test jprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), v, Jv) ≈ J(x) * v
  @test jprod!(nlp, x, jac_structure(nlp)..., v, Jv) ≈ J(x) * v
  @test jtprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), w, Jtw) ≈ J(x)' * w
  @test jtprod!(nlp, x, jac_structure(nlp)..., w, Jtw) ≈ J(x)' * w
  Jop = jac_op!(nlp, x, Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  Jop = jac_op!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  Jop = jac_op!(nlp, x, jac_structure(nlp)..., Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  for j = 1:(nlp.meta.ncon)
    eⱼ = [i == j ? 1.0 : 0.0 for i = 1:m]
    @test jth_hess(nlp, x, j) == H(x, eⱼ) - H(x)
    @test sparse(hess_structure(nlp)..., jth_hess_coord(nlp, x, j), n, n) == H(x, eⱼ) - H(x)
    @test jth_hprod(nlp, x, v, j) == (H(x, eⱼ) - H(x)) * v
  end
  ghjv = zeros(m)
  for j = 1:m
    eⱼ = [i == j ? 1.0 : 0.0 for i = 1:m]
    Cⱼ(x) = H(x, eⱼ) - H(x)
    ghjv[j] = dot(gx, Cⱼ(x) * v)
  end
  @test ghjvprod(nlp, x, gx, v) ≈ ghjv
  @test hess_coord!(nlp, x, Hvals) == hess_coord!(nlp, x, y * 0, Hvals)
  @test hprod!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), v, Hv) ≈ H(x) * v
  @test hprod!(nlp, x, hess_structure(nlp)..., v, Hv) ≈ H(x) * v
  @test hprod!(nlp, x, y, hess_structure(nlp)..., v, Hv) ≈ H(x, y) * v
  Hop = hess_op(nlp, x)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nlp, x, Hv)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), Hv)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nlp, x, hess_structure(nlp)..., Hv)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op(nlp, x, y)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nlp, x, y, Hv)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x, y), Hv)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nlp, x, y, hess_structure(nlp)..., Hv)
  @test Hop * v ≈ H(x, y) * v
end

@testset "test vector types Float32" begin
  nlp = SimpleNLPModel(Float32)
  @test Float32 ==
        eltype(nlp.meta.x0) ==
        eltype(nlp.meta.lvar) ==
        eltype(nlp.meta.uvar) ==
        eltype(nlp.meta.y0) ==
        eltype(nlp.meta.lcon) ==
        eltype(nlp.meta.ucon)
end
