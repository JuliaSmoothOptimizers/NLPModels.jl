@testset "NLS API test on a simple model" begin
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  JF(x) = [-1.0 0.0; -20*x[1] 10]
  HF(x, w) = w[2] * [-20.0 0; 0 0]

  nls = SimpleNLSModel()
  n = nls.meta.nvar
  m = nls.meta.ncon
  ne = nls_meta(nls).nequ

  x = randn(n)
  v = randn(n)
  w = randn(ne)
  Jv = zeros(ne)
  Jtw = zeros(n)
  Hv = zeros(n)

  @test residual(nls, x) ≈ F(x)
  @test jac_residual(nls, x) ≈ JF(x)
  @test hess_residual(nls, x, w) ≈ HF(x, w)
  @test jprod_residual(nls, x, v) ≈ JF(x) * v
  @test jtprod_residual(nls, x, w) ≈ JF(x)' * w
  @test jprod_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), v, Jv) ≈
        JF(x) * v
  @test jtprod_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), w, Jtw) ≈
        JF(x)' * w
  @test jprod_residual!(nls, x, jac_structure_residual(nls)..., v, Jv) ≈ JF(x) * v
  @test jtprod_residual!(nls, x, jac_structure_residual(nls)..., w, Jtw) ≈ JF(x)' * w
  Jop = jac_op_residual(nls, x)
  @test Jop * v ≈ JF(x) * v
  @test Jop' * w ≈ JF(x)' * w
  Jop = jac_op_residual!(nls, x, Jv, Jtw)
  @test Jop * v ≈ JF(x) * v
  @test Jop' * w ≈ JF(x)' * w
  Jop = jac_op_residual!(nls, jac_structure_residual(nls)..., jac_coord_residual(nls, x), Jv, Jtw)
  @test Jop * v ≈ JF(x) * v
  @test Jop' * w ≈ JF(x)' * w
  Jop = jac_op_residual!(nls, x, jac_structure_residual(nls)..., Jv, Jtw)
  @test Jop * v ≈ JF(x) * v
  @test Jop' * w ≈ JF(x)' * w
  I, J, V = findnz(sparse(HF(x, w)))
  @test hess_structure_residual(nls) == (I, J)
  @test hess_coord_residual(nls, x, w) ≈ V
  for j = 1:ne
    eⱼ = [i == j ? 1.0 : 0.0 for i = 1:ne]
    @test jth_hess_residual(nls, x, j) ≈ HF(x, eⱼ)
    @test hprod_residual(nls, x, j, v) ≈ HF(x, eⱼ) * v
    Hop = hess_op_residual(nls, x, j)
    @test Hop * v ≈ HF(x, eⱼ) * v
    Hop = hess_op_residual!(nls, x, j, Hv)
    @test Hop * v ≈ HF(x, eⱼ) * v
  end
end

@testset "NLP API test on a simple NLS model" begin
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  JF(x) = [-1.0 0.0; -20*x[1] 10]
  HF(x, w) = w[2] * [-20.0 0; 0 0]
  f(x) = norm(F(x))^2 / 2
  ∇f(x) = JF(x)' * F(x)
  H(x) = JF(x)' * JF(x) + HF(x, F(x))
  c(x) = [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  J(x) = [1 2x[2]; 2x[1] 1; 2x[1] 2x[2]]
  H(x, y) = H(x) + diagm(0 => [2y[2] + 2y[3]; 2y[1] + 2y[3]])

  nls = SimpleNLSModel()
  n = nls.meta.nvar
  m = nls.meta.ncon

  x = randn(n)
  y = randn(m)
  v = randn(n)
  w = randn(m)
  Jv = zeros(m)
  Jtw = zeros(n)
  Hv = zeros(n)
  Hvals = zeros(nls.meta.nnzh)

  fx, gx = objgrad!(nls, x, v)
  @test obj(nls, x) ≈ norm(F(x))^2 / 2 ≈ fx ≈ f(x)
  @test grad(nls, x) ≈ JF(x)' * F(x) ≈ gx ≈ ∇f(x)
  @test hess(nls, x) ≈ tril(H(x))
  @test hprod(nls, x, v) ≈ H(x) * v
  @test cons(nls, x) ≈ c(x)
  @test jac(nls, x) ≈ J(x)
  @test jprod(nls, x, v) ≈ J(x) * v
  @test jtprod(nls, x, w) ≈ J(x)' * w
  @test hess(nls, x, y) ≈ tril(H(x, y))
  @test hprod(nls, x, y, v) ≈ H(x, y) * v
  fx, cx = objcons(nls, x)
  @test fx ≈ f(x)
  @test cx ≈ c(x)
  fx, _ = objcons!(nls, x, cx)
  @test fx ≈ f(x)
  @test cx ≈ c(x)
  fx, gx = objgrad(nls, x)
  @test fx ≈ f(x)
  @test gx ≈ ∇f(x)
  fx, _ = objgrad!(nls, x, gx)
  @test fx ≈ f(x)
  @test gx ≈ ∇f(x)
  @test jprod!(nls, jac_structure(nls)..., jac_coord(nls, x), v, Jv) ≈ J(x) * v
  @test jprod!(nls, x, jac_structure(nls)..., v, Jv) ≈ J(x) * v
  @test jtprod!(nls, jac_structure(nls)..., jac_coord(nls, x), w, Jtw) ≈ J(x)' * w
  @test jtprod!(nls, x, jac_structure(nls)..., w, Jtw) ≈ J(x)' * w
  Jop = jac_op!(nls, x, Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  Jop = jac_op!(nls, jac_structure(nls)..., jac_coord(nls, x), Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  Jop = jac_op!(nls, x, jac_structure(nls)..., Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  ghjv = zeros(m)
  for j = 1:m
    eⱼ = [i == j ? 1.0 : 0.0 for i = 1:m]
    Cⱼ(x) = H(x, eⱼ) - H(x)
    ghjv[j] = dot(gx, Cⱼ(x) * v)
  end
  @test ghjvprod(nls, x, gx, v) ≈ ghjv
  @test hess_coord!(nls, x, Hvals) == hess_coord!(nls, x, y * 0, Hvals)
  @test hprod!(nls, hess_structure(nls)..., hess_coord(nls, x), v, Hv) ≈ H(x) * v
  @test hprod!(nls, x, hess_structure(nls)..., v, Hv) ≈ H(x) * v
  @test hprod!(nls, x, y, hess_structure(nls)..., v, Hv) ≈ H(x, y) * v
  Hop = hess_op(nls, x)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nls, x, Hv)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nls, hess_structure(nls)..., hess_coord(nls, x), Hv)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nls, x, hess_structure(nls)..., Hv)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op(nls, x, y)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nls, x, y, Hv)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nls, hess_structure(nls)..., hess_coord(nls, x, y), Hv)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nls, x, y, hess_structure(nls)..., Hv)
  @test Hop * v ≈ H(x, y) * v
end

@testset "test vector types Float32" begin
  nls = SimpleNLSModel(Float32)
  @test Float32 ==
        eltype(nls.meta.x0) ==
        eltype(nls.meta.lvar) ==
        eltype(nls.meta.uvar) ==
        eltype(nls.meta.y0) ==
        eltype(nls.meta.lcon) ==
        eltype(nls.meta.ucon)
end
