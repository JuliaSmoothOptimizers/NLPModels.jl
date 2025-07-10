@testset "NLP API test on a simple model" begin
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  ∇f(x) = [2 * (x[1] - 2); 2 * (x[2] - 1)]
  H(x) = [2.0 0; 0 2.0]
  c(x) = [x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
  J(x) = [1.0 -2.0; -0.5x[1] -2.0x[2]]
  H(x, y) = H(x) + y[2] * [-0.5 0; 0 -2.0]

  nlp = SimpleNLPModel()
  @test eltype(nlp) == Float64
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
  @test cons_nln(nlp, x) ≈ c(x)[2:2]
  @test cons_lin(nlp, x) == c(x)[1:1]
  @test jac(nlp, x) ≈ J(x)
  @test jac_nln(nlp, x) ≈ J(x)[2:2, :]
  @test jac_lin(nlp) ≈ J(x)[1:1, :]
  @test jprod(nlp, x, v) ≈ J(x) * v
  @test jprod_nln(nlp, x, v) ≈ J(x)[2:2, :] * v
  @test jprod_lin(nlp, x, v) ≈ J(x)[1:1, :] * v
  @test jtprod(nlp, x, w) ≈ J(x)' * w
  @test jtprod_nln(nlp, x, w[2:2]) ≈ J(x)[2:2, :]' * w[2:2]
  @test jtprod_lin(nlp, x, w[1:1]) ≈ J(x)[1:1, :]' * w[1:1]
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
  @test jprod_nln!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), v, Jv[2:2]) ≈
        J(x)[2:2, :] * v
  @test jprod_lin!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp), v, Jv[1:1]) ≈
        J(x)[1:1, :] * v
  @test jtprod!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), w, Jtw) ≈ J(x)' * w
  @test jtprod_nln!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), w[2:2], Jtw) ≈
        J(x)[2:2, :]' * w[2:2]
  @test jtprod_lin!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp), w[1:1], Jtw) ≈
        J(x)[1:1, :]' * w[1:1]
  Jop = jac_op!(nlp, x, Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  res = J(x) * v - w
  @test mul!(w, Jop, v, 1.0, -1.0) ≈ res
  res = J(x)' * w - v
  @test mul!(v, Jop', w, 1.0, -1.0) ≈ res
  Jop = jac_op!(nlp, jac_structure(nlp)..., jac_coord(nlp, x), Jv, Jtw)
  @test Jop * v ≈ J(x) * v
  @test Jop' * w ≈ J(x)' * w
  res = J(x) * v - w
  @test mul!(w, Jop, v, 1.0, -1.0) ≈ res
  res = J(x)' * w - v
  @test mul!(v, Jop', w, 1.0, -1.0) ≈ res
  Jop = jac_nln_op!(nlp, x, Jv[2:2], Jtw)
  @test Jop * v ≈ J(x)[2:2, :] * v
  @test Jop' * w[2:2] ≈ J(x)[2:2, :]' * w[2:2]
  res = J(x)[2:2, :] * v - w[2:2]
  @test mul!(w[2:2], Jop, v, 1.0, -1.0) ≈ res
  res = J(x)[2:2, :]' * w[2:2] - v
  @test mul!(v, Jop', w[2:2], 1.0, -1.0) ≈ res
  Jop = jac_nln_op!(nlp, jac_nln_structure(nlp)..., jac_nln_coord(nlp, x), Jv[2:2], Jtw)
  @test Jop * v ≈ J(x)[2:2, :] * v
  @test Jop' * w[2:2] ≈ J(x)[2:2, :]' * w[2:2]
  res = J(x)[2:2, :] * v - w[2:2]
  @test mul!(w[2:2], Jop, v, 1.0, -1.0) ≈ res
  res = J(x)[2:2, :]' * w[2:2] - v
  @test mul!(v, Jop', w[2:2], 1.0, -1.0) ≈ res
  Jop = jac_lin_op!(nlp, Jv[1:1], Jtw)
  @test Jop * v ≈ J(x)[1:1, :] * v
  @test Jop' * w[1:1] ≈ Jtw
  res = J(x)[1:1, :] * v - w[1:1]
  @test mul!(w[1:1], Jop, v, 1.0, -1.0) ≈ res
  res = J(x)[1:1, :]' * w[1:1] - v
  @test mul!(v, Jop', w[1:1], 1.0, -1.0) ≈ res
  Jop = jac_lin_op!(nlp, jac_lin_structure(nlp)..., jac_lin_coord(nlp), Jv[1:1], Jtw)
  @test Jop * v ≈ J(x)[1:1, :] * v
  @test Jop' * w[1:1] ≈ Jtw
  res = J(x)[1:1, :] * v - w[1:1]
  @test mul!(w[1:1], Jop, v, 1.0, -1.0) ≈ res
  res = J(x)[1:1, :]' * w[1:1] - v
  @test mul!(v, Jop', w[1:1], 1.0, -1.0) ≈ res
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
  Hop = hess_op(nlp, x)
  @test Hop * v ≈ H(x) * v
  Hop = hess_op!(nlp, x, Hv)
  @test Hop * v ≈ H(x) * v
  z = ones(nlp.meta.nvar)
  res = H(x) * v - z
  @test mul!(z, Hop, v, 1.0, -1.0) ≈ res
  Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x), Hv)
  @test Hop * v ≈ H(x) * v
  res = H(x) * v - z
  @test mul!(z, Hop, v, 1.0, -1.0) ≈ res
  Hop = hess_op(nlp, x, y)
  @test Hop * v ≈ H(x, y) * v
  Hop = hess_op!(nlp, x, y, Hv)
  @test Hop * v ≈ H(x, y) * v
  res = H(x, y) * v - z
  @test mul!(z, Hop, v, 1.0, -1.0) ≈ res
  Hop = hess_op!(nlp, hess_structure(nlp)..., hess_coord(nlp, x, y), Hv)
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
