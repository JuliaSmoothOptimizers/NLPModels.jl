function test_simple_model()
  rtol = 1e-8
  x0 = zeros(2)
  f(x) = x[1]^4 - 4*x[1]*x[2] + x[2]^2
  g(x) = [4*x[1]^3 - 4*x[2]; 2*x[2] - 4*x[1]]
  g!(x, gx) = begin gx[:] = [4*x[1]^3 - 4*x[2]; 2*x[2] - 4*x[1]] end
  H(x; obj_weight=1.0) = obj_weight*[12*x[1]^2 0.0; -4.0 2.0]
  Hc(x; obj_weight=1.0) = ([1,2,2], [1,1,2], obj_weight*[12*x[1]^2,-4.0,-2.0])
  Hp(x, v; obj_weight=1.0) = obj_weight*[12*x[1]^2*v[1] - 4.0*v[2]; -4*v[1] + 2.0*v[2]]
  Hp!(x, v, Hv; obj_weight=1.0) = begin
    Hv[:] = obj_weight*[12*x[1]^2*v[1] - 4.0*v[2]; -4*v[1] + 2.0*v[2]]
  end
  nlp = SimpleNLPModel(f, x0, g=g, g! =g!, H=H, Hp=Hp, Hp! =Hp!, Hcoord=Hc)
  v = rand(2)
  w = zeros(2)
  @test isapprox(obj(nlp, x0), f(x0), rtol=rtol)
  @test isapprox(grad(nlp, x0), g(x0), rtol=rtol)
  grad!(nlp, x0, w)
  @test isapprox(w, g!(x0, w), rtol=rtol)
  @test isapprox(hess(nlp, x0), H(x0), rtol=rtol)
  @test hess_coord(nlp, x0) == Hc(x0)
  @test isapprox(hprod(nlp, x0, v), Hp(x0, v), rtol=rtol)
  hprod!(nlp, x0, v, w)
  @test isapprox(w, Hp!(x0, v, w), rtol=rtol)
  @test_throws NotImplementedError cons(nlp, x0)

  c(x) = [x[1] + x[2] - 1; x[1]*x[2] - 1]
  c!(x, cx) = begin cx[:] = [x[1] + x[2] - 1; x[1]*x[2] - 1] end
  J(x) = [1.0 1.0; x[2] x[1]]
  Jp(x, v) = [v[1] + v[2]; x[2]*v[1] + x[1]*v[2]]
  Jp!(x, v, Jv) = begin Jv[:] = [v[1] + v[2]; x[2]*v[1] + x[1]*v[2]] end
  Jtp(x, v) = [v[1] + x[2]*v[2]; v[1] + x[1]*v[2]]
  Jtp!(x, v, Jtv) = begin Jtv[:] = [v[1] + x[2]*v[2]; v[1] + x[1]*v[2]] end
  W(x; obj_weight=1.0, y=[0.0;0.0]) = obj_weight*H(x) + [0.0 0.0; 1.0 0.0]*y[2]
  Wc(x; obj_weight=1.0, y=[0.0;0.0]) = ([1,2,2,2], [1,1,2,1],
      obj_weight*[12*x[1]^2,-4.0,-2.0,0.0] + [0.0;0.0;0.0;y[2]])
  Wp(x, v; obj_weight=1.0, y=[0.0;0.0]) = obj_weight*Hp(x, v) + y[2]*[v[2]; v[1]]
  Wp!(x, v, Wv; obj_weight=1.0, y=[0.0;0.0]) = begin Wv[:] = obj_weight*Hp(x, v) + y[2]*[v[2]; v[1]] end
  nlp = SimpleNLPModel(f, x0, g=g, g! =g!, H=W, Hp=Wp, Hp! =Wp!, Hcoord=Wc, c=c,
      c! =c!, J=J, Jp=Jp, Jp! =Jp!, Jtp=Jtp, Jtp! =Jtp!, lcon=zeros(2),
      ucon=zeros(2))
  y0 = rand(2)
  @test isapprox(obj(nlp, x0), f(x0), rtol=rtol)
  @test isapprox(grad(nlp, x0), g(x0), rtol=rtol)
  grad!(nlp, x0, w)
  @test isapprox(w, g!(x0, w), rtol=rtol)
  @test isapprox(hess(nlp, x0, y=zeros(2)), W(x0), rtol=rtol)
  @test isapprox(hess(nlp, x0, y=y0), W(x0, y=y0), rtol=rtol)
  @test hess_coord(nlp, x0, y=zeros(2)) == Wc(x0)
  @test hess_coord(nlp, x0, y=y0) == Wc(x0, y=y0)
  @test isapprox(hprod(nlp, x0, v, y=zeros(2)), Wp(x0, v), rtol=rtol)
  @test isapprox(hprod(nlp, x0, v, y=y0), Wp(x0, v, y=y0), rtol=rtol)
  hprod!(nlp, x0, v, w, y=zeros(2))
  @test isapprox(w, Wp!(x0, v, w), rtol=rtol)
  hprod!(nlp, x0, v, w, y=y0)
  @test isapprox(w, Wp!(x0, v, w, y=y0), rtol=rtol)
  @test isapprox(cons(nlp, x0), c(x0), rtol=rtol)
  cons!(nlp, x0, w)
  @test isapprox(w, c!(x0, w), rtol=rtol)
  @test isapprox(jac(nlp, x0), J(x0), rtol=rtol)
  @test isapprox(jprod(nlp, x0, v), Jp(x0, v), rtol=rtol)
  jprod!(nlp, x0, v, w)
  @test isapprox(w,Jp!(x0, v, w), rtol=rtol)
  jtprod!(nlp, x0, v, w)
  @test isapprox(w, Jtp!(x0, v, w), rtol=rtol)
  @test isapprox(jtprod(nlp, x0, v), Jtp(x0, v), rtol=rtol)
end

function test_objgrad_objcons()
  n = 100
  G = rand(n, n)
  G = G*G'
  v = rand(n)

  x0 = zeros(n)
  f(x) = dot(x, G*x)/2 + dot(v, x)
  g(x) = G*x + v
  g!(x, gx) = begin gx[:] = G*x + v end
  c(x) = [dot(v, G*x) - 1]
  c!(x, cx) = begin cx[1] = dot(v, G*x) - 1 end
  fg(x) = begin
    g = G*x
    f = dot(x, g)/2 + dot(v, x)
    g += v
    return f, g
  end
  fg!(x, g) = begin
    g[:] = G*x
    f = dot(x, g)/2 + dot(v, x)
    g[:] += v
    return f, g
  end
  fc(x) = begin
    y = G*x
    f = dot(x, y)/2 + dot(v, x)
    c = [dot(v, y) - 1]
    return f, c
  end
  fc!(x, c) = begin
    y = G*x
    f = dot(x, y)/2 + dot(v, x)
    c[1] = dot(v, y) - 1
    return f, c
  end
  nlp = SimpleNLPModel(f, x0, g=g, g! =g!, fg=fg, fg! =fg!, c=c, c! =c!, fc =fc, fc! =fc!)
end

test_simple_model()
test_objgrad_objcons()
