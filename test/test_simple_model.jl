function test_simple_model()
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
  @test obj(nlp, x0) == f(x0)
  @test grad(nlp, x0) == g(x0)
  grad!(nlp, x0, w)
  @test w == g!(x0, w)
  @test hess(nlp, x0) == H(x0)
  @test hess_coord(nlp, x0) == Hc(x0)
  @test hprod(nlp, x0, v) == Hp(x0, v)
  hprod!(nlp, x0, v, w)
  @test w == Hp!(x0, v, w)
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
  @test obj(nlp, x0) == f(x0)
  @test grad(nlp, x0) == g(x0)
  grad!(nlp, x0, w)
  @test w == g!(x0, w)
  @test hess(nlp, x0, y=zeros(2)) == W(x0)
  @test hess(nlp, x0, y=y0) == W(x0, y=y0)
  @test hess_coord(nlp, x0, y=zeros(2)) == Wc(x0)
  @test hess_coord(nlp, x0, y=y0) == Wc(x0, y=y0)
  @test hprod(nlp, x0, v, y=zeros(2)) == Wp(x0, v)
  @test hprod(nlp, x0, v, y=y0) == Wp(x0, v, y=y0)
  hprod!(nlp, x0, v, w, y=zeros(2))
  @test w == Wp!(x0, v, w)
  hprod!(nlp, x0, v, w, y=y0)
  @test w == Wp!(x0, v, w, y=y0)
  @test cons(nlp, x0) == c(x0)
  cons!(nlp, x0, w)
  @test w == c!(x0, w)
  @test jac(nlp, x0) == J(x0)
  @test jprod(nlp, x0, v) == Jp(x0, v)
  jprod!(nlp, x0, v, w)
  @test w == Jp!(x0, v, w)
  jtprod!(nlp, x0, v, w)
  @test w == Jtp!(x0, v, w)
  @test jtprod(nlp, x0, v) == Jtp(x0, v)
end

test_simple_model()
