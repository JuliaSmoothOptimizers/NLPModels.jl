#Problem 14 in the Hock-Schittkowski suite
function hs14_autodiff()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [x[1] - 2 * x[2] + 1; -x[1]^2/4 - x[2]^2 + 1]
  lcon = [0.0; 0.0]
  ucon = [0.0; Inf]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)
end

function hs14_simple()

  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  g(x) = [2*(x[1] - 2); 2*(x[2] - 1)]
  g!(x, gx) = begin
    gx[1] = 2*(x[1] - 2)
    gx[2] = 2*(x[2] - 1)
    return gx
  end

  c(x) = [-x[1]^2/4 - x[2]^2 + 1; x[1] - 2 * x[2] + 1]
  c!(x, cx) = begin cx[1:2] = c(x) end
  J(x) = [-x[1]/2  -2*x[2];  1.0  -2.0]
  Jc(x) = ([1,2,1,2], [1,1,2,2], J(x)[:])
  Jp(x, v) = J(x) * v
  Jp!(x, v, w) = begin w[1:2] = J(x) * v end
  Jtp(x, v) = J(x)' * v
  Jtp!(x, v, w) = begin w[1:2] = J(x)' * v end

  H(x) = 2 * Matrix(1.0I, 2, 2)
  C(x, y) = [-0.5  0.0; 0.0  -2.0]*y[1]
  W(x; obj_weight=1.0, y=zeros(2)) = tril(obj_weight*H(x) + C(x,y))
  Wcoord(x; obj_weight=1.0, y=zeros(2)) = begin
    Wx = W(x; obj_weight=obj_weight, y=y)
    return [1, 2, 2], [1, 1, 2], [Wx[1,1], Wx[2,1], Wx[2,2]]
  end
  Wp(x, v; obj_weight=1.0, y=zeros(1)) = (obj_weight*H(x) + C(x,y))*v
  Wp!(x, v, Wv; obj_weight=1.0, y=zeros(1)) = begin Wv[1:2] = (obj_weight*H(x) + C(x,y))*v end
  lcon = [0.0; 0.0]
  ucon = [Inf; 0.0]

  return SimpleNLPModel(f, x0, g=g, g! =g!, c=c, c! =c!, J=J, Jcoord=Jc, Jp=Jp,
      Jp! =Jp!, Jtp=Jtp, Jtp! =Jtp!, H=W, Hcoord=Wcoord, Hp=Wp, Hp! =Wp!,
      lcon=lcon, ucon=ucon, nnzj=4, nnzh=3)
end
