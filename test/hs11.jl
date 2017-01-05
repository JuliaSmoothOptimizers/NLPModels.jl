"Problem 11 in the Hock-Schittkowski suite"
function hs11()

  nlp = Model()

  @variable(nlp, x[i=1:2])
  setvalue(x[1], 4.9)
  setvalue(x[2], 0.1)

  @NLobjective(
    nlp,
    Min,
    (x[1] - 5)^2 + x[2]^2 - 25
  )

  @NLconstraint(
    nlp,
    -x[1]^2 + x[2] >= 0
  )

  return nlp
end

function hs11_autodiff()

  x0 = [4.9; 0.1]
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  c(x) = [-x[1]^2 + x[2]]
  lcon = [-Inf]
  ucon = [0.0]

  return ADNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon)

end

function hs11_simple()

  x0 = [4.9; 0.1]
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  g(x) = [2*(x[1] - 5); 2*x[2]]
  g!(x, gx) = begin
    gx[1] = 2*(x[1] - 5)
    gx[2] = 2*x[2]
    return gx
  end

  c(x) = [-x[1]^2 + x[2]]
  c!(x, cx) = begin cx[1:1] = c(x) end
  J(x) = [-2*x[1]  1.0]
  Jc(x) = findnz(sparse(J(x)))
  Jp(x, v) = J(x) * v
  Jp!(x, v, w) = begin w[1:1] = J(x) * v end
  Jtp(x, v) = J(x)' * v
  Jtp!(x, v, w) = begin w[1:2] = J(x)' * v end

  H(x) = 2*eye(2)
  C(x, y) = [-2.0 0.0; 0.0 0.0]*y[1]
  W(x; obj_weight=1.0, y=zeros(1)) = tril(obj_weight*H(x) + C(x,y))
  Wcoord(x; obj_weight=1.0, y=zeros(1)) = findnz(sparse(W(x; obj_weight=obj_weight, y=y)))
  Wp(x, v; obj_weight=1.0, y=zeros(1)) = (obj_weight*H(x) + C(x,y))*v
  Wp!(x, v, Wv; obj_weight=1.0, y=zeros(1)) = begin Wv[1:2] = (obj_weight*H(x) + C(x,y))*v end
  lcon = [-Inf]
  ucon = [0.0]

  return SimpleNLPModel(f, x0, g=g, g! =g!, c=c, c! =c!, J=J, Jcoord=Jc, Jp=Jp,
      Jp! =Jp!, Jtp=Jtp, Jtp! =Jtp!, H=W, Hcoord=Wcoord, Hp=Wp, Hp! =Wp!,
      lcon=lcon, ucon=ucon)
end
