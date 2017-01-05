"Problem 5 in the Hock-Schittkowski suite"
function hs5()

  nlp = Model()

  l = [-1.5, -3]
  u = [4, 3]
  @variable(nlp, l[i] ≤ x[i=1:2] ≤ u[i], start=0.0)

  @NLobjective(
    nlp,
    Min,
    sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
  )

  return nlp
end

function hs5_autodiff()

  x0 = [0.0; 0.0]
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
  l = [-1.5; -3.0]
  u = [4.0; 3.0]

  return ADNLPModel(f, x0, lvar=l, uvar=u)

end

function hs5_simple()

  x0 = [0.0; 0.0]
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
  g(x) = cos(x[1] + x[2])*[1.0; 1.0] + 2*(x[1]-x[2])*[1.0; -1.0] + [-1.5; 2.5]
  g!(x, gx) = begin
    gx[1] = cos(x[1] + x[2]) + 2*(x[1] - x[2]) - 1.5
    gx[2] = cos(x[1] + x[2]) - 2*(x[1] - x[2]) + 2.5
    return gx
  end
  Hf(x; obj_weight=1.0) = (-sin(x[1] + x[2])*ones(2,2) + [2.0 -2.0; -2.0 2.0])*obj_weight
  H(x; obj_weight=1.0) = tril(Hf(x; obj_weight=obj_weight))
  Hcoord(x; obj_weight=1.0) = findnz(sparse(H(x, obj_weight=obj_weight)))
  Hp(x, v; obj_weight=1.0) = Hf(x, obj_weight=obj_weight) * v
  Hp!(x, v, Hv; obj_weight=1.0) = begin Hv[:] = Hp(x, v, obj_weight=obj_weight) end
  l = [-1.5; -3.0]
  u = [4.0; 3.0]

  return SimpleNLPModel(f, x0, lvar=l, uvar=u, g=g, g! =g!, H=H, Hcoord=Hcoord, Hp=Hp, Hp! =Hp!)
end
