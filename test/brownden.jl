# Brown and Dennis functions
#
#   Source: Problem 16 in
#   J.J. Moré, B.S. Garbow and K.E. Hillstrom,
#   "Testing Unconstrained Optimization Software",
#   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981
#
#   classification SUR2-AN-4-0

function brownden()

  nlp = Model()

  @variable(nlp, x[1:4])
  setvalue(x, [25.0; 5.0; -5.0; -1.0])

  @NLobjective(
    nlp,
    Min,
    sum( ((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4]*sin(i/5) -
      cos(i/5))^2)^2 for i = 1:20)
  )

  return nlp
end

function brownden_autodiff()

  x0 = [25.0; 5.0; -5.0; -1.0]
  f(x) = begin
    s = 0.0
    for i = 1:20
      s += ((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4]*sin(i/5) -
          cos(i/5))^2)^2
    end
    return s
  end

  return ADNLPModel(f, x0)
end

function brownden_simple()

  x0 = [25.0; 5.0; -5.0; -1.0]
  α(x,i) = x[1] + x[2] * i/5 - exp(i/5)
  β(x,i) = x[3] + x[4]*sin(i/5) - cos(i/5)
  θ(x,i) = α(x,i)^2 + β(x,i)^2
  f(x) = begin
    s = 0.0
    for i = 1:20
      s += θ(x,i)^2
    end
    return s
  end
  g(x) = begin
    s = zeros(4)
    for i = 1:20
      s += 2*θ(x,i)*(2*α(x,i)*[1;i/5;0;0] + 2*β(x,i)*[0;0;1;sin(i/5)])
    end
    return s
  end
  g!(x, gx) = begin gx[:] = g(x) end
  # TODO: Explicitly define these functions
  Hf(x) = ForwardDiff.hessian(f, x)
  H(x; obj_weight=1.0) = tril(obj_weight*Hf(x))
  Hcoord(x; obj_weight=1.0) = findnz(sparse(obj_weight*H(x)))
  Hp(x,v; obj_weight=1.0) = obj_weight*Hf(x)*v
  Hp!(x,v,w; obj_weight=1.0) = begin w[:] = obj_weight*Hf(x)*v end

  return SimpleNLPModel(f, x0, g=g, g! =g!, H=H, Hcoord=Hcoord, Hp=Hp, Hp! =Hp!)
end
