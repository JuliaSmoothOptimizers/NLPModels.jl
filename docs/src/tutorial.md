# Tutorial

```@contents
Pages = ["tutorial.md"]
```

NLPModels.jl was created for two purposes:

 - Allow users to access problem databases in an unified way.
 Mainly, this means
 [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl),
 but it also gives access to [AMPL
 problems](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl),
 as well as JuMP defined problems (e.g. as in
 [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl)).
 - Allow users to create their own problems in the same way.
 As a consequence, optimization methods designed according to the NLPModels API
 will accept NLPModels of any provenance.
 See, for instance,
 [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) and
 [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl).

The main interface for user defined problems is [ADNLPModel](@ref), which defines a
model easily, using automatic differentiation.

## ADNLPModel Tutorial

ADNLPModel is simple to use and is useful for classrooms.
It only needs the objective function ``f`` and a starting point ``x^0`` to be
well-defined.
For constrained problems, you'll also need the constraints function ``c``, and
the constraints vectors ``c_L`` and ``c_U``, such that ``c_L \leq c(x) \leq c_U``.
Equality constraints will be automatically identified as those indices ``i`` for
which ``c_{L_i} = c_{U_i}``.

Let's define the famous Rosenbrock function
```math
f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2,
```
with starting point ``x^0 = (-1.2,1.0)``.

```@example adnlp
using NLPModels

nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2 , [-1.2; 1.0])
```

This is enough to define the model.
Let's get the objective function value at ``x^0``, using only `nlp`.

```@example adnlp
fx = obj(nlp, nlp.meta.x0)
println("fx = $fx")
```

Done.
Let's try the gradient and Hessian.

```@example adnlp
gx = grad(nlp, nlp.meta.x0)
Hx = hess(nlp, nlp.meta.x0)
println("gx = $gx")
println("Hx = $Hx")
```

Notice how only the lower triangle of the Hessian is stored.
Also notice that it is *dense*. This is a current limitation of this model. It
doesn't return sparse matrices, so use it with care.

Let's do something a little more complex here, defining a function to try to
solve this problem through steepest descent method with Armijo search.
Namely, the method

1. Given ``x^0``, ``\varepsilon > 0``, and ``\eta \in (0,1)``. Set ``k = 0``;
2. If ``\Vert \nabla f(x^k) \Vert < \varepsilon`` STOP with ``x^* = x^k``;
3. Compute ``d^k = -\nabla f(x^k)``;
4. Compute ``\alpha_k \in (0,1]`` such that ``f(x^k + \alpha_kd^k) < f(x^k) + \alpha_k\eta \nabla f(x^k)^Td^k``
5. Define ``x^{k+1} = x^k + \alpha_kx^k``
6. Update ``k = k + 1`` and go to step 2.

```@example adnlp
using LinearAlgebra

function steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)
  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  slope = dot(∇fx, ∇fx)
  ∇f_norm = sqrt(slope)
  iter = 0
  while ∇f_norm > eps && iter < itmax
    t = 1.0
    x_trial = x - t * ∇fx
    f_trial = obj(nlp, x_trial)
    while f_trial > fx - eta * t * slope
      t *= sigma
      x_trial = x - t * ∇fx
      f_trial = obj(nlp, x_trial)
    end
    x = x_trial
    fx = f_trial
    ∇fx = grad(nlp, x)
    slope = dot(∇fx, ∇fx)
    ∇f_norm = sqrt(slope)
    iter += 1
  end
  optimal = ∇f_norm <= eps
  return x, fx, ∇f_norm, optimal, iter
end

x, fx, ngx, optimal, iter = steepest(nlp)
println("x = $x")
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
```

Maybe this code is too complicated? If you're in a class you just want to show a
Newton step.

```@example adnlp
g(x) = grad(nlp, x)
H(x) = Symmetric(hess(nlp, x), :L)
x = nlp.meta.x0
d = -H(x)\g(x)
```

or a few

```@example adnlp
for i = 1:5
  global x
  x = x - H(x)\g(x)
  println("x = $x")
end
```

Also, notice how we can reuse the method.

```@example adnlp
f(x) = (x[1]^2 + x[2]^2 - 5)^2 + (x[1]*x[2] - 2)^2
x0 = [3.0; 2.0]
nlp = ADNLPModel(f, x0)

x, fx, ngx, optimal, iter = steepest(nlp)
```

Even using a different model. In this case, a model from
[NLPModelsJuMP](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl) implemented in
[OptimizationProblems](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl).

```@example adnlp
#= Commented out until NLPModelsJuMP is updated
using NLPModelsJuMP, OptimizationProblems

nlp = MathProgNLPModel(woods())
x, fx, ngx, optimal, iter = steepest(nlp)
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
=#
```

For constrained minimization, you need the constraints vector and bounds too.
Bounds on the variables can be passed through a new vector.

```@example adnlp2
using NLPModels # hide
f(x) = (x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2
x0 = [-1.2; 1.0]
lvar = [-Inf; 0.1]
uvar = [0.5; 0.5]
c(x) = [x[1] + x[2] - 2; x[1]^2 + x[2]^2]
lcon = [0.0; -Inf]
ucon = [Inf; 1.0]
nlp = ADNLPModel(f, x0, c=c, lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon)

println("cx = $(cons(nlp, nlp.meta.x0))")
println("Jx = $(jac(nlp, nlp.meta.x0))")
```

## Manual model

Sometimes you want or need to input your derivatives by hand the easier way to do so is
to define a new model. Which functions you want to define depend on which solver you are
using. In out `test` folder, we have the files `hs5.jl`, `hs6.jl`, `hs10.jl`, `hs11.jl`,
`hs14.jl` and `brownden.jl` as examples. We present the relevant part of `hs6.jl` here
as well:

```@example hs6
import NLPModels: increment!
using NLPModels

mutable struct HS6 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS6()
  meta = NLPModelMeta(2, ncon=1, nnzh=1, nnzj=2, x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6(meta, Counters())
end

function NLPModels.obj(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 1); 0.0]
  return gx
end

function NLPModels.hess(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return [2.0 * obj_weight - 20 * w   0.0; 0.0 0.0]
end

function NLPModels.hess_coord(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return ([1], [1], [2.0 * obj_weight - 20 * w])
end

function NLPModels.hprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  w = length(y) > 0 ? y[1] : 0.0
  Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
  return Hv
end

function NLPModels.cons!(nlp :: HS6, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.jac(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-20 * x[1]  10.0]
end

function NLPModels.jac_coord(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-20 * x[1]; 10] * v[1]
  return Jtv
end
```

```@example hs6
hs6 = HS6()
x = hs6.meta.x0
(obj(hs6, x), grad(hs6, x))
```

```@example hs6
cons(hs6, x)
```

Notice that we did not define `grad` nor `cons`, but `grad!` and `cons!` were defined.
The default `grad` and `cons` uses the inplace version, so there's no need to redefine
them.

## Nonlinear least squares models

In addition to the general nonlinear model, we can define the residual function for a
nonlinear least-squares problem. In other words, the objective function of the problem
is of the form ``f(x) = \tfrac{1}{2}\|F(x)\|^2``, and we can define the function ``F``
and its derivatives.

A simple way to define an NLS problem is with `ADNLSModel`, which uses automatic
differentiation.

```@example nls
using NLPModels # hide
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2) # 2 nonlinear equations
residual(nls, x0)
```

```@example nls
jac_residual(nls, x0)
```

We can also define a linear least squares by passing the matrices that define the
problem
```math
\begin{align*}
\min \quad & \tfrac{1}{2}\|Ax - b\|^2 \\
& c_L  \leq Cx \leq c_U \\
& \ell \leq  x \leq u.
\end{align*}
```
```@example nls
using LinearAlgebra # hide
A = rand(10, 3)
b = rand(10)
C = rand(2, 3)
nls = LLSModel(A, b, C=C, lcon=zeros(2), ucon=zeros(2), lvar=-ones(3), uvar=ones(3))

@info norm(jac_residual(nls, zeros(3)) - A)
@info norm(jac(nls, zeros(3)) - C)
```

Another way to define a nonlinear least squares is using `FeasibilityResidual` to
consider the constraints of a general nonlinear problem as the residual of the NLS.
```@example nls
nlp = ADNLPModel(x->0, # objective doesn't matter,
                 ones(2), c=x->[x[1] + x[2] - 1; x[1] * x[2] - 2],
                 lcon=zeros(2), ucon=zeros(2))
nls = FeasibilityResidual(nlp)
s = 0.0
for t = 1:100
  global s
  x = rand(2)
  s += norm(residual(nls, x) - cons(nlp, x))
end
@info "s = $s"
```
