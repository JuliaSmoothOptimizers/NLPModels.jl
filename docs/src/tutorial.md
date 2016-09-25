# Tutorial

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
 [Optimize.jl](https://github.com/JuliaSmoothOptimizers/Optimize.jl).

The main interfaces for user defined problems are

- [ADNLPModel](models/#adnlpmodel), which defines a model easily, using automatic
  differentiation.
- [SimpleNLPModel](models/#simplenlpmodel), which allows users to handle all functions themselves,
  giving

## ADNLPModel Tutorial

ADNLPModel is simple to use and is useful for classrooms.
It only needs the objective function $f$ and a starting point $x^0$ to be
well-defined.
For constrained problems, you'll also need the constraints function $c$, and
the constraints vectors $c_L$ and $c_U$, such that $c_L \leq c(x) \leq c_U$.
Equality constraints will be automatically identified as those indices $i$ for
which $c_{L_i} = c_{U_i}$.

Let's define the famous Rosenbrock function
\begin{align*}
f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2,
\end{align*}
with starting point $x^0 = (-1.2,1.0)$.

```@example adnlp
using NLPModels

nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2 , [-1.2; 1.0])
```

This is enough to define the model.
Let's get the objective function value at $x^0$, using only `nlp`.

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

1. Given $x^0$, $\varepsilon > 0$, and $\eta \in (0,1)$. Set $k = 0$;
2. If $\Vert \nabla f(x^k) \Vert < \varepsilon$ STOP with $x^* = x^k$;
3. Compute $d^k = -\nabla f(x^k)$;
4. Compute $\alpha_k \in (0,1]$ such that
$ f(x^k + \alpha_kd^k) < f(x^k) + \alpha_k\eta \nabla f(x^k)^Td^k $
5. Define $x^{k+1} = x^k + \alpha_kx^k$
6. Update $k = k + 1$ and go to step 2.

```@example adnlp
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
    while obj(nlp, x - t*∇fx) > fx - eta * t * slope
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
H(x) = hess(nlp, x) + triu(hess(nlp, x)', 1)
x = nlp.meta.x0
d = -H(x)\g(x)
```

or a few

```@example adnlp
for i = 1:5
  x = x - H(x)\g(x)
  println("x = $x")
end
```

Also, notice how we can reuse the method.

```@example adnlp
f(x) = (x[1]^2 + x[2]^2 - 4)^2 + (x[1]*x[2] - 1)^2
x0 = [2.0; 1.0]
nlp = ADNLPModel(f, x0)

x, fx, ngx, optimal, iter = steepest(nlp)
```

Even using a different model.

```@example adnlp
using OptimizationProblems # Defines a lot of JuMP models

nlp = MathProgNLPModel(woods())
x, fx, ngx, optimal, iter = steepest(nlp)
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
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

## SimpleNLPModel Tutorial

SimpleNLPModel allows you to pass every single function of the model.
On the other hand, it doesn't handle anything else. Calling an undefined
function will throw a `NotImplementedError`.
Only the objective function is mandatory (if you don't need it, pass `x->0`).

```@example slp
using NLPModels

f(x) = (x[1] - 1.0)^2 + 4*(x[2] - 1.0)^2
x0 = zeros(2)
nlp = SimpleNLPModel(f, x0)

fx = obj(nlp, nlp.meta.x0)
println("fx = $fx")

# grad(nlp, nlp.meta.x0) # This is undefined
```

```@example slp
g(x) = [2*(x[1] - 1.0); 8*(x[2] - 1.0)]
nlp = SimpleNLPModel(f, x0, g=g)

grad(nlp, nlp.meta.x0)
```

"But what's to stop me from defining `g` however I want?"
Nothing. So you have to be careful on how you're defining it.
You should probably check your derivatives.
If the function is simply defined, you can try using automatic differentiation.
Alternatively, you can use the [derivative checker](dercheck).

```@example slp
gradient_check(nlp)
```

```@example slp
gwrong(x) = [2*(x[1] - 1.0); 8*x[2] - 1.0] # Find the error
nlp = SimpleNLPModel(f, x0, g=gwrong)
gradient_check(nlp)
```

For constrained problems, we still need the constraints function, `lcon` and `ucon`.
Also, let's pass the Jacobian-vector product.

```@example slp
c(x) = [x[1]^2 + x[2]^2; x[1]*x[2] - 1]
lcon = [1.0; 0.0]
ucon = [4.0; 0.0]
Jacprod(x, v) = [2*x[1]*v[1] + 2*x[2]*v[2]; x[2]*v[1] + x[1]*v[2]]
nlp = SimpleNLPModel(f, x0, c=c, lcon=lcon, ucon=ucon, g=g, Jp=Jacprod)
jprod(nlp, ones(2), ones(2))
```

Furthermore, NLPModels also works with inplace operations.
Since some models do not take full advantage of this (like ADNLPModel),
a user might want to define his/her own functions that do.

```@example slp2
using NLPModels # hide
f(x) = (x[1] - 1.0)^2 + 4*(x[2] - 1.0)^2
x0 = zeros(2)
g!(x, gx) = begin
  gx[1] = 2*(x[1] - 1.0)
  gx[2] = 8*(x[2] = 1.0)
  return gx
end
nlp = SimpleNLPModel(f, x0, g! =g!) # Watchout, g!=g! is interpreted as g != g!
gx = zeros(2)
grad!(nlp, nlp.meta.x0, gx)
```
