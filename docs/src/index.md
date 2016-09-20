# NLPModels.jl documentation

This package provides general guidelines to represent optimization problems in
Julia and a standardized API to evaluate the functions and their derivatives.
The main objective is to be able to rely on that API when designing optimization
solvers in Julia.

## Introduction

The general form of the optimization problem is
\begin{align*}
\min \quad & f(x) \\\\
& c_i(x) = 0, \quad i \in E, \\\\
& c_{L_i} \leq c_i(x) \leq c_{U_i}, \quad i \in I, \\\\
& \ell \leq x \leq u,
\end{align*}
where $f:\mathbb{R}^n\rightarrow\mathbb{R}$,
$c:\mathbb{R}^n\rightarrow\mathbb{R}^m$,
$E\cup I = \\{1,2,\dots,m\\}$, $E\cap I = \emptyset$,
and
$c_{L_i}, c_{U_i}, \ell_j, u_j \in \mathbb{R}\cup\\{\pm\infty\\}$
for $i = 1,\dots,m$ and $j = 1,\dots,n$.

For computational reasons, we write
\begin{align*}
\min \quad & f(x) \\\\
& c_L \leq c(x) \leq c_U \\\\
& \ell \leq x \leq u,
\end{align*}
defining $c_{L_i} = c_{U_i}$ for all $i \in E$.
The Lagrangian of this problem is defined as
\begin{align*}
L(x,\lambda,z^L,z^U;\sigma) = \sigma f(x) + c(x)^T\lambda  + \sum_{i=1}^n z_i^L(x_i-l_i)
+ \sum_{i=1}^nz_i^U(u_i-x_i),
\end{align*}
where $\sigma$ is a scaling parameter included for computational reasons.

Optimization problems are represented by an instance/subtype of `AbstractNLPModel`.
Such instances are composed of

- an instance of `NLPModelMeta`, which provides information about the problem,
  including the number of variables, constraints, bounds on the variables, etc.
- other data specific to the provenance of the problem.

## Install

The current usable version of NLPModels.jl is in the development branch.
Install with the following commands.
```
Pkg.clone("https://github.com/JuliaSmoothOptimizers/NLPModels.jl")
Pkg.build("NLPModels")
```
If you want the `ADNLPModel` or the `JumpNLPModel`, you also need the
```
Pkg.add("ForwardDiff")
Pkg.add("JuMP")
```
respectively.

## Internal Interfaces

 - [`ADNLPModel`](@ref): Uses
   [`ForwardDiff`](http://github.com/JuliaDiff/ForwardDiff.jl) to compute the
   derivatives. It has a very simple interface, though it isn't very efficient
   for larger problems.
 - [`JuMPNLPModel`](@ref): Uses a [`JuMP`](https://github.com/JuliaOpt/JuMP.jl) model.
 - [`SimpleNLPModel`](@ref): Only uses user defined functions.
 - [`SlackModel`](@ref): Creates an equality constrained problem with bounds
    on the variables using an existing NLPModel.

## External Interfaces

 - `AmplModel`: Defined in
   [`AmplNLReader.jl`](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)
   for problems modeled using [AMPL](http://www.ampl.com)
 - `CUTEstModel`: Defined in
   [`CUTEst.jl`](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) for
   problems from [CUTEst](https://ccpforge.cse.rl.ac.uk/gf/project/cutest/wiki).

If you want your interface here, open a PR.
