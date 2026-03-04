# Parametric API

This section documents the optional parametric API of `NLPModels.jl`.
The parametric API provides derivatives of the problem data with respect to parameters `p`, for problems of the form

```math
\begin{aligned}
\min \quad & f(x, p) \\
& c_L(p) \leq c(x, p) \leq c_U(p) \\
& \ell(p) \leq x \leq u(p).
\end{aligned}
```

The main use-case is implicit differentation of KKT conditions, where the forming the RHS requires evaluating the derivatives of the problem data with respect to parameters.

Note that `p` does not appear as an explicit argument to any of the functions below.
Implementations are responsible for storing the current parameter values internally (e.g., as a field of the model struct) and reading them when evaluating the functions.

---

## Parameter access

NLPModels provides a common interface for setting and getting the current parameter values:

| Function | Signature |
|:--------:|:---------:|
| `get_param_values`  | `p = get_param_values(nlp)` |
| `set_param_values!` | `set_param_values!(nlp, p)` |

---

## Objective gradient wrt parameters

Evaluate ``\nabla_p f(x, p)``, the gradient of the objective with respect to the parameters at the current `x`:

| Function | Signature |
|:--------:|:---------:|
| `grad_param`  | `g = grad_param(nlp, x)` |
| `grad_param!` | `g = grad_param!(nlp, x, g)` |

---

## Sparse constraint Jacobian wrt parameters

Evaluate ``J_p(x) = \nabla_p c(x, p)^T``, the Jacobian of the constraints with respect to the parameters:

| Function | Signature |
|:--------:|:---------:|
| `jac_param_structure`  | `(rows, cols) = jac_param_structure(nlp)` |
| `jac_param_structure!` | `(rows, cols) = jac_param_structure!(nlp, rows, cols)` |
| `jac_param_coord`  | `vals = jac_param_coord(nlp, x)` |
| `jac_param_coord!` | `vals = jac_param_coord!(nlp, x, vals)` |

---

## Constraint Jacobian-vector products wrt parameters

Evaluate products with ``J_p(x)`` and ``J_p(x)^T`` without forming the matrix explicitly:

| Function | Signature |
|:--------:|:---------:|
| `jpprod`  | `Jv = jpprod(nlp, x, v)` |
| `jpprod!` | `Jv = jpprod!(nlp, x, v, Jv)` |
| `jptprod`  | `Jtv = jptprod(nlp, x, v)` |
| `jptprod!` | `Jtv = jptprod!(nlp, x, v, Jtv)` |

---

## Sparse variable-parameter Hessian of the Lagrangian

Evaluate ``\nabla^2_{xp} L(x, y, p)``, the mixed variable-parameter Hessian of the Lagrangian.
When `y` is omitted, only the objective contribution should be included (i.e., `y = 0`):

| Function | Signature |
|:--------:|:---------:|
| `hess_param_structure`  | `(rows, cols) = hess_param_structure(nlp)` |
| `hess_param_structure!` | `(rows, cols) = hess_param_structure!(nlp, rows, cols)` |
| `hess_param_coord`  | `vals = hess_param_coord(nlp, x; obj_weight)` |
| `hess_param_coord`  | `vals = hess_param_coord(nlp, x, y; obj_weight)` |
| `hess_param_coord!` | `vals = hess_param_coord!(nlp, x, vals; obj_weight)` |
| `hess_param_coord!` | `vals = hess_param_coord!(nlp, x, y, vals; obj_weight)` |

---

## Variable-parameter Hessian-vector products

Evaluate products with ``\nabla^2_{xp} L(x, y, p)`` and its transpose.
When `y` is omitted, only the objective contribution should be included (i.e., `y = 0`):

| Function | Signature |
|:--------:|:---------:|
| `hpprod`  | `Hv = hpprod(nlp, x, v; obj_weight)` |
| `hpprod`  | `Hv = hpprod(nlp, x, y, v; obj_weight)` |
| `hpprod!` | `Hv = hpprod!(nlp, x, v, Hv; obj_weight)` |
| `hpprod!` | `Hv = hpprod!(nlp, x, y, v, Hv; obj_weight)` |
| `hptprod`  | `Htv = hptprod(nlp, x, y, v; obj_weight)` |
| `hptprod!` | `Htv = hptprod!(nlp, x, y, v, Htv; obj_weight)` |

---

## Sparse constraint lower-bound Jacobian wrt parameters

Evaluate ``\nabla_p c_L(p)``, the Jacobian of the constraint lower bounds with respect to the parameters:

| Function | Signature |
|:--------:|:---------:|
| `lcon_jac_param_structure`  | `(rows, cols) = lcon_jac_param_structure(nlp)` |
| `lcon_jac_param_structure!` | `(rows, cols) = lcon_jac_param_structure!(nlp, rows, cols)` |
| `lcon_jac_param_coord`  | `vals = lcon_jac_param_coord(nlp)` |
| `lcon_jac_param_coord!` | `vals = lcon_jac_param_coord!(nlp, vals)` |
| `lcon_jpprod`  | `Jv = lcon_jpprod(nlp, v)` |
| `lcon_jpprod!` | `Jv = lcon_jpprod!(nlp, v, Jv)` |
| `lcon_jptprod`  | `Jtv = lcon_jptprod(nlp, v)` |
| `lcon_jptprod!` | `Jtv = lcon_jptprod!(nlp, v, Jtv)` |

---

## Sparse constraint upper-bound Jacobian wrt parameters

Evaluate ``\nabla_p c_U(p)``, the Jacobian of the constraint upper bounds with respect to the parameters:

| Function | Signature |
|:--------:|:---------:|
| `ucon_jac_param_structure`  | `(rows, cols) = ucon_jac_param_structure(nlp)` |
| `ucon_jac_param_structure!` | `(rows, cols) = ucon_jac_param_structure!(nlp, rows, cols)` |
| `ucon_jac_param_coord`  | `vals = ucon_jac_param_coord(nlp)` |
| `ucon_jac_param_coord!` | `vals = ucon_jac_param_coord!(nlp, vals)` |
| `ucon_jpprod`  | `Jv = ucon_jpprod(nlp, v)` |
| `ucon_jpprod!` | `Jv = ucon_jpprod!(nlp, v, Jv)` |
| `ucon_jptprod`  | `Jtv = ucon_jptprod(nlp, v)` |
| `ucon_jptprod!` | `Jtv = ucon_jptprod!(nlp, v, Jtv)` |

---

## Sparse variable lower-bound Jacobian wrt parameters

Evaluate ``\nabla_p \ell(p)``, the Jacobian of the variable lower bounds with respect to the parameters:

| Function | Signature |
|:--------:|:---------:|
| `lvar_jac_param_structure`  | `(rows, cols) = lvar_jac_param_structure(nlp)` |
| `lvar_jac_param_structure!` | `(rows, cols) = lvar_jac_param_structure!(nlp, rows, cols)` |
| `lvar_jac_param_coord`  | `vals = lvar_jac_param_coord(nlp)` |
| `lvar_jac_param_coord!` | `vals = lvar_jac_param_coord!(nlp, vals)` |
| `lvar_jpprod`  | `Jv = lvar_jpprod(nlp, v)` |
| `lvar_jpprod!` | `Jv = lvar_jpprod!(nlp, v, Jv)` |
| `lvar_jptprod`  | `Jtv = lvar_jptprod(nlp, v)` |
| `lvar_jptprod!` | `Jtv = lvar_jptprod!(nlp, v, Jtv)` |

---

## Sparse variable upper-bound Jacobian wrt parameters

Evaluate ``\nabla_p u(p)``, the Jacobian of the variable upper bounds with respect to the parameters:

| Function | Signature |
|:--------:|:---------:|
| `uvar_jac_param_structure`  | `(rows, cols) = uvar_jac_param_structure(nlp)` |
| `uvar_jac_param_structure!` | `(rows, cols) = uvar_jac_param_structure!(nlp, rows, cols)` |
| `uvar_jac_param_coord`  | `vals = uvar_jac_param_coord(nlp)` |
| `uvar_jac_param_coord!` | `vals = uvar_jac_param_coord!(nlp, vals)` |
| `uvar_jpprod`  | `Jv = uvar_jpprod(nlp, v)` |
| `uvar_jpprod!` | `Jv = uvar_jpprod!(nlp, v, Jv)` |
| `uvar_jptprod`  | `Jtv = uvar_jptprod(nlp, v)` |
| `uvar_jptprod!` | `Jtv = uvar_jptprod!(nlp, v, Jtv)` |
