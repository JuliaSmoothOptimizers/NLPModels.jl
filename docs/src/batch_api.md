# Batch API

This section documents the batch API of `NLPModels.jl`.  
The batch API reuses the same symbols as the standard (non-batch) API.  
The only new in-place function is `obj!`, which returns a vector of all objectives.

---

## Abstract types

- [`AbstractBatchNLPModel`](@ref)
- [`AbstractBatchNLPModelMeta`](@ref)

---

## Batch metadata

- [`BatchNLPModelMeta`](@ref)

---

## Objectives

| Function | Signature |
|:--------:|:---------:|
| `obj`  | `obj(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` |
| `obj!` | `obj!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bf::AbstractVector)` |

---

## Gradients

| Function | Signature |
|:--------:|:---------:|
| `grad`  | `grad(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` |
| `grad!` | `grad!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bg::AbstractMatrix)` |

---

## Constraints

| Function | Signature |
|:--------:|:---------:|
| `cons`  | `cons(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` |
| `cons!` | `cons!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bc::AbstractMatrix)` |

---

## Sparse jacobians

| Function | Signature |
|:--------:|:---------:|
| `jac_structure`  | `(jrows, jcols) = jac_structure(bnlp::AbstractBatchNLPModel)` |
| `jac_structure!` | `(jrows, jcols) = jac_structure!(bnlp::AbstractBatchNLPModel, jrows::AbstractVector, jcols::AbstractVector)` |
| `jac_coord`  | `bjvals = jac_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` |
| `jac_coord!` | `bjvals = jac_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bjvals::AbstractMatrix)` |

---

## Dense Jacobians

| Function | Signature |
|:--------:|:---------:|
| `jac_dense!` | `bJx = jac_dense!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bJx::AbstractArray)` |

---

## Jacobian-vector products

| Function | Signature |
|:--------:|:---------:|
| `jprod`  | `bJv = jprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)` |
| `jprod!` | `bJv = jprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix)` |
| `jtprod`  | `bJtv = jtprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)` |
| `jtprod!` | `bJtv = jtprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix)` |

---

## Sparse Hessians of the Lagrangian

| Function | Signature |
|:--------:|:---------:|
| `hess_structure`  | `(hrows, hcols) = hess_structure(bnlp::AbstractBatchNLPModel)` |
| `hess_structure!` | `(hrows, hcols) = hess_structure!(bnlp::AbstractBatchNLPModel, hrows::AbstractVector, hcols::AbstractVector)` |
| `hess_coord`  | `bhvals = hess_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector)` |
| `hess_coord!` | `bhvals = hess_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix)` |

---

## Dense Hessians of the Lagrangian

| Function | Signature |
|:--------:|:---------:|
| `hess_dense!` | `bHx = hess_dense!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bHx::AbstractArray)` |

---

## Hessian-vector products

| Function | Signature |
|:--------:|:---------:|
| `hprod`  | `bHv = hprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector)` |
| `hprod!` | `bHv = hprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix)` |
