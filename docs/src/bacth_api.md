# Batch API

This section documents the batch API of `NLPModels.jl`.  
The batch API reuses the same symbols as the standard (non-batch) API.  
The only new in-place function is `obj!`, which returns a vector of all objectives.

---

## Abstract types

```@docs
AbstractBatchNLPModel
AbstractBatchNLPModelMeta
```

---

## Batch metadata

```@docs
BatchNLPModelMeta
```

---

## Objective functions

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `obj`  | `obj(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` | Computes objectives for each column of `bx` (out-of-place). |
| `obj!` | `obj!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bf::AbstractVector)` | In-place version storing results in `bf`. |

---

## Gradients

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `grad`  | `grad(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` | Compute gradients for each column of `bx` (out-of-place). |
| `grad!` | `grad!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bg::AbstractMatrix)` | In-place version storing results in `bg`. |

---

## Constraints

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `cons`  | `cons(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` | Evaluate constraints for batch input. |
| `cons!` | `cons!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bc::AbstractMatrix)` | In-place version storing results in `bc`. |

---

## Jacobians

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `jac_structure`  | `(jrows, jcols) = jac_structure(bnlp::AbstractBatchNLPModel)` | Get common sparsity pattern. |
| `jac_structure!` | `(jrows, jcols) = jac_structure!(bnlp::AbstractBatchNLPModel, jrows::AbstractVector, jcols::AbstractVector)` | In-place version. |
| `jac_coord`  | `bjvals = jac_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix)` | Compute Jacobian values. |
| `jac_coord!` | `bjvals = jac_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bjvals::AbstractMatrix)` | In-place version. |

---

## Jacobian-vector products

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `jprod`  | `bJv = jprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)` | Jacobian-vector product. |
| `jprod!` | `bJv = jprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix)` | In-place version. |
| `jtprod`  | `bJtv = jtprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix)` | Transposed Jacobian-vector product. |
| `jtprod!` | `bJtv = jtprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix)` | In-place version. |

---

## Hessians

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `hess_structure`  | `(hrows, hcols) = hess_structure(bnlp::AbstractBatchNLPModel)` | Get common sparsity pattern. |
| `hess_structure!` | `(hrows, hcols) = hess_structure!(bnlp::AbstractBatchNLPModel, hrows::AbstractVector, hcols::AbstractVector)` | In-place version. |
| `hess_coord`  | `bhvals = hess_coord(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector)` | Compute Hessian of the Lagrangian entries. |
| `hess_coord!` | `bhvals = hess_coord!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bhvals::AbstractMatrix)` | In-place version. |

---

## Hessian-vector products

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `hprod`  | `bHv = hprod(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector)` | Hessian-vector product. |
| `hprod!` | `bHv = hprod!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix)` | In-place version. |

---

## Dense Jacobian and Hessian

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `jac_dense!` | `bJx = jac_dense!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, bJx::AbstractArray)` | Dense Jacobian in-place. |
| `hess_dense!` | `bHx = hess_dense!(bnlp::AbstractBatchNLPModel, bx::AbstractMatrix, by::AbstractMatrix, bobj_weight::AbstractVector, bHx::AbstractArray)` | Dense Hessian of the Lagrangian in-place. |
