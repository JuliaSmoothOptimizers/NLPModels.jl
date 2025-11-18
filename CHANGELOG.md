# Changelog

## Unreleased

- Add temporary backwards-compatibility for old `jac_lin*` signatures (those that accepted an `x` argument).
  - The package now prefers the new, no-`x` signatures but will call the old `(..., x, ...)` implementations if a downstream package only provides them. This avoids MethodErrors for packages that haven't migrated yet (e.g., QuadraticModels.jl, ADNLPModels.jl).
  - The compatibility is implemented with runtime fallbacks and emits deprecation warnings when the old signatures are used. Downstream packages are encouraged to update to the new signatures.

(See issue #514 for context.)
