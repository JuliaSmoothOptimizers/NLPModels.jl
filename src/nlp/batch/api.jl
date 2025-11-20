export AbstractBatchNLPModel
export batch_obj, batch_grad, batch_grad!, batch_objgrad, batch_objgrad!, batch_objcons, batch_objcons!
export batch_cons, batch_cons!, batch_cons_lin, batch_cons_lin!, batch_cons_nln, batch_cons_nln!
export batch_jth_con, batch_jth_congrad, batch_jth_congrad!, batch_jth_sparse_congrad
export batch_jac_structure!, batch_jac_structure, batch_jac_coord!, batch_jac_coord
export batch_jac, batch_jprod, batch_jprod!, batch_jtprod, batch_jtprod!, batch_jac_op, batch_jac_op!
export batch_jac_lin_structure!, batch_jac_lin_structure, batch_jac_lin_coord!, batch_jac_lin_coord
export batch_jac_lin, batch_jprod_lin, batch_jprod_lin!, batch_jtprod_lin, batch_jtprod_lin!, batch_jac_lin_op, batch_jac_lin_op!
export batch_jac_nln_structure!, batch_jac_nln_structure, batch_jac_nln_coord!, batch_jac_nln_coord
export batch_jac_nln, batch_jprod_nln, batch_jprod_nln!, batch_jtprod_nln, batch_jtprod_nln!, batch_jac_nln_op, batch_jac_nln_op!
export batch_jth_hess_coord, batch_jth_hess_coord!, batch_jth_hess
export batch_jth_hprod, batch_jth_hprod!, batch_ghjvprod, batch_ghjvprod!
export batch_hess_structure!, batch_hess_structure, batch_hess_coord!, batch_hess_coord
export batch_hess, batch_hprod, batch_hprod!, batch_hess_op, batch_hess_op!
export batch_varscale, batch_lagscale, batch_conscale

abstract type AbstractBatchNLPModel end

function NLPModels.increment!(bnlp::AbstractBatchNLPModel, fun::Symbol)
  NLPModels.increment!(bnlp, Val(fun))
end
