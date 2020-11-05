export compute_relative_feasibility

"""
Compute transform-domain relative feasibility w.r.t. a set
r_feas =  ||P(A*x)-A*x||/||A*x||
"""
function compute_relative_feasibility(m::Vector{TF}, feasibility_initial::Vector{TF},
                                      TD_OP::Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}},
                                       P_sub) where {TF<:Real,TI<:Integer}

feasibility_initial[1]=norm(P_sub[1](TD_OP[1]*m) .- TD_OP[1]*m) ./ (norm(TD_OP[1]*m)+(100*eps(TF)))
#gc() check if we need this in julia >0.7
end
