export compute_relative_feasibility
function compute_relative_feasibility{TF<:Real,TI<:Integer}(m::Vector{TF},
                                                feasibility_initial::Vector{TF},
                                                TD_OP::Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}},
                                                P_sub
                                                )


feasibility_initial[1]=norm(P_sub[1](TD_OP[1]*m).-TD_OP[1]*m)./(norm(TD_OP[1]*m)+(100*eps(TF)))
gc()
end
