export get_projector

function get_projector(constraint,comp_grid,special_operator_list::Array{String,1},A,TD_n::Tuple,TF::DataType)

  if constraint.set_type == "bounds"
    if constraint.app_mode[1] in ["matrix","tensor"]
      if constraint.TD_OP in special_operator_list
        P = x -> copyto!(x,A'*project_bounds!(A*x,constraint.min,constraint.max))
      else
        P = x -> project_bounds!(x,constraint.min,constraint.max)
      end
    else
      if constraint.TD_OP in special_operator_list
        P = x -> copyto!(x,A'*project_bounds!(reshape(A*x,TD_n),constraint.min,constraint.max,constraint.app_mode))
      else
        P = x -> project_bounds!(reshape(x,TD_n),constraint.min,constraint.max,constraint.app_mode)
      end
    end
  end

  if constraint.set_type == "prox_l1"
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,.5f0*A'*prox_l1!(A*x,constraint.max))
    else
      P = x -> prox_l1!(x,constraint.max)
    end
  end

  if constraint.set_type == "l1"
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,A'*project_l1_Duchi!(A*x,constraint.max))
    else
      P = x -> project_l1_Duchi!(x,constraint.max)
    end
  end

  if constraint.set_type == "l2"
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,A'*project_l2!(A*x,constraint.max))
    else
      P = x -> project_l2!(x,constraint.max)
    end
  end

  if constraint.set_type == "annulus"
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,A'*project_annulus!(A*x,constraint.min,constraint.max))
    else
      P =  x -> project_annulus!(x,constraint.min,constraint.max)
    end
  end

  if constraint.set_type == "subspace"
    if constraint.app_mode[1]  in ["matrix","tensor"]
      P = x -> project_subspace!(x,constraint.custom_TD_OP[1],constraint.custom_TD_OP[2])
    else
      P = x -> project_subspace!(reshape(x,comp_grid.n),constraint.custom_TD_OP[1],constraint.custom_TD_OP[2],constraint.app_mode)
    end
  end

  if constraint.set_type == "nuclear"
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,A'*project_nuclear!(reshape(A*x,TD_n),constraint.max,constraint.app_mode))
    else
      P = x -> project_nuclear!(reshape(x,TD_n),constraint.max,constraint.app_mode)
    end
  end

  if constraint.set_type == "rank"
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,A'*project_rank!(reshape(A*x,TD_n),constraint.max,constraint.app_mode[2]))
    else
      P = x -> project_rank!(reshape(x,TD_n),constraint.max,constraint.app_mode)
    end
  end

  if constraint.set_type == "histogram" #1D histogram only
    if constraint.TD_OP in special_operator_list
      P = x -> copyto!(x,A'*project_histogram_relaxed!(A*x,constraint.min,constraint.max)) #min and max should be vector valued
    else
      P = x -> project_histogram_relaxed!(x,constraint.min,constraint.max) #min and max should be vector valued
    end
  end

  if constraint.set_type == "cardinality"
    if constraint.app_mode[1]  in ["matrix","tensor"]
      if constraint.TD_OP in special_operator_list
        P = x -> copyto!(x,A'*project_cardinality!(A*x,convert(Integer,constraint.max)))
      else
        P = x -> project_cardinality!(x,convert(Integer,constraint.max))
      end
    else
      if constraint.TD_OP in special_operator_list
        P = x -> copyto!(x,A'*project_cardinality!(reshape(A*x,TD_n),convert(Integer,constraint.max)),constraint.app_mode)
      else
        P = x -> project_cardinality!(reshape(x,TD_n),convert(Integer,constraint.max),constraint.app_mode)
      end
    end
  end


return P
end
