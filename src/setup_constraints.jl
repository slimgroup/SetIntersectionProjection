export setup_constraints

"""
input:

# Arguments
- constraint: dictionary with information about which constraints to use and their specifications
- comp_grid:  structure with computational grid information; comp_grid.n = (nx,ny,nz) number of gridpoints in each direction; comp_grid.d = (dx,dy,dz) spacing between gridpoints in each direction.
- TF:         Floating point precision (either Float32 or Float64)

output:
- P_sub    - vector of projection functions onto C: P_sub = [P_C_1(.); P_C_2(.); ... ; P_C_p(.)]
- TD_OP    - vector of linear operators [A_1;A_2;...;A_p]
- set_Prop - set properties, structure where each property is vector which has a lenght of the number of sets
                             (see SetIntersectionProjection.jl and setup_constraints.jl)
"""
function setup_constraints(constraint,comp_grid,TF)

# if    TF == Float64
#   TI = Int64
# elseif TF == Float32
#   TI = Int32
# end
TI=Int64

[@spawnat pid comp_grid for pid in workers()] #send computational grid to all workers in parallel mode

nr_constraints = length(constraint)

#convert all integers and floats to the desired precision
for i=1:nr_constraints
  if length(constraint[i].min) == 1 #scalar
    if typeof(constraint[i].min) in [Int32,Int64]
      #do nothing
    elseif typeof(constraint[i].min) in [Float32,Float64]
      constraint[i].min = TF(constraint[i].min)
      constraint[i].max = TF(constraint[i].max)
    end
  elseif length(constraint[i].min) > 1 #vector
    constraint[i].min = convert(Vector{TF},constraint[i].min)
    constraint[i].max = convert(Vector{TF},constraint[i].max)
  end
end

#allocate
P_sub = Vector{Any}(undef,nr_constraints)
TD_OP = Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}}(undef,nr_constraints)
AtA   = Vector{SparseMatrixCSC{TF,TI}}(undef,nr_constraints)

#initialize storage for set properties
set_Prop=set_properties(zeros(nr_constraints),zeros(nr_constraints),zeros(nr_constraints),Vector{Tuple{TI,TI}}(undef,nr_constraints),Vector{Tuple{String,String,String,String}}(undef,nr_constraints),zeros(nr_constraints),Vector{Vector{TI}}(undef,nr_constraints))

#other initialization
special_operator_list = ["DFT", "DCT", "wavelet"] #complex valued operators that are orthogonal
N =  prod(comp_grid.n)

for i = 1:nr_constraints

      #verify input consistency
      if (constraint[i].set_type in ["nuclear", "rank"]) && (constraint[i].app_mode[1] in ["matrix","tensor"]) && (length(comp_grid.n)==3)
        error("requested rank or nuclear norm constraints on a tensor, use mode=(slice,x) e.t.c. to define constraints per slice")
      end

      # #catch a few things not currently implemented, yet..
      if constraint[i].set_type in ["l1", "l2"] && constraint[i].app_mode[1] in ["slice", "fiber"]
        error("l1 and l2 constraints only available for matrix or tensor mode, currently")
      end


      (A,AtA_diag,dense,TD_n,banded) = get_TD_operator(comp_grid,constraint[i].TD_OP,TF)

      P_sub[i] = get_projector(constraint[i],comp_grid,special_operator_list,A,TD_n,TF)

      if constraint[i].TD_OP in special_operator_list
        TD_OP[i] =  SparseMatrixCSC{TF}(LinearAlgebra.I,N,N)
      else
        TD_OP[i] = A
      end

      set_Prop.AtA_diag[i] = AtA_diag
      set_Prop.dense[i]    = dense
      set_Prop.TD_n[i]     = TD_n
      set_Prop.banded[i]   = banded
      set_Prop.tag[i]      = (constraint[i].set_type,constraint[i].TD_OP,constraint[i].app_mode[1],constraint[i].app_mode[2])

      #determine set convexity
      if constraint[i].set_type in ["rank","cardinality"]
        set_Prop.ncvx[i]     = true
      elseif constraint[i].set_type == "bounds" && constraint[i].TD_OP != "identity" && TF(maximum(constraint[i].min))>TF(0.0)
        set_Prop.ncvx[i]     = true
      elseif constraint[i].set_type == "histogram" && constraint[i].TD_OP != "identity" && TF(maximum(constraint[i].min))>TF(0.0)
        set_Prop.ncvx[i]     = true
      else
        set_Prop.ncvx[i]     = false
      end

end

return P_sub,TD_OP,set_Prop
end #end setup_constraints
