export PARSDMM_precompute_distribute

"""
Precomputes and distributes some quantities that serve as input for PARSDMM.jl
"""
function PARSDMM_precompute_distribute(
                                      TD_OP    ::Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}},
                                      set_Prop,
                                      comp_grid,
                                      options
                                      ) where {TF<:Real,TI<:Integer}


N = prod(comp_grid.n)

#add the identity matrix as the operator for the squared distance from the point we want to project
if options.feasibility_only==false
  push!(TD_OP,SparseMatrixCSC{TF}(LinearAlgebra.I,N,N));
  push!(set_Prop.TD_n,comp_grid.n)
  push!(set_Prop.AtA_offsets,[0])
  push!(set_Prop.banded,true)
  push!(set_Prop.AtA_diag,true)
  push!(set_Prop.ncvx,false)
  push!(set_Prop.dense,false)
  push!(set_Prop.tag,("distance squared","identity","matrix",""))
end


p     = length(TD_OP);

AtA=Vector{SparseMatrixCSC{TF,TI}}(undef,p)
for i=1:p
  if set_Prop.dense[i]==true
    if set_Prop.AtA_diag[i]==true
      AtA[i]=SparseMatrixCSC{TF}(LinearAlgebra.I,N,N)
    else
      error("provided a dense non orthogoal transform-domain operator")
    end
  else
    AtA[i] = TD_OP[i]'*TD_OP[i]
  end
end

#if all AtA are banded -> convert to CDS (DIA) format
if sum(set_Prop.banded[1:p].=true)==p
  for i=1:p
    (AtA[i],set_Prop.AtA_offsets[i])  = mat2CDS(AtA[i])
    set_Prop.AtA_offsets[i]=convert(Vector{TI},set_Prop.AtA_offsets[i])
  end
   AtA=convert(Vector{Array{TF,2}},AtA);
  set_Prop.AtA_offsets=set_Prop.AtA_offsets[1:p]
end

#allocate arrays of vectors
y       = Vector{Vector{TF}}(undef,p);
l       = Vector{Vector{TF}}(undef,p);

for ii=1:p #initialize all rho's, gamma's, y's and l's
    y[ii]       = zeros(TF,size(TD_OP[ii],1))
    l[ii]       = zeros(TF,length(y[ii]));
end

if options.parallel==true
  TD_OP = distribute(TD_OP)
  y     = distribute(y)
  l     = distribute(l)
end

return TD_OP,AtA,l,y
end
