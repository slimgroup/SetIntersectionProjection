export PARSDMM_precompute_distribute_Minkowski

function PARSDMM_precompute_distribute_Minkowski(
                                      TD_OP_c1::Vector{Union{SparseMatrixCSC{TF,TI}, joAbstractLinearOperator{TF,TF}}},
                                      TD_OP_c2::Vector{Union{SparseMatrixCSC{TF,TI}, joAbstractLinearOperator{TF,TF}}},
                                      TD_OP_sum::Vector{Union{SparseMatrixCSC{TF,TI}, joAbstractLinearOperator{TF,TF}}},
                                      set_Prop_c1,
                                      set_Prop_c2,
                                      set_Prop_sum,
                                      comp_grid,
                                      options
                                      ) where {TF<:Real,TI<:Integer}

N = prod(comp_grid.n)
p = length(TD_OP_c1)
q = length(TD_OP_c2)
r = length(TD_OP_sum)

if options.feasibility_only==true
  s=p+q+r
else
  s=p+q+r+1
end

joli_op = false

AtA = Vector{Union{Array{TF, 2}, SparseMatrixCSC{TF, TI}, joAbstractLinearOperator{TF, TF}}}(undef, s)
ZERO_MAT_J = joZeros(N, N; DDT=TF, RDT=TF)
ZERO_MAT = spzeros(TF, N, N)
Id_MAT = SparseMatrixCSC{TF}(LinearAlgebra.I, N, N)

for i=1:p #first component
  if set_Prop_c1.dense[i]==true
    if set_Prop_c1.AtA_diag[i]==true
      AtA[i]=[ Id_MAT ZERO_MAT ; ZERO_MAT ZERO_MAT ]
      #set_Prop_c1.AtA_offsets[i]=[0] #do not need to add additional offset info
    else
      error("provided a dense non orthogoal transform-domain operator")
    end
  else
    typeof(TD_OP_c1[i]) <: joAbstractLinearOperator ? ZM = ZERO_MAT_J : ZM = ZERO_MAT
    joli_op = joli_op || (typeof(TD_OP_c1[i]) <: joAbstractLinearOperator)
    AtA[i] = [ TD_OP_c1[i]'*TD_OP_c1[i] ZM ; ZM ZM ]
    #set_Prop_c1.AtA_offsets[i] #do not need to add additional offset info
  end
end
for i=1:q #second component
  if set_Prop_c2.dense[i]==true
    if set_Prop_c2.AtA_diag[i]==true
      AtA[p+i]= [ ZERO_MAT ZERO_MAT ; ZERO_MAT Id_MAT ]
      #set_Prop_c1.AtA_offsets[p+i]=[0] #do not need to add additional offset info
    else
      error("provided a dense non orthogoal transform-domain operator")
    end
  else
    typeof(TD_OP_c2[i]) <: joAbstractLinearOperator ? ZM = ZERO_MAT_J : ZM = ZERO_MAT
    joli_op = joli_op || (typeof(TD_OP_c1[i]) <: joAbstractLinearOperator)
    AtA[p+i] = [ ZM ZM ; ZM TD_OP_c2[i]'*TD_OP_c2[i] ]
    #do not need to add additional offset info
  end
end
for i=1:r #total image
  if set_Prop_sum.dense[i]==true
    if set_Prop_sum.AtA_diag[i]==true
      AtA[i+p+q]=[ Id_MAT Id_MAT ; Id_MAT Id_MAT ]
      #set_Prop_sum.AtA_offsets[i]=[-N 0 N]
    else
      error("provided a dense non orthogoal transform-domain operator")
    end
  else
    AtA[i+p+q] = [ TD_OP_sum[i]'*TD_OP_sum[i] TD_OP_sum[i]'*TD_OP_sum[i] ; TD_OP_sum[i]'*TD_OP_sum[i] TD_OP_sum[i]'*TD_OP_sum[i] ]
    #set_Prop_sum.AtA_offsets[i]=[set_Prop_sum.AtA_offsets[i]-N set_Prop_sum.AtA_offsets[i] set_Prop_sum.AtA_offsets[i]+N]
  end
end

#modify the transform-domain operators to include two blocks per operator, i.e.,
# TD_OP = [A] -> TD_OP = [A 0] or [0 A] or [A A]. This is one row of \tilde{A}
for i=1:length(TD_OP_c1)
  typeof(TD_OP_c1[i])<:joAbstractLinearOperator ? ZM = joZeros(size(TD_OP_c1[i],1), N; DDT=TF, RDT=TF) : ZM = spzeros(TF, size(TD_OP_c1[i],1), N)
  TD_OP_c1[i] = [ TD_OP_c1[i] ZM ];
end
for i=1:length(TD_OP_c2)
  typeof(TD_OP_c2[i])<:joAbstractLinearOperator ? ZM = joZeros(size(TD_OP_c2[i],1), N; DDT=TF, RDT=TF) : ZM = spzeros(TF, size(TD_OP_c2[i],1), N)
  TD_OP_c2[i] = [ ZM TD_OP_c2[i] ];
end
for i=1:length(TD_OP_sum)
  TD_OP_sum[i] = [ TD_OP_sum[i] TD_OP_sum[i] ];
end

#add the identity matrix as the operator for the squared distance from the point we want to project
if options.feasibility_only==false
  push!(TD_OP_sum,[ Id_MAT Id_MAT ]);
  push!(set_Prop_sum.TD_n,comp_grid.n)
  push!(set_Prop_sum.AtA_offsets,[0]) #push a dummy value (doesn't matter because the offsets will bet detected automatically)
  push!(set_Prop_sum.banded,true)
  push!(set_Prop_sum.AtA_diag,false)
  push!(set_Prop_sum.dense,false)
  push!(set_Prop_sum.ncvx,false)
  push!(set_Prop_sum.tag,("distance squared","identity","matrix",""))
  AtA[s] = [ Id_MAT Id_MAT ; Id_MAT Id_MAT ]
end

#create one set_Prop for all operators
set_Prop = deepcopy(set_Prop_c1)

append!(set_Prop.AtA_diag,set_Prop_c2.AtA_diag)
append!(set_Prop.AtA_offsets,set_Prop_c2.AtA_offsets)
append!(set_Prop.TD_n,set_Prop_c2.TD_n)
append!(set_Prop.banded,set_Prop_c2.banded)
append!(set_Prop.dense,set_Prop_c2.dense)
append!(set_Prop.ncvx,set_Prop_c2.ncvx)
append!(set_Prop.tag,set_Prop_c2.tag)

append!(set_Prop.AtA_diag,set_Prop_sum.AtA_diag)
append!(set_Prop.AtA_offsets,set_Prop_sum.AtA_offsets)
append!(set_Prop.TD_n,set_Prop_sum.TD_n)
append!(set_Prop.banded,set_Prop_sum.banded)
append!(set_Prop.dense,set_Prop_sum.dense)
append!(set_Prop.ncvx,set_Prop_sum.ncvx)
append!(set_Prop.tag,set_Prop_sum.tag)

#if all AtA are banded -> convert to compressed diagonal storage (CDS/DIA) format
if sum(set_Prop.banded[1:s].=true)==s &&  ~joli_op
  for i=1:s
    
    (AtA[i],set_Prop.AtA_offsets[i]) = mat2CDS(AtA[i])
    set_Prop.AtA_offsets[i]=convert(Vector{TI},set_Prop.AtA_offsets[i])
  end
  AtA=convert(Vector{Array{TF,2}}, AtA);
  set_Prop.AtA_offsets=set_Prop.AtA_offsets[1:s]
end

#allocate arrays of vectors
y = Vector{Vector{TF}}(undef,s);
l = Vector{Vector{TF}}(undef,s);

#create one TD_OP that contains all transform-domain operators
TD_OP = Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}}(undef,s)
TD_OP[1:p]     = TD_OP_c1
TD_OP[1+p:p+q] = TD_OP_c2
TD_OP[1+p+q:s] = TD_OP_sum



for ii=1:s #initialize all rho's, gamma's, y's and l's
    y[ii]       = zeros(TF,size(TD_OP[ii],1))
    l[ii]       = zeros(TF,length(y[ii]));
end

if options.parallel==true
  TD_OP = distribute(TD_OP)
  y     = distribute(y)
  l     = distribute(l)
end

return TD_OP,set_Prop,AtA,l,y
end
