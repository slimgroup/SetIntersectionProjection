export update_y_l_parallel

"""
  update l and compute y . This is a subfunction for PARSDMM.jl
"""
function update_y_l_parallel(
  x                     ::Vector{TF},
  i                     ::Integer,
  Blas_active           ::Bool,
  y                     ::Vector{Vector{TF}},
  y_old                 ::Vector{Vector{TF}},
  l                     ::Vector{Vector{TF}},
  l_old                 ::Vector{Vector{TF}},
  rho                   ::Vector{TF},
  gamma                 ::Vector{TF},
  prox                  ::Vector{Any},
  TD_OP                 ::Vector{<:Union{SparseMatrixCSC{TF,TI},joAbstractLinearOperator{TF,TF}}},
  P_sub,
  x_hat                 ::Vector{Vector{TF}},
  r_pri                 ::Vector{Vector{TF}},
  s                     ::Vector{Vector{TF}},
  set_feas              ::Vector{TF},
  feasibility_only=false::Bool
  ) where {TF<:Real,TI<:Integer}

rho1 = TF
rho1 = TF(1.0) ./ rho[1];

  copy!(y_old[1],y[1]);
  copy!(l_old[1],l[1]);

  s[1] = TD_OP[1]*x;
  if Blas_active
    if gamma[1]==1 #without relaxation
      copyto!(y[1],s[1])
      #y[1]       = prox[1]( BLAS.axpy!(-rho1[1],l[1],y[1]) )
      BLAS.axpy!(-rho1[1],l[1],y[1])
      y[1] = prox[1](y[1])
      @. r_pri[1] = -s[1]+y[1];
      BLAS.axpy!(rho[1],r_pri[1],l[1]);
    else #relaxed iterations
      @. x_hat[1] = gamma[1]*s[1]
      BLAS.axpy!(TF(1.0)-gamma[1],y[1],x_hat[1]);
      #y[1]       = copy(x_hat[1]);
      copyto!(y[1],x_hat[1])
      #y[1]       = prox[1]( BLAS.axpy!(-rho1[1],l[1],y[1]))
      BLAS.axpy!(-rho1[1],l[1],y[1])
      y[1] = prox[1](y[1])
       @. r_pri[1] = -s[1]+y[1];
      BLAS.axpy!(rho[1],y[1]-x_hat[1],l[1]);
    end
  else
    if gamma[1]==1 #without relaxation
      #y[1]       = prox[1]( s[1]-l[1]*rho1[1] );
       @. y[1]       = s[1]-l[1]*rho1[1]
       y[1] = prox[1](y[1]);
       @. r_pri[1]   = -s[1]+y[1];
       @. l[1]       = l[1]+rho[1]*r_pri[1];
    else #relaxed iterations
       @. x_hat[1]   = gamma[1]*s[1] + ( TF(1.0)-gamma[1] )*y[1]
      #y[1]       = prox[1]( x_hat[1]-l[1]*rho1[1] );
       @. y[1]       = x_hat[1]-l[1]*rho1[1]
       y[1] = prox[1]( y[1] )
       @. r_pri[1]   = -s[1]+y[1]
       @. l[1]       = l[1]+rho[1]*( -x_hat[1]+y[1] )
    end
  end #end blas

#log_PARSDMM.r_pri[i,1]  = norm(r_pri[1])#./(norm(y[1])+(100*eps(TF))); #add 1-14, because y may be 0 for certain initializations of x,y,l
#log r_pri outside this function for parallel use (temporarily)
# if Blas_active
#   log_PARSDMM.r_dual[i,1] = norm( BLAS.scal!(lx,rho[1],TD_OP[1]'*(y[1]-y_old[1]),1 ))
# else
#  log_PARSDMM.r_dual[i,1] = norm( rho[1]*(TD_OP[1]'*(y[1]-y_old[1])) )
# end
#log feasibility
if feasibility_only==false
  if mod(i,10)==0 && myid()<nprocs() #ii<length(P_sub)#log every 10 it, or whatever number is suitable
     temp = P_sub[1](deepcopy(s[1]))
     set_feas[1] = norm(temp-s[1]) ./ (norm(s[1])+(100*eps(TF)))
  end
else
  if mod(i,10)==0 #ii<length(P_sub)#log every 10 it, or whatever number is suitable
     temp = P_sub[1](deepcopy(s[1]))
     set_feas[1] = norm(temp-s[1]) ./ (norm(s[1])+(100*eps(TF)))
  end
end



# if mod(i,10)==0
#   counter+=1;
# end

#return y,l,r_pri,s,log_PARSDMM,counter,y_old,l_old,x_hat,set_feas_vec

end #end of function
