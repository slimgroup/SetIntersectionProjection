export PARSDMM

"""
Computes the projection onto an intersection of sets
min_x 1/2||x-m||2^2 s.t. x in C_1(A_1*x), x in C_2(A_2*x), ... , x in C_p(A_p*x)

input:

#Arguments
- m                   - vector to be projected
- TD_OP               - vector of linear operators [A_1;A_2;...;A_p]
- set_Prop            - set properties, structure where each property is vector which has a lenght of the number of sets
                            (see SetIntersectionProjection.jl and setup_constraints.jl)
- comp_grid           - structure with grid info. comp_grid.d = (dx,dy,dz), comp_grid.n = (nx,ny,nz)
- P_sub               - vector of projection functions onto C: P_sub = [P_C_1(.); P_C_2(.); ... ; P_C_p(.)]
- options             - structure with options, see SetIntersectionProjection.jl
- x                   - output vector
- l                   - vector of vector of Lagrangian multipliers l = [l_1;l_2,...;l_p] (optional)
- y                   - vector of vector of auxiliary vectors y = [y_1;y_2,...;y_p] (optional)

output:
 - x           -   result
 - log_PARSDMM -   constains log information about various quantities per iteration
"""
function PARSDMM(m         ::Vector{TF},
                 AtA       ::Union{Vector{Array{TF, 2}}, Vector{SparseMatrixCSC{TF, TI}}, Vector{joAbstractLinearOperator{TF, TF}}, Vector{Union{Array{TF, 2}, SparseMatrixCSC{TF, TI}, joAbstractLinearOperator{TF, TF}}}},
                 TD_OP     ::Union{Vector{Union{SparseMatrixCSC{TF,TI},joAbstractLinearOperator{TF,TF}}},DistributedArrays.DArray{Union{JOLI.joAbstractLinearOperator{TF,TF}, SparseMatrixCSC{TF,TI}},1,Array{Union{joAbstractLinearOperator{TF,TF}, SparseMatrixCSC{TF,TI}},1}} },
                 set_Prop,
                 P_sub     ::Union{Vector{Any},DistributedArrays.DArray{Any,1,Array{Any,1}}},
                 comp_grid,
                 options,
                 x=zeros(TF,length(m)) ::Vector{TF},
                 l=[],
                 y=[]
                 ) where {TF<:Real,TI<:Integer}

#tic()
# Parse default options
convert_options!(options,TF)
@unpack  x_min_solver,maxit,evol_rel_tol,feas_tol,obj_tol,rho_ini,rho_update_frequency,gamma_ini,
adjust_rho,adjust_gamma,adjust_feasibility_rho,Blas_active,
feasibility_only,FL,parallel,zero_ini_guess = options

# Input checks
# PARSDMM can work with imaginary numbers, but we want to keep it real
if isreal(m)==false || isreal(x)==false || isreal(l)==false || isreal(y)==false
    error("input for PARSDMM is not real")
end

# initialize
pp=length(TD_OP);
if feasibility_only==false; pp=pp-1; end;

(ind_ref,N,TD_OP,AtA,p,rho_update_frequency,adjust_gamma,adjust_rho,adjust_feasibility_rho,gamma_ini,rho,gamma,y,y_0,y_old,l,l_0,l_old,
l_hat_0,x_0,x_old,r_dual,rhs,s,s_0,Q,prox,log_PARSDMM,l_hat,x_hat,r_pri,d_l_hat,d_H_hat,d_l,
d_G_hat,P_sub,Q_offsets,stop,feasibility_initial,set_feas,Ax_out)=PARSDMM_initialize(x,l,y,AtA,TD_OP,set_Prop,P_sub,comp_grid,maxit,rho_ini,gamma_ini,
x_min_solver,rho_update_frequency,adjust_gamma,adjust_rho,adjust_feasibility_rho,m,parallel,options,zero_ini_guess,feasibility_only)

if stop==true #stop if feasibility of input is detected by PARSDMM_initialize
  copy!(x,m)
  if options.Minkowski == true
    if length(x)==length(m)
      x = [x ; zeros(TF,length(x)) ]
    end
  end
  log_PARSDMM.obj             = log_PARSDMM.obj[[1]]
  log_PARSDMM.evol_x          = log_PARSDMM.evol_x[[1]]
  log_PARSDMM.r_pri_total     = log_PARSDMM.r_pri_total[[1]]
  log_PARSDMM.r_dual_total    = log_PARSDMM.r_dual_total[[1]]
  log_PARSDMM.r_pri           = log_PARSDMM.r_pri[[1],:]
  log_PARSDMM.r_dual          = log_PARSDMM.r_dual[[1],:]
  log_PARSDMM.cg_it           = log_PARSDMM.cg_it[[1]]
  log_PARSDMM.cg_relres       = log_PARSDMM.cg_relres[[1]]
  log_PARSDMM.set_feasibility = log_PARSDMM.set_feasibility[[1],:]
  log_PARSDMM.gamma           = log_PARSDMM.gamma[[1],:]
  log_PARSDMM.rho             = log_PARSDMM.rho[[1],:]
  return x, log_PARSDMM, l, y
end

#make sure proper vectors are supplied as initial guess for Minkowski sets.
if options.Minkowski == true
  if length(x)==length(m)
    x = [x ; zeros(TF,length(m)) ]
  end
end

counter=2

x_solve_tol_ref=TF(1.0) #scalar required to determine tolerance for x-minimization, initial value doesn't matter

log_PARSDMM.T_ini=0.0#toq();;

for i=1:maxit #main loop

  #form right hand side for x-minimization
  #tic();
  rhs               = rhs_compose(rhs,l,y,rho,TD_OP,p,Blas_active,parallel)
  log_PARSDMM.T_rhs = log_PARSDMM.T_rhs+0.0#toq();;

  # x-minimization
  #tic()
  copy!(x_old,x);
  (x,iter,relres,x_solve_tol_ref) = argmin_x(Q,rhs,x,x_min_solver,x_solve_tol_ref,i,log_PARSDMM,Q_offsets,Ax_out)
  log_PARSDMM.cg_it[i]     = iter;#cg_log.iters;
  log_PARSDMM.cg_relres[i] = relres;#cg_log[:resnorm][end];
  log_PARSDMM.T_cg         = log_PARSDMM.T_cg+0.0#toq();;

  # y-minimization & l-update
  #tic()
  if parallel==true
    [ @spawnat pid update_y_l_parallel(x,i,Blas_active,
      y[:L],y_old[:L],l[:L],l_old[:L],rho[:L],gamma[:L],prox[:L],TD_OP[:L],P_sub[:L],
      x_hat[:L],r_pri[:L],s[:L],set_feas[:L],feasibility_only) for pid in y.pids]

    # [@spawnat pid update_y_l_parallel_exp(x,i,Blas_active,
    #   y[:L][1],y_old[:L][1],l[:L][1],l_old[:L][1],rho[:L][1],gamma[:L][1],prox[:L][1],TD_OP[:L][1],P_sub[:L],
    #   x_hat[:L][1],r_pri[:L][1],s[:L][1],set_feas[:L]) for pid in y.pids]
      #logging distributed quantities
      @sync for ii=1:p
        @async log_PARSDMM.r_pri[i,ii]=@fetchfrom r_pri.pids[ii] norm(r_pri[:L][1])
      end
      if mod(i,10)==0 #log every 10 it, or whatever number is suitable
        for ii=1:pp
          log_PARSDMM.set_feasibility[counter,ii]=@fetchfrom set_feas.pids[ii] set_feas[:L][1]
        end
        counter+=1
      end
  else
    (y,l,r_pri,s,log_PARSDMM,counter,y_old,l_old)=update_y_l(x,p,i,Blas_active,y,y_old,l,l_old,rho,gamma,prox,TD_OP,log_PARSDMM,P_sub,counter,x_hat,r_pri,s,feasibility_only);
    log_PARSDMM.r_dual_total[i] = sum(log_PARSDMM.r_dual[i,:]);
  end

  #some more logging
  log_PARSDMM.r_pri_total[i]  = sum(log_PARSDMM.r_pri[i,:]);
  if options.Minkowski == false
    log_PARSDMM.obj[i]          = TF(0.5) .* norm(x .- m)^2
  else
    log_PARSDMM.obj[i]          = TF(0.5) .* norm(TD_OP[end]*x .- m)^2
  end

  log_PARSDMM.evol_x[i]       = norm(x_old .- x) ./ norm(x);
  log_PARSDMM.rho[i,:]        = rho;
  log_PARSDMM.gamma[i,:]      = gamma;

  log_PARSDMM.T_y_l_upd = log_PARSDMM.T_y_l_upd+0.0#toq();;

  # Stopping conditions
  #tic()
  (stop,adjust_rho,adjust_gamma,adjust_feasibility_rho,ind_ref)=stop_PARSDMM(log_PARSDMM,i,evol_rel_tol,feas_tol,obj_tol,adjust_rho,adjust_gamma,adjust_feasibility_rho,ind_ref,counter);
  if stop==true
    (TD_OP,AtA,log_PARSDMM) = output_check_PARSDMM(x,TD_OP,AtA,log_PARSDMM,i,counter)
    return x, log_PARSDMM, l, y
  end
  log_PARSDMM.T_stop=log_PARSDMM.T_stop+0.0#toq();;

  # adjust penalty parameters rho and relaxation parameters gamma
  #tic()
  if i==1
    if parallel==true
      [@spawnat pid l_hat[:L][1] = l_old[:L][1] .+ rho[:L][1] .* ( -s[:L][1] .+ y_old[:L][1] ) for pid in l_hat.pids]
      [@spawnat pid copy!(l_hat_0[:L][1],l_hat[:L][1] ) for pid in l_hat.pids]
      [@spawnat pid copy!(y_0[:L][1] , y[:L][1] ) for pid in y.pids]
      [@spawnat pid copy!(s_0[:L][1] , s[:L][1] ) for pid in s.pids]
      [@spawnat pid copy!(l_0[:L][1] , l[:L][1] ) for pid in l.pids]
    else
      for ii = 1:p
        l_hat[ii]   = l_old[ii] .+ rho[ii] .* ( -s[ii] .+ y_old[ii] );
        copy!(l_hat_0[ii],l_hat[ii])
        copy!(y_0[ii],y[ii])
        copy!(s_0[ii],s[ii])
        copy!(l_0[ii],l[ii])
      end
    end
  end

     if (adjust_rho == true || adjust_gamma == true) && mod(i,rho_update_frequency)==TF(0)
       if parallel==true
         [ @spawnat pid adapt_rho_gamma_parallel(gamma[:L],rho[:L],adjust_gamma,adjust_rho,y[:L],
         y_old[:L],s[:L],s_0[:L],l[:L],l_hat_0[:L],l_0[:L],l_old[:L],y_0[:L],l_hat[:L],d_l_hat[:L],d_H_hat[:L],d_l[:L],d_G_hat[:L]) for pid in y.pids]
         #[ @spawnat pid adapt_rho_gamma_parallel_exp(gamma[:L][1],rho[:L][1],adjust_gamma,adjust_rho,y[:L][1],
         #y_old[:L][1],s[:L][1],s_0[:L][1],l[:L][1],l_hat_0[:L][1],l_0[:L][1],l_old[:L][1],y_0[:L][1],l_hat[:L][1],d_l_hat[:L][1],d_H_hat[:L][1],d_l[:L][1],d_G_hat[:L][1]) for pid in y.pids]
       else
         (rho,gamma,l_hat,d_l_hat,d_H_hat,d_l,d_G_hat)=adapt_rho_gamma(i,gamma,rho,adjust_gamma,adjust_rho,y,y_old,s,s_0,l,l_hat_0,l_0,l_old,y_0,p,l_hat,d_l_hat,d_H_hat,d_l,d_G_hat);
       end
         if i>1
           if parallel==true
             [@spawnat pid copy!(l_hat_0[:L][1],l_hat[:L][1] ) for pid in l_hat.pids]
             [@spawnat pid copy!(y_0[:L][1] , y[:L][1] ) for pid in y.pids]
             [@spawnat pid copy!(s_0[:L][1] , s[:L][1] ) for pid in s.pids]
             [@spawnat pid copy!(l_0[:L][1] , l[:L][1] ) for pid in l.pids]
           else
             for ii=1:p
               copy!(l_hat_0[ii],l_hat[ii])
               copy!(y_0[ii],y[ii])
               copy!(s_0[ii],s[ii])
               copy!(l_0[ii],l[ii])
             end
           end #end if parallel
         end
     end #end adjust rho and gamma

     #adjust rho to set-feasibility estimates
     if parallel==true
        rho=convert(Vector{TF},rho); #gather rho
     end
     if adjust_feasibility_rho == true && mod(i,10)==TF(0) #&& norm(rho-log_PARSDMM.rho[i,:])<(10*eps(TF))#only update rho if it is not already updated
         ## adjust rho feasibility
         #if primal residual for a set is much larger than for the other sets
         #and the feasibility error is also much larger, increase rho to lower
         #primal residual and (hopefully) feasibility error
        (max_set_feas,max_set_feas_ind) = findmax(log_PARSDMM.set_feasibility[counter-1,:])
        sort_feas = sort(log_PARSDMM.set_feasibility[counter-1,:]);
        if i>10
          rho[max_set_feas_ind] = TF(2.0) .* rho[max_set_feas_ind]
        end
     end #end adjust_feasibility_rho

     #enforce max and min values for rho, to prevent the condition number of Q -> inf
     rho = max.(min.(rho,TF(1e4)),TF(1e-2)) #hardcoded bounds
     log_PARSDMM.T_adjust_rho_gamma=log_PARSDMM.T_adjust_rho_gamma+0.0#toq();;

     #tic()
     ind_updated = findall(rho .!= log_PARSDMM.rho[i,:]) # locate changed rho index
     ind_updated = convert(Array{TI,1},ind_updated)

     #re-assemble total transform domain operator as a matrix
     if isempty(findall((in)(ind_updated),p))==false
       if parallel==true && feasibility_only==false
         prox=convert(Vector{Any},prox); #gather rho
         prox[p] = input -> prox_l2s!(input,rho[p],m)
         prox=distribute(prox)
       elseif feasibility_only==false
         prox[p] = input -> prox_l2s!(input,rho[p],m)
       end
     end
     Q=Q_update!(Q,AtA,set_Prop,rho,ind_updated,log_PARSDMM,i,Q_offsets)
     if parallel==true
       rho=distribute(rho) #distribute again (gather -> process -> distribute is a ugly hack, fix this later)
     end
     log_PARSDMM.T_Q_upd=log_PARSDMM.T_Q_upd+0.0#toq();;
     if i==maxit
       println("PARSDMM reached maxit")
       (TD_OP,AtA,log_PARSDMM) = output_check_PARSDMM(x,TD_OP,AtA,log_PARSDMM,i,counter)
     end

 end #end main loop

 # x=P_sub[1](x); #force bound constraints on final output

return x , log_PARSDMM, l , y
end #end funtion

# output checks
function output_check_PARSDMM(x,TD_OP,AtA,log_PARSDMM,i,counter)
  if isreal(x)==false
      println("warning: Result of PARSDMM is not real")
  end
  log_PARSDMM.obj             = log_PARSDMM.obj[1:i]
  log_PARSDMM.evol_x          = log_PARSDMM.evol_x[1:i]
  log_PARSDMM.r_pri_total     = log_PARSDMM.r_pri_total[1:i]
  log_PARSDMM.r_dual_total    = log_PARSDMM.r_dual_total[1:i]
  log_PARSDMM.r_pri           = log_PARSDMM.r_pri[1:i,:]
  log_PARSDMM.r_dual          = log_PARSDMM.r_dual[1:i,:]
  log_PARSDMM.cg_it           = log_PARSDMM.cg_it[1:i]
  log_PARSDMM.cg_relres       = log_PARSDMM.cg_relres[1:i]
  log_PARSDMM.set_feasibility = log_PARSDMM.set_feasibility[1:counter,:]
  log_PARSDMM.gamma           = log_PARSDMM.gamma[1:i,:]
  log_PARSDMM.rho             = log_PARSDMM.rho[1:i,:]

  return TD_OP , AtA , log_PARSDMM
end #end output_check_PARSDMM
