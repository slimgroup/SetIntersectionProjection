export PARSDMM_initialize

"""
subfunctions that initializes and checks, and distributes some quantities required for PARSDMM
"""
function PARSDMM_initialize(
                            x                       ::Vector{TF},
                            l                       ::Union{Vector{Vector{TF}},DistributedArrays.DArray{Array{TF,1},1,Array{Array{TF,1},1}},Array{Any,1}},
                            y                       ::Union{Vector{Vector{TF}},DistributedArrays.DArray{Array{TF,1},1,Array{Array{TF,1},1}},Array{Any,1}},
                            AtA                     ::Union{Vector{Array{TF, 2}}, Vector{SparseMatrixCSC{TF, TI}}, Vector{joAbstractLinearOperator{TF, TF}}, Vector{Union{Array{TF, 2}, SparseMatrixCSC{TF, TI}, joAbstractLinearOperator{TF, TF}}}},
                            TD_OP                   ::Union{Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}},DistributedArrays.DArray{Union{JOLI.joAbstractLinearOperator{TF,TF}, SparseMatrixCSC{TF,TI}},1,Array{Union{JOLI.joAbstractLinearOperator{TF,TF}, SparseMatrixCSC{TF,TI}},1}} },
                            set_Prop,
                            P_sub                   ::Union{Vector{Any},DistributedArrays.DArray{Any,1,Array{Any,1}}},
                            comp_grid,
                            maxit                   ::Integer,
                            rho_ini                 ::Vector{Real},
                            gamma_ini               ::TF,
                            x_min_solver            ::String,
                            rho_update_frequency    ::Integer,
                            adjust_gamma            ::Bool,
                            adjust_rho              ::Bool,
                            adjust_feasibility_rho  ::Bool,
                            m                       ::Vector{TF},
                            parallel                ::Bool,
                            options,
                            zero_ini_guess          ::Bool,
                            feasibility_only=false  ::Bool
                            ) where {TF<:Real,TI<:Integer}

                            ind_ref = maxit
                            if options.Minkowski == false
                              N = length(x)
                            elseif options.Minkowski == true && zero_ini_guess==true
                              N = length(x)*2
                            elseif options.Minkowski == true && zero_ini_guess==false
                              @assert length(x)==(2*length(m))
                              N = length(x)
                            end
                            # const N = size(TD_OP[1],2) #this line will give really weird errors later on for some reason..


                            # if typeof(AtA[1])==SparseMatrixCSC{TF,TI}
                            #   push!(AtA,speye(TF,N));
                            # elseif typeof(AtA[1])==Array{TF,2}
                            #   push!(AtA,ones(TF,N,1))
                            # end
                            # push!(TD_OP,speye(TF,N));
                            #
                            # push!(set_Prop.AtA_offsets,[0])
                            # push!(set_Prop.banded,true)
                            # push!(set_Prop.AtA_diag,true)
                            # push!(set_Prop.dense,false)

                            p     = length(TD_OP); #number of terms in in the sum of functions of the projeciton problem
                            pp=p-1;
                            if feasibility_only==true; pp=p; end;

                            rho     = Vector{TF}(undef,p)
                            if length(rho_ini)==1
                              fill!(rho,rho_ini[1])
                            else
                              copy!(rho,rho_ini)
                            end
                            prox = copy(P_sub)
                            if feasibility_only==false
                              #define prox for all terms in the sum (projectors onto sets)
                              #add prox for the data fidelity term 0.5||m-x||_2^2
                              m_orig = deepcopy(m) ::Vector{TF}
                              prox_data = input -> prox_l2s!(input,rho[end],m_orig)
                              push!(prox,prox_data);
                            end
                            if parallel==true
                              prox  = distribute(prox)
                              P_sub = distribute(P_sub)
                              rho   = distribute(rho)
                            end

                            if parallel==true && nworkers()<length(TD_OP)
                                error("parallel PARSDMM requires at least one JULIA worker per constraint set + 1 (",length(TD_OP),")")
                            end

                            # detect feasibility of model that needs to be projected
                            stop=false
                            feasibility_initial = zeros(TF,length(P_sub))
                            if options.Minkowski == true
                              m = [m ; zeros(TF,length(m)) ]
                            end
                            if parallel
                              feasibility_initial=distribute(feasibility_initial)
                              #[@spawnat pid m for pid in P_sub.pids]
                              [@spawnat pid compute_relative_feasibility(m,feasibility_initial[:L],TD_OP[:L],P_sub[:L]) for pid in P_sub.pids]
                              feasibility_initial=convert(Vector{TF},feasibility_initial)
                            else
                              for ii=1:length(P_sub)
                                feasibility_initial[ii]=norm(P_sub[ii](TD_OP[ii]*m) .- TD_OP[ii]*m) ./ (norm(TD_OP[ii]*m)+(100*eps(TF)));
                              end
                            end
                            if maximum(feasibility_initial)<options.feas_tol #accept input as feasible and return
                                println("input to PARSDMM is feasible, returning")
                                stop=true
                            end

                            # if one of the sets is non-convex, use different lambda and rho update frequency, don't update gamma and set a different fixed gamma
                            for ii=1:pp
                                if set_Prop.ncvx[ii] == true
                                    println("non-convex set(s) involved, using special settings")
                                    rho_update_frequency  = 3;
                                    adjust_gamma          = false
                                    gamma_ini             = TF(0.75)
                                end
                            end

                            #allocate arrays of vectors
                            Ax_out=zeros(TF,N)

                            #if y and l are empty, allocate them and fill with zeros
                            if isempty(l)==true
                              l=Vector{Vector{TF}}(undef,p)
                              for i=1:length(TD_OP); l[i]=zeros(TF,size(TD_OP[i],1)); end
                            end
                            if isempty(y)==true
                              y=Vector{Vector{TF}}(undef,p)
                              for i=1:length(TD_OP); y[i]=zeros(TF,size(TD_OP[i],1)); end
                            end

                            gamma   = Vector{TF}(undef,p);

                            #y       = Vector{Vector{TF}}(p);
                            y_0     = Vector{Vector{TF}}(undef,p);
                            y_old   = Vector{Vector{TF}}(undef,p);

                            #l       = Vector{Vector{TF}}(p);
                            l_0     = Vector{Vector{TF}}(undef,p);
                            l_old   = Vector{Vector{TF}}(undef,p);
                            l_hat_0 = Vector{Vector{TF}}(undef,p);
                            l_hat   = Vector{Vector{TF}}(undef,p)

                            x_0     = Vector{TF}(undef,N);
                            x_old   = Vector{TF}(undef,N);
                            x_hat   = Vector{Vector{TF}}(undef,p);

                            r_dual  = Vector{Vector{TF}}(undef,p);
                            r_pri   = Vector{Vector{TF}}(undef,p);

                            rhs     = Vector{TF}(undef,N);

                            s       = Vector{Vector{TF}}(undef,p);
                            s_0     = Vector{Vector{TF}}(undef,p);

                            d_l_hat = Vector{Vector{TF}}(undef,p)
                            d_H_hat = Vector{Vector{TF}}(undef,p)
                            d_l     = Vector{Vector{TF}}(undef,p)
                            d_G_hat = Vector{Vector{TF}}(undef,p)

                            for ii=1:p #initialize all rho's, gamma's, y's and l's
                                gamma[ii]   = gamma_ini;

                                if parallel == false
                                  ly          = size(TD_OP[ii],1)
                                else
                                 ly = 1
                                end
                                y_0[ii]     = zeros(TF,ly);#copy(y[ii])#zeros(ly);#0.*y[ii];
                                y_old[ii]   = zeros(TF,ly);#copy(y[ii])#zeros(ly);#0.*y[ii];

                                l_old[ii]   = zeros(TF,ly);#0.*y[ii];
                                #l[ii]       = zeros(ly);#0.*y[ii];
                                l_0[ii]     = zeros(TF,ly);#0.*y[ii];
                                l_hat_0[ii] = zeros(TF,ly);#0.*y[ii];
                                l_hat[ii]   = zeros(TF,ly)

                                x_hat[ii]   = zeros(TF,ly)
                                s_0[ii]     = zeros(TF,ly)
                                s[ii]       = zeros(TF,ly)
                                r_pri[ii]   = zeros(TF,ly)

                                d_l_hat[ii] = zeros(TF,ly)
                                d_H_hat[ii] = zeros(TF,ly)
                                d_l[ii]     = zeros(TF,ly)
                                d_G_hat[ii] = zeros(TF,ly)
                            end

                            #assemble total transform domain operator as a matrix
                            if typeof(AtA[1])==SparseMatrixCSC{TF,TI}
                              Q = SparseMatrixCSC{TF,TI}
                              Q= rho[1]*AtA[1];
                              for i=2:p
                                  Q = Q + rho[i]*AtA[i]
                              end
                              Q_offsets=[]
                            elseif typeof(AtA[1])==Array{TF,2}
                              all_offsets=zeros(TI,999,99)
                              for i=1:length(AtA) #find all unique offset
                                all_offsets[1:length(set_Prop.AtA_offsets[i]),i]=set_Prop.AtA_offsets[i]
                              end
                              Q_offsets=convert(Vector{TI},unique(all_offsets))
                              Q=zeros(TF,size(AtA[1],1),length(Q_offsets))
                              for i=1:length(AtA)
                                for j=1:length(set_Prop.AtA_offsets[i])
                                  #Q_current_col = findin(Q_offsets,set_Prop.AtA_offsets[i][j])
                                  Q_current_col = findall((in)(set_Prop.AtA_offsets[i][j]),Q_offsets)
                                  Q[:,Q_current_col] .= Q[:,Q_current_col] .+ rho[i] .* AtA[i][:,j]
                                end
                              end
                            end

                            log_PARSDMM = log_type_PARSDMM(zeros(maxit,pp),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),zeros(maxit),
                            zeros(maxit),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),TF(0),TF(0),TF(0),TF(0),TF(0),TF(0),TF(0));
                            log_PARSDMM.set_feasibility[1,:]=feasibility_initial

                            if parallel==true

                              gamma = distribute(gamma)

                              #y     = distribute(y)
                              y_0   = distribute(y_0)
                              y_old = distribute(y_old)

                              #l       = distribute(l)
                              l_0     = distribute(l_0)
                              l_old   = distribute(l_old)
                              l_hat_0 = distribute(l_hat_0)
                              l_hat   = distribute(l_hat)

                              x_hat   = distribute(x_hat)
                              s_0     = distribute(s_0)
                              s       = distribute(s)
                              r_pri   = distribute(r_pri)

                              d_l_hat = distribute(d_l_hat)
                              d_H_hat = distribute(d_H_hat)
                              d_l     = distribute(d_l)
                              d_G_hat = distribute(d_G_hat)
                            end

                            #fill distributed vectors with zeros (from 1 entry to N entries, because this is faster than first fill and then dist. Should be able to do in one go ideally)
                            if parallel == true
                              [@spawnat pid y_0[:L][1]     = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid y_old[:L][1]   = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid l_0[:L][1]     = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid l_old[:L][1]   = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid l_hat_0[:L][1] = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid l_hat[:L][1]   = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid x_hat[:L][1]   = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid s_0[:L][1]     = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid s[:L][1]       = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid r_pri[:L][1]   = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid d_l_hat[:L][1] = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid d_H_hat[:L][1] = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid d_l[:L][1]     = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                              [@spawnat pid d_G_hat[:L][1] = zeros(TF,size(TD_OP[:L][1],1)) for pid in y_0.pids]
                            end

                            #distribute y and l if they are not already distributed
                            if (parallel==true) && (typeof(l)<:DistributedArrays.DArray) == false
                              l=distribute(l)
                            end
                            if parallel==true && (typeof(y)<:DistributedArrays.DArray) == false
                               y=distribute(y)
                            end
                            if parallel==true
                              [ @spawnat pid adjust_gamma for pid in y.pids ]
                              [ @spawnat pid adjust_rho for pid in y.pids ]
                              [ @spawnat pid adjust_feasibility_rho for pid in y.pids ]
                            end
                            if parallel == true
                              [@spawnat pid comp_grid for pid in  y.pids]
                            end

                            if parallel==true
                              set_feas=zeros(TF,length(TD_OP)) #initialize a vector only required for parallel stuff
                              set_feas=distribute(set_feas)
                            else
                              set_feas=[]
                            end

                            if zero_ini_guess==true
                              if parallel==false
                                for i=1:length(l); fill!(l[i],TF(0.0)); end
                                for i=1:length(y); fill!(y[i],TF(0.0)); end
                              else
                                [@spawnat pid fill!(l[:L][1],TF(0.0)) for pid in l.pids]
                                [@spawnat pid fill!(y[:L][1],TF(0.0)) for pid in y.pids]
                              end
                              fill!(x,TF(0.0))
                            end
return ind_ref,N,TD_OP,AtA,p,rho_update_frequency,adjust_gamma,adjust_rho,adjust_feasibility_rho,gamma_ini,rho,gamma,y,y_0,y_old,
       l,l_0,l_old,l_hat_0,x_0,x_old,r_dual,rhs,s,s_0,Q,prox,log_PARSDMM,l_hat,x_hat,r_pri,
       d_l_hat,d_H_hat,d_l,d_G_hat,P_sub,Q_offsets,stop,feasibility_initial,set_feas,Ax_out
end #end parsdmm_initialize
