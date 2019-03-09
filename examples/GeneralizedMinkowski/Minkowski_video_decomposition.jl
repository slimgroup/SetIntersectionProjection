#background-anomaly separation escalator video with Minkowski decomposition
using Distributed
@everywhere using SetIntersectionProjection
using MAT
using LinearAlgebra
using Statistics

ENV["MPLBACKEND"]="qt5agg"
using PyPlot

#define computational grid structure
@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

#select working precision
TF=Float32 #Float32

#load escalator video) (Mablab files for compatibility with matlab only solvers for comparison...)
if isfile("escalator_data.mat") == true
  file     = matopen("escalator_data.mat")
else
  error("download escalator video from http://cvxr.com/tfocs/demos/rpca/escalator_data.mat")
end
mtrue    = read(file, "X")
n1       = convert(Integer,read(file, "m"))
n2       = convert(Integer,read(file, "n"))
m_mat    = convert(Array{TF,2},mtrue)
m_tensor = convert(Array{TF,3},reshape(mtrue,n1,n2,Integer(200)))

#preprocess data
for i=1:size(m_tensor,3)
  m_tensor[:,:,i].=m_tensor[:,:,i].-median(m_tensor[:,:,i])
end

#split data tensor in a training and evaluation data
m_train      = m_tensor[:,:,181:200] #last 20 frames do not have people -> this should be the background so we learn constraints from it
m_evaluation = m_tensor[:,:,1:180]
m_est        = zeros(TF,size(m_evaluation)) #allocate results


#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid_eval  = compgrid((1f0,1f0,1f0),(size(m_evaluation,1),size(m_evaluation,2), size(m_evaluation,3)))
comp_grid_train = compgrid((1f0,1f0,1f0),(size(m_train,1),size(m_train,2), size(m_train,3)))

#initialize constraint dictionaries
constraint_background = Vector{SetIntersectionProjection.set_definitions}()
constraint_anomaly    = Vector{SetIntersectionProjection.set_definitions}()
constraint_sum        = Vector{SetIntersectionProjection.set_definitions}()

#observe and define bound constraints for background
  #bounds:
  m_min     = minimum(m_train,dims=3)
  m_min     = vec(repeat(m_min[:,:,:],outer=[1,1,comp_grid_eval.n[3]])) .- 40.0
  m_max     = maximum(m_train,dims=3)
  m_max     = vec(repeat(m_max[:,:,:],outer=[1,1,comp_grid_eval.n[3]])) .+ 40.0
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint_background, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #subspace constraints on each z-slice of the tensor
  m_min    = 0.0
  m_max    = 0.0
  set_type = "subspace"
  TD_OP     = "identity"
  A = m_train
  A = reshape(A,size(A,1)*size(A,2),size(A,3))
  F = svd(A)
  app_mode  = ("slice","z")
  custom_TD_OP = (F.U,true)
  push!(constraint_background, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#set bound constraints for sum
  m_min     = minimum(m_tensor) .- 10f0
  m_max     = maximum(m_tensor) .+ 10f0
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint_sum, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#define constraints for anomaly
  #set bound constraints for anomaly (this should be bounds-sum - bounds-background)
  m_min     = minimum(m_tensor) .- 10f0 .- constraint_background[1].min .- 20.0
  m_max     = maximum(m_tensor) .- 10f0 .- constraint_background[1].max .+ 20.0
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint_anomaly, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #set maximum of nonzero pixels in anomaly component for every time-frame
  m_min     = 0
  m_max     = Integer(round(0.25*comp_grid_eval.n[1]*0.25*comp_grid_eval.n[2]))
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("slice","z")
  custom_TD_OP = ([],false)
  push!(constraint_anomaly, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #set maximum of nonzero jumps (x) in pixel values in anomaly component for every time-frame
  m_min     = 0
  m_max     = 4*12*10
  set_type  = "cardinality"
  TD_OP     = "D_x"
  app_mode  = ("slice","z")
  custom_TD_OP = ([],false)
  push!(constraint_anomaly, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #set maximum of nonzero jumps (xy in pixel values in anomaly component for every time-frame
  m_min     = 0
  m_max     = 22*2*10
  set_type  = "cardinality"
  TD_OP     = "D_y"
  app_mode  = ("slice","z")
  custom_TD_OP = ([],false)
  push!(constraint_anomaly, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


#define solver options
options=PARSDMM_options()
options.FL=Float32
options=default_PARSDMM_options(options,options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 100
options.feas_tol               = 0.001
options.obj_tol                = 0.001
options.evol_rel_tol           = 0.00001

set_zero_subnormals(true)
BLAS.set_num_threads(1)
options.parallel         = false
options.feasibility_only = false
options.zero_ini_guess   = true

#set up constraints, precompute some things and define projector
(P_sub_sum,TD_OP_sum,set_Prop_sum) = setup_constraints(constraint_sum,comp_grid_eval,options.FL)
(P_sub_c1,TD_OP_c1,set_Prop_c1)    = setup_constraints(constraint_background,comp_grid_eval,options.FL)
(P_sub_c2,TD_OP_c2,set_Prop_c2)    = setup_constraints(constraint_anomaly,comp_grid_eval,options.FL)

P_sub=deepcopy(P_sub_c1);
append!(P_sub,P_sub_c2)
append!(P_sub,P_sub_sum)

options.Minkowski = true
(TD_OP,set_Prop,AtA,l,y) = PARSDMM_precompute_distribute_Minkowski(TD_OP_c1,TD_OP_c2,TD_OP_sum,set_Prop_c1,set_Prop_c2,set_Prop_sum,comp_grid,options)

options.rho_ini = [1.0f0]
options.rho_ini = ones(length(TD_OP))*options.rho_ini[1]
for i=1:length(options.rho_ini)
  if set_Prop.ncvx[i]==true
    options.rho_ini[i]=0.1f0
  end
end

#compute generalized Minkowski decomposition
@time (v_decomposition,log_PARSDMM,dummy1,dummy2) = PARSDMM(vec(m_evaluation),AtA,TD_OP,set_Prop,P_sub,comp_grid_eval,options);
@time (v_decomposition,log_PARSDMM,dummy1,dummy2) = PARSDMM(vec(m_evaluation),AtA,TD_OP,set_Prop,P_sub,comp_grid_eval,options);

figure();
subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")
tight_layout()

#plot
m_est_background = v_decomposition[1:prod(comp_grid_eval.n)]
m_est_anomaly    = v_decomposition[1+prod(comp_grid_eval.n):end]
m_est_sum        = m_est_background + m_est_anomaly

m_est_background = reshape(m_est_background,comp_grid_eval.n)
m_est_anomaly    = reshape(m_est_anomaly,comp_grid_eval.n)
m_est_sum        = reshape(m_est_sum,comp_grid_eval.n)

figure()
for i=1:4
  subplot(3,4,i);imshow(m_est_background[:,:,i*25],cmap="gray");axis("off");colorbar();title(string("background est, ",i*25)) #title("training image", fontsize=10)
  subplot(3,4,i+4);imshow(m_est_anomaly[:,:,i*25],cmap="gray");axis("off");colorbar();title(string("anomaly est, ",i*25))
  #subplot(4,4,i+8);imshow(m_est_sum[:,:,i*25],cmap="gray");axis("off");colorbar();;title(string("sum, ",i*25))
  subplot(3,4,i+8);imshow(m_evaluation[:,:,i*25],cmap="gray");axis("off");colorbar();;title(string("original, ",i*25))
end

figure()
for i=1:4
  subplot(3,4,i);imshow(m_est_background[:,:,i*25+70],cmap="gray");axis("off");colorbar();title(string("background est, ",i*25)) #title("training image", fontsize=10)
  subplot(3,4,i+4);imshow(m_est_anomaly[:,:,i*25+70],cmap="gray");axis("off");colorbar();title(string("anomaly est, ",i*25))
  subplot(3,4,i+8);imshow(m_evaluation[:,:,i*25+70],cmap="gray");axis("off");colorbar();;title(string("original, ",i*25))
end

#plot results
figure()
for i=1:4
  subplot(3,4,i);imshow(m_est_background[:,:,i*25+70],cmap="gray",vmin=0.0,vmax=255.0);axis("off");colorbar();title(string("background est, ",i*25)) #title("training image", fontsize=10)
  subplot(3,4,i+4);imshow(m_est_anomaly[:,:,i*25+70],cmap="gray",vmin=-150.0,vmax=150.0);axis("off");colorbar();title(string("anomaly est, ",i*25))
  subplot(3,4,i+8);imshow(m_evaluation[:,:,i*25+70],cmap="gray",vmin=0.0,vmax=255.0);axis("off");colorbar();;title(string("original, ",i*25))
end

figure()
for i=1:4
  subplot(3,4,i);imshow(m_est_background[:,:,i*25+70],cmap="gray",vmin=0.0,vmax=255.0);axis("off");colorbar();title(string("background est, ",i*25+70)) #title("training image", fontsize=10)
  subplot(3,4,i+4);imshow(m_est_anomaly[:,:,i*25+70],cmap="gray",vmin=-150.0,vmax=150.0);axis("off");colorbar();title(string("anomaly est, ",i*25+70))
  subplot(3,4,i+8);imshow(m_evaluation[:,:,i*25+70],cmap="gray",vmin=0.0,vmax=255.0);axis("off");colorbar();;title(string("original, ",i*25+70))
end

#play videos
# figure();
# for i=1:size(m_est_background,3)
#   imshow(m_est_background[:,:,i],cmap="gray");title(i)
#   sleep(1.0)
# end
# figure();
# for i=1:size(m_est_anomaly,3)
#   imshow(m_est_anomaly[:,:,i],cmap="gray");title(i)
#   sleep(1.0)
# end
 file = matopen(joinpath(data_dir,"anomaly_est.mat"), "w")
 write(file, "anomaly_est", convert(Array{Float64,3},m_est_anomaly))
 close(file)
 file = matopen(joinpath(data_dir,"background_est.mat"), "w")
 write(file, "background_est", convert(Array{Float64,3},m_est_background))
 close(file)

##################################################################
## Now solve the background-anomaly separation as a matrix problem
##################################################################
#
# #split data tensor in a training and evaluation data
# m_train      = m_mat[:,181:200] #last 20 frames do not have people -> this should be the background so we learn constraints from it
# m_evaluation = m_mat[:,1:180]
# m_est        = zeros(TF,size(m_evaluation)) #allocate results
#
# #learn as a matrix  ( (x times y) -time )
#
# #computational grid for the training images (all images assumed to be on the same grid here)
# comp_grid_eval  = compgrid((1f0,1f0),(size(m_evaluation,1),size(m_evaluation,2)))
# comp_grid_train = compgrid((1f0,1f0),(size(m_train,1),size(m_train,2)))
#
# #initialize constraint dictionaries
# constraint_background = Dict()
# constraint_anomaly    = Dict()
# constraint_sum        = Dict()
#
# #observe and define bound constraints for background
# constraint_background["use_bounds"]=true
# constraint_background["m_max"] = maximum(m_train,3); constraint_background["m_max"]=vec(repeat(constraint_background["m_max"][:,:,:],outer=[1,1,comp_grid_eval.n[3]]))
# constraint_background["m_min"] = minimum(m_train,3); constraint_background["m_min"]=vec(repeat(constraint_background["m_min"][:,:,:],outer=[1,1,comp_grid_eval.n[3]]))
#
#
# #set bound constraints for sum
# constraint_sum["use_bounds"]=true
# constraint_sum["m_max"] = 255f0
# constraint_sum["m_min"] = 0f0
#
# #set bound constraints for anomaly (this should be bounds-sum - bounds-background)
# constraint_anomaly["use_bounds"]=true
# constraint_anomaly["m_max"] = constraint_sum["m_max"]-constraint_background["m_max"]
# constraint_anomaly["m_min"] = constraint_sum["m_min"]-constraint_background["m_min"]
#
# constraint_anomaly["use_rank_slice_z"] = true
# constraint_anomaly["max_rank_slice_z"] = 100
#
# constraint_anomaly["use_TD_card_1"] = false
# constraint_anomaly["TD_card_operator_1"] = "identity"
# constraint_anomaly["card_1"] = Integer(round(0.2*comp_grid_eval.n[1]*0.2*comp_grid_eval.n[2]))
#
# #define solver options
# options=PARSDMM_options()
# options.FL=Float32
# options=default_PARSDMM_options(options,options.FL)
# options.adjust_gamma           = true
# options.adjust_rho             = true
# options.adjust_feasibility_rho = true
# options.Blas_active            = true
# options.maxit                  = 100
# options.feas_tol= 0.001
# options.obj_tol=0.001
# options.evol_rel_tol = 0.00001
#
# options.rho_ini=[0.1]
#
# set_zero_subnormals(true)
# BLAS.set_num_threads(2)
# FFTW.set_num_threads(2)
# options.parallel=false
# options.linear_inv_prob_flag = false
# options.zero_ini_guess=true
#
# #set up constraints, precompute some things and define projector
# (P_sub_sum,TD_OP_sum,set_Prop_sum) = setup_constraints(constraint_sum,comp_grid_eval,options.FL)
# (P_sub_c1,TD_OP_c1,set_Prop_c1)    = setup_constraints(constraint_background,comp_grid_eval,options.FL)
# (P_sub_c2,TD_OP_c2,set_Prop_c2)    = setup_constraints(constraint_anomaly,comp_grid_eval,options.FL)
# P_sub=deepcopy(P_sub_c1);
# append!(P_sub,P_sub_c2)
# append!(P_sub,P_sub_sum)
#
# options.Minkowski=true
# (TD_OP,set_Prop,AtA,l,y) = PARSDMM_precompute_distribute_Minkowski(TD_OP_c1,TD_OP_c2,TD_OP_sum,set_Prop_c1,set_Prop_c2,set_Prop_sum,comp_grid_eval,options)
#
# #compute generalized Minkowski decomposition
#
# (v_decomposition,log_PARSDMM,dummy1,dummy2) = PARSDMM(vec(m_evaluation),AtA,TD_OP,set_Prop,P_sub,comp_grid_eval,options);
#
# figure();
# subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
# subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
# subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
# subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
# subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
# subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
# subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
# subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
# subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")
# tight_layout()
#
# #plot
# m_est_background = v_decomposition[1:prod(comp_grid_eval.n)]
# m_est_anomaly    = v_decomposition[1+prod(comp_grid_eval.n):end]
# m_est_sum        = m_est_background + m_est_anomaly
#
# m_est_background = reshape(m_est_background,comp_grid_eval.n)
# m_est_anomaly    = reshape(m_est_anomaly,comp_grid_eval.n)
# m_est_sum        = reshape(m_est_sum,comp_grid_eval.n)
#
# #plot results
# figure()
# for i=1:4
#   subplot(3,4,i);imshow(m_est_background[:,:,i*25],cmap="gray",vmin=0.0,vmax=255.0);axis("off");colorbar();title(string("background est, ",i*25)) #title("training image", fontsize=10)
#   subplot(3,4,i+4);imshow(m_est_anomaly[:,:,i*25],cmap="gray",vmin=-150.0,vmax=150.0);axis("off");colorbar();title(string("anomaly est, ",i*25))
#   subplot(3,4,i+8);imshow(m_evaluation[:,:,i*25],cmap="gray",vmin=0.0,vmax=255.0);axis("off");colorbar();;title(string("original, ",i*25))
# end
