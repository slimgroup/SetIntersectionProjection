# test projections onto intersection for julia for image inpainting/matrix completion
# as set theoretical image recovery problem (a feasibility problem)
# We will use a VERY simple learning appraoch to obtain 'good' constraints. This
# learing works with just a few even <10 training examples.
# South America dataset

@everywhere include("../src/SetIntersectionProjection.jl")
@everywhere using SetIntersectionProjection
@everywhere using MAT
@everywhere using PyPlot
#using LatexStrings

@everywhere type compgrid
  d :: Tuple
  n :: Tuple
end

#select working precision
FL=32
if     FL==64
  TF = Float64
  TI = Int64
elseif FL==32
  TF = Float32
  TI = Int32
end

#load a very small data set (12 images only) (Mablab files for compatibility with matlab only solvers for comparison...)
file = matopen("SA_patches.mat")
mtrue=read(file, "SA_patches")
mtrue=convert(Array{TF,3},mtrue)

#split data tensor in a training and evaluation data
#patches have been randomized already in matlab
m_train = mtrue[1:35,:,:]
m_evaluation = mtrue[36:39,:,:]
m_est = zeros(TF,size(m_evaluation))

#plot training images
figure();title("training image", fontsize=10)
for i=1:size(m_train,1)
  subplot(7,5,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); #title("training image", fontsize=10)
end
savefig("training_data_all.pdf",bbox_inches="tight")

for i=1:35
  figure();title(string("training image", i), fontsize=10)
  imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
  savefig(string("training_data_", i,".pdf"),bbox_inches="tight")
end

#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid = compgrid((1, 1),(size(m_evaluation,2), size(m_evaluation,3)))
#convert model and computational grid parameters
comp_grid=compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]) ), comp_grid.n )

#create true observed data by blurring and setting pixels to zero
d_obs = zeros(TF,size(m_evaluation,1),comp_grid.n[1]-25,comp_grid.n[2])
n1=comp_grid.n[1]
Bx=speye(n1)./25
for i=1:25
temp=  spdiagm(ones(n1)./25,i)
temp=temp[1:n1,1:n1];
Bx+= temp;
end
Bx=Bx[1:end-25,:]
Iz=speye(TF,comp_grid.n[2]);
BF=kron(Iz,Bx);
BF=convert(SparseMatrixCSC{TF,TI},BF);

#second: subsample
(e1,e2,e3)=size(d_obs)
mask = ones(TF,e2*e3)
s    = randperm(e3*e2)
zero_ind = s[1:Int(8.*round((e2*e3)/10))]
mask[zero_ind].=0.0f0
mask = spdiagm(mask,0)
FWD_OP = convert(SparseMatrixCSC{TF,TI},mask*BF)

#blur and subsample
for i=1:size(d_obs,1)
  d_obs[i,:,:] = reshape(FWD_OP*vec(m_evaluation[i,:,:]),comp_grid.n[1]-25,comp_grid.n[2])
end



#"train" by observing constraints on the data images in training data set
observations = constraint_learning_by_obseration(comp_grid,m_train)

#define a few constraints and what to do with the observations
constraint=Dict()
constraint["use_bounds"]=true
constraint["m_min"]=0.0
constraint["m_max"]=255.0

constraint["use_rank"]=true;
observations["rank_095"]=sort(vec(observations["rank_095"]))
constraint["max_rank"] = convert(TI,round(quantile(observations["rank_095"],0.50)))

constraint["use_TD_nuclear_1"]=true;
constraint["TD_nuclear_operator_1"]="identity"
constraint["TD_nuclear_norm_1"] = convert(TF,quantile(vec(observations["nuclear_norm"]),0.50))

constraint["use_TD_nuclear_2"]=true;
constraint["TD_nuclear_operator_2"]="D_x"
constraint["TD_nuclear_norm_2"] = convert(TF,quantile(vec(observations["nuclear_Dx"]),0.50))

constraint["use_TD_nuclear_3"]=true;
constraint["TD_nuclear_operator_3"]="D_z"
constraint["TD_nuclear_norm_3"] = convert(TF,quantile(vec(observations["nuclear_Dz"]),0.50))

constraint["use_TD_l1_1"]=true
constraint["TD_l1_operator_1"]="TV"
constraint["TD_l1_sigma_1"] = convert(TF,quantile(vec(observations["TV"]),0.50))

constraint["use_TD_l2_1"]=true
constraint["TD_l2_operator_1"]="TV"
constraint["TD_l2_sigma_1"] = convert(TF,quantile(vec(observations["D_l2"]),0.50))

constraint["use_TD_l1_2"]=false
constraint["TD_l1_operator_2"]="curvelet"
constraint["TD_l1_sigma_2"] = 0.5f0*convert(TF,quantile(vec(observations["curvelet_l1"]),0.50))

constraint["use_TD_l1_3"]=true
constraint["TD_l1_operator_3"]="DFT"
constraint["TD_l1_sigma_3"] = convert(TF,quantile(vec(observations["DFT_l1"]),0.50))

constraint["use_TD_bounds_1"]=true
constraint["TDB_operator_1"]="D_x"
constraint["TD_LB_1"]=convert(TF,quantile(vec(observations["D_x_min"]),0.15))
constraint["TD_UB_1"]=convert(TF,quantile(vec(observations["D_x_max"]),0.85))

constraint["use_TD_bounds_2"]=true
constraint["TDB_operator_2"]="D_z"
constraint["TD_LB_2"]=convert(TF,quantile(vec(observations["D_z_min"]),0.15))
constraint["TD_UB_2"]=convert(TF,quantile(vec(observations["D_z_max"]),0.85))
#
# constraint["use_TD_bounds_3"]=true
# constraint["TDB_operator_3"]="DCT"
# constraint["TD_LB_3"]=observations["DCT_LB"]
# constraint["TD_UB_3"]=observations["DCT_UB"]
#
# constraint["TD_UB_3"]=reshape(constraint["TD_UB_3"],comp_grid.n)
# constraint["TD_LB_3"]=reshape(constraint["TD_LB_3"],comp_grid.n)
# temp1=zeros(TF,comp_grid.n)
# temp2=zeros(TF,comp_grid.n)
# for j=2:size(constraint["TD_LB_3"],1)-1
#   for k=2:size(constraint["TD_LB_3"],2)-1
#     temp1[j,k].=minimum(constraint["TD_LB_3"][j-1:j+1,k-1:k+1]);
#     temp2[j,k].=maximum(constraint["TD_UB_3"][j-1:j+1,k-1:k+1]);
#   end
# end
# temp1[1,:]=constraint["TD_LB_3"][1,:];
# temp1[:,1]=constraint["TD_LB_3"][:,1]
# temp1[end,:]=constraint["TD_LB_3"][end,:]
# temp1[:,end]=constraint["TD_LB_3"][:,end]
#
# temp2[1,:]=constraint["TD_UB_3"][1,:]
# temp2[:,1]=constraint["TD_UB_3"][:,1]
# temp2[end,:]=constraint["TD_UB_3"][end,:]
# temp2[:,end]=constraint["TD_UB_3"][:,end]
#
# for j=1:size(constraint["TD_LB_3"],1)-1
#   for k=1:size(constraint["TD_LB_3"],2)-1
#     temp1[j,k].=minimum(constraint["TD_LB_3"][j:j+1,k:k+1]);
#     temp2[j,k].=maximum(constraint["TD_UB_3"][j:j+1,k:k+1]);
#   end
# end
# for j=2:size(constraint["TD_LB_3"],1)
#   for k=2:size(constraint["TD_LB_3"],2)
#     temp1[j,k].=minimum(constraint["TD_LB_3"][j-1:j,k-1:k]);
#     temp2[j,k].=maximum(constraint["TD_UB_3"][j-1:j,k-1:k]);
#   end
# end
#
# constraint["TD_LB_3"]=vec(temp1)
# constraint["TD_UB_3"]=vec(temp2)


constraint["use_TD_card_1"]=false
constraint["TD_card_operator_1"]="curvelet"
constraint["card_1"]=convert(TI,round(quantile(vec(observations["curvelet_card_095"]),0.85)))

constraint["use_TD_card_2"]=false
constraint["TD_card_operator_2"]="TV"
constraint["card_2"]=convert(TI,round(quantile(vec(observations["TV_card_095"]),0.85)))

#PARSDMM options:
options=PARSDMM_options()
options=default_PARSDMM_options(options,options.FL)
options.evol_rel_tol=1f-6
options.feas_tol=0.001f0
options.obj_tol=0.0002f0
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 5000
set_zero_subnormals(true)

options.linear_inv_prob_flag = true #this is important to set
options.parallel             = true
options.zero_ini_guess       = true
BLAS.set_num_threads(2)

multi_level=false
n_levels=2
coarsening_factor=2.0
#(m_est,mask_save)=ICLIP_inpainting(FWD_OP,d_obs,m_evaluation,constraint,comp_grid,options,multi_level,n_levels,coarsening_factor)

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)

#add the mask*blurring filer sparse matrix as a transform domain matrix
push!(TD_OP,FWD_OP)
push!(TD_Prop.AtA_offsets,convert(Vector{TI},0:25))
push!(TD_Prop.ncvx,false)
push!(TD_Prop.banded,true)
push!(TD_Prop.AtA_diag,true)
push!(TD_Prop.dense,false)
push!(TD_Prop.tag,("subsampling blurring filer","x-motion-blur"))

#also add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u
data = vec(d_obs[1,:,:])
LBD=data.-2.0;  LBD=convert(Vector{TF},LBD)
UBD=data.+2.0;  UBD=convert(Vector{TF},UBD)
push!(P_sub,x -> project_bounds!(x,LBD,UBD))

dummy=zeros(TF,size(BF,2))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(dummy,TD_OP,TD_Prop,options)

for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
  @time (x,log_PARSDMM) = PARSDMM(dummy,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  println("SNR:", SNR(vec(m_evaluation[i,:,:]),x))
  m_est[i,:,:]=reshape(x,comp_grid.n)

  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
    UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
    P_sub[end] = x -> project_bounds!(x,LBD,UBD)
  end
end


SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

for i=1:size(m_est,1)
    figure();imshow(d_obs[i,26:end-26,:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(string("deblurring_observed",i,".pdf"),bbox_inches="tight")
    figure();imshow(m_est[i,26:end-26,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,26:end-26,:]),vec(m_est[i,26:end-26,:])),2)))
    savefig(string("PARSDMM_deblurring",i,".pdf"),bbox_inches="tight")
    figure();imshow(m_evaluation[i,26:end-26,:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(string("deblurring_evaluation",i,".pdf"),bbox_inches="tight")

end

#test TV and bounds only, to see if all the other constraints contribute anything in reconstructuion quality
#
# #constraints
# constraint=Dict()
# constraint["use_bounds"]=true
# constraint["m_min"]=0.0
# constraint["m_max"]=255.0
#
# constraint["use_TD_l1_1"]=true
# constraint["TD_l1_operator_1"]="TV"
# constraint["TD_l1_sigma_1"] = convert(TF,quantile(vec(observations["TV"]),0.50))
#
# (m_est,mask_save)=ICLIP_inpainting(FWD_OP,d_obs,m_evaluation,constraint,comp_grid,options,multi_level,n_levels,coarsening_factor)
#
# for i=1:size(m_est,1)
#   figure()
#   subplot(3,1,1);imshow(d_obs[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("observed")
#   subplot(3,1,2);imshow(m_est[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("Reconstruction")
#   subplot(3,1,3);imshow(m_evaluation[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
# end

#
# file = matopen("m_evaluation.mat", "w")
# write(file, "m_evaluation", convert(Array{Float64,3},m_evaluation))
# close(file)
#
# file = matopen("m_train.mat", "w")
# write(file, "m_train", convert(Array{Float64,3},m_train))
# close(file)

file = matopen("d_obs.mat", "w")
write(file, "d_obs", convert(Array{Float64,3},d_obs))
close(file)

file = matopen("FWD_OP.mat", "w")
write(file, "FWD_OP", convert(SparseMatrixCSC{Float64,Int64},FWD_OP))
close(file)

# (TV_OP, dummy1, dummy2, dummy3)=get_TD_operator(comp_grid,"TV",TF)
# file = matopen("TV_OP.mat", "w")
# write(file, "TV_OP", convert(SparseMatrixCSC{Float64,Int64},TV_OP))
# close(file)
#
#load TFOCS matlab results and plot
# file = matopen("x_TFOCS_tv_save_SA.mat")
# x_TFOCS_tv_save=read(file, "x_TFOCS_tv_save_SA")
SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
for i=1:size(m_evaluation,1)
  figure()
  imshow(x_TFOCS_tv_save[i,26:end-26,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("TFOCS BPDN-TV, SNR=", round(SNR(vec(m_evaluation[i,26:end-26,:]),vec(x_TFOCS_tv_save[i,26:end-26,:])),2)))
  savefig(string("TFOCS_TV_inpainting",i,".pdf"),bbox_inches="tight")
end
#

file = matopen("x_TFOCS_W_save_SA.mat")
x_TFOCS_tv_save=read(file, "x_TFOCS_W_save_SA")
x_TFOCS_tv_save=reshape(x_TFOCS_tv_save,4,comp_grid.n[1],comp_grid.n[2])

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
for i=1:size(m_evaluation,1)
  figure()
  imshow(x_TFOCS_tv_save[i,26:end-26,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("TFOCS BPDN-Wavelet, SNR=", round(SNR(vec(m_evaluation[i,26:end-26,:]),vec(x_TFOCS_tv_save[i,26:end-26,:])),2)))
  savefig(string("TFOCS_Wavelet_inpainting",i,".pdf"),bbox_inches="tight")
end


file = matopen("x_SPGL1_C2_save_SA.mat")
x_TFOCS_tv_save=read(file, "x_SPGL1_C2_save_SA")
x_TFOCS_tv_save=reshape(x_TFOCS_tv_save,4,comp_grid.n[1],comp_grid.n[2])

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
for i=1:size(m_evaluation,1)
  figure()
  imshow(x_TFOCS_tv_save[i,26:end-26,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("SPGL1 BPDN-curvelet, SNR=", round(SNR(vec(m_evaluation[i,26:end-26,:]),vec(x_TFOCS_tv_save[i,26:end-26,:])),2)))
  savefig(string("SPGL1_curvelet_inpainting",i,".pdf"),bbox_inches="tight")
end


#
# #plot zoomed section
# z_parsdmm_1 = m_est[1,1:400,1700:end]
# z_TFOCS_1   = reshape(x_TFOCS_tv_save[1,:,:],comp_grid.n)
# z_TFOCS_1   = z_TFOCS_1[1:400,1700:2051];
# z_true_1    = m_evaluation[1,1:400,1700:end];
#
# z_parsdmm_2 = m_est[2,500:end,600:800]
# z_TFOCS_2   = reshape(x_TFOCS_tv_save[2,:,:],comp_grid.n)
# z_TFOCS_2   = z_TFOCS_2[500:926,600:800];
# z_true_2    = m_evaluation[2,500:end,600:800];
#
# figure();imshow(z_parsdmm_1,cmap="gray",vmin=0.0,vmax=255.0); title("PARSDMM zoomed")
# savefig(string("PARSDMM_inpainting_zoomed_1.pdf"),bbox_inches="tight")
# figure();imshow(z_parsdmm_2,cmap="gray",vmin=0.0,vmax=255.0); title("PARSDMM zoomed")
# savefig(string("PARSDMM_inpainting_zoomed_2.pdf"),bbox_inches="tight")
#
# figure();imshow(z_TFOCS_1,cmap="gray",vmin=0.0,vmax=255.0); title("BPDN-TV zoomed")
# savefig(string("TFOCS_inpainting_zoomed_1.pdf"),bbox_inches="tight")
# figure();imshow(z_TFOCS_2,cmap="gray",vmin=0.0,vmax=255.0); title("BPDN-TV zoomed")
# savefig(string("TFOCS_inpainting_zoomed_2.pdf"),bbox_inches="tight")
#
# figure();imshow(z_true_1,cmap="gray",vmin=0.0,vmax=255.0); title("true zoomed")
# savefig(string("true_inpainting_zoomed_1.pdf"),bbox_inches="tight")
# figure();imshow(z_true_2,cmap="gray",vmin=0.0,vmax=255.0); title("true zoomed")
# savefig(string("true_inpainting_zoomed_2.pdf"),bbox_inches="tight")

# ###### can't get it to work
# TF=Float32
# TI=Int32
# n1=20;
# n2=31;
# m=randn(TF,n1*n2)
# mask = zeros(TF,length(m))
# zero_ind=randperm(length(m));
# zero_ind=zero_ind[1:20*21];
# m[zero_ind].=0.0
# mask[find(m)]=TF(1.0) #a 1 for observed entries
# mask_mat=spdiagm(mask,0); #create diagonal sparse matrix
# mask_mat=convert(SparseMatrixCSC{TF,TI},mask_mat)

#
# using GenSPGL
#
# #use GNSPGL1 for LR factor minimization
# # opts = spgOptions(project     = TraceNorm_project,
# #                   primal_norm = TraceNorm_primal,
# #                   dual_norm   = TraceNorm_dual,
# #                   proxy = true)
# # opts.iterations=25;
# #  #
# #
# #  afunT(x) = reshape(x,n1,n2)
# #     params = Dict("nr"=> 10,
# #                     "Ind"=> vec(m) .== 0,
# #                     "numr"=> n1,
# #                     "numc"=> n2,
# #                     "funForward"=> NLfunForward,
# #                     "afunT"=> afunT,
# #                     "afun"=> afun,
# #                     "mode"=> 1,
# #                     "ls"=> 1,
# #                     "logical"=> 0,
# #                     "funPenalty"=> funLS)
# # #x_gspgl, r, g, info = spgl1(mask_mat,m,tau = 0.0f0,sigma=2756.24817460257f0,options = opts)
# # LR_ini=randn(TF,n1*10+n2*10)
# #
# #  x_gspgl, r, g, info = spgl1(NLfunForward,m,sigma=1.234,options = opts,params=params,x=LR_ini)
#
# ###### #use GNSPGL1 with sparse matrix
# params=Dict{String,Any}()
# params["numr"]=comp_grid.n[1]
# params["numc"]=comp_grid.n[2]
# params["nr"]=100
#
# opts = spgOptions(project     = TraceNorm_project,
#                   primal_norm = TraceNorm_primal,
#                   dual_norm   = TraceNorm_dual)
# opts.iterations=50;
#
# temp=diag(mask_mat)
# Restriction=mask_mat[find(temp),:];
# data=m[find(temp)];
#
# x_gspgl, r, g, info = spgl1(Restriction,data,x=1.0.*m,tau=1.5224746e7,sigma=87.0,options = opts,params=params)
#
# x_gspgl2, r, g, info = spgl1(mask_mat,m,sigma=2756.24817460257,options = opts,params=params,x=1f6.*vec(randn(Float32,comp_grid.n[1],comp_grid.n[2])))
#
# figure();imshow(reshape(x_gspgl,comp_grid.n),cmap="gray",vmin=0.0,vmax=255.0); title("Reconstruction nuclear norm minimization")
#
#
# figure();imshow(reshape(m,comp_grid.n),cmap="gray",vmin=0.0,vmax=255.0); title("Reconstruction nuclear norm minimization")

# #does it work for TV??? TV is not a norm, nor is the TV matrix orthogonal or square
# opts = spgOptions(verbosity = 1)
# opts.iterations=25;
# m=convert(Vector{Float64},m)
# TV_OP=convert(SparseMatrixCSC{Float64,Int64},TD_OP[4])
# MM=convert(SparseMatrixCSC{Float64,Int64},mask_mat)
#
#
# x, r, g, info = spgl1(TV_OP*MM, TV_OP*m, tau = 0.0,sigma=2756.24817460257,options = opts)


#
# ############ use Convex.jl ############ runs out of memory for a small image already...
# # Let us first make the Convex.jl module available
# using Convex
#
# nucelar norm minimization:
# # Create a (column vector) variable of size n x 1. (Does not seem to work with Float32...)
# X = Variable(Int64(comp_grid.n[1]),Int64(comp_grid.n[2]))
# M=reshape(convert(Vector{Float32},m),Int64(comp_grid.n[1]),Int64(comp_grid.n[2]))
# #convert(Vector{Float32},m)
#
# # The problem is to minimize ||Ax - b||^2 subject to x >= 0
# # This can be done by: minimize(objective, constraints)
# problem = minimize(nuclearnorm(X), vecnorm(X-M, 2)<= 2756.2)
# # Solve the problem by calling solve!
# solve!(problem)
#
# # Check the status of the problem
# problem.status # :Optimal, :Infeasible, :Unbounded etc.
#
# println("relative l2 error:", norm(vec(X.value)-vec(mt))/norm(vec(mt)) )
# SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
# println("SNR:", SNR(vec(mt),vec(X.value)))
#
# figure();imshow(reshape(m,comp_grid.n),cmap="jet"); title("observed")
# figure();imshow(X.value,cmap="jet"); title("Reconstruction")
# figure();imshow(reshape(m_evaluation[10,10,:,:],comp_grid.n),cmap="jet"); title("True")

#TV-minimization
#x = Variable(Int64(comp_grid.n[1])*Int64(comp_grid.n[2]))
#problem = minimize(norm_1(TD_OP[2]*x), norm(x-m, 2)<= 2756.2)
