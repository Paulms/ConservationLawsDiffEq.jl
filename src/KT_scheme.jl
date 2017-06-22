immutable FVKTAlgorithm <: AbstractFVAlgorithm
  Θ :: Float64
end

function FVKTAlgorithm(;Θ=1.0)
  FVKTAlgorithm(Θ)
end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

@def kt_rhs_header begin
  #Compute diffusion
  λ = dt/dx
  #update vector
  # 1. slopes
  ∇u = zeros(uu)
  for i = 1:M
    for j = 1:N
      ∇u[j,i] = minmod(Θ*(uu[j,i]-uu[j-1,i]),(uu[j+1,i]-uu[j-1,i])/2,Θ*(uu[j+1,i]-uu[j,i]))
    end
  end
  # Local speeds of propagation
  uminus = zeros(N+1,M);uplus=zeros(N+1,M)
  uminus[:,:] = uu[0:N,1:M]+0.5*∇u[0:N,1:M]
  uplus[:,:] = uu[1:N+1,1:M]-0.5*∇u[1:N+1,1:M]
  aa = zeros(N+1)
  for j = 1:N+1
    aa[j]=max(fluxρ(uminus[j,:],Flux),fluxρ(uplus[j,:],Flux))
  end
  #Flux slopes
  u_l = zeros(N+1,M)
  u_r = zeros(N+1,M)
  for i = 1:M
    for j = 1:N+1
      u_l[j,i] = uu[j-1,i] + (0.5-λ*aa[j])*∇u[j-1,i]
      u_r[j,i] = uu[j,i] - (0.5-λ*aa[j])*∇u[j,i]
    end
  end
  ∇f_l = zeros(N+1,M)
  ∇f_r = zeros(N+1,M)
  for j = 2:N
    Ful = Flux(u_l[j,:]); Fulm = Flux(u_l[j-1,:]); Fulp = Flux(u_l[j+1,:])
    Fur = Flux(u_r[j,:]); Furm = Flux(u_r[j-1,:]); Furp = Flux(u_r[j+1,:])
    for i = 1:M
      ∇f_l[j,i] = minmod(Θ*(Ful[i]-Fulm[i]),(Fulp[i]-Fulm[i])/2,Θ*(Fulp[i]-Ful[i]))
      ∇f_r[j,i] = minmod(Θ*(Fur[i]-Furm[i]),(Furp[i]-Furm[i])/2,Θ*(Furp[i]-Fur[i]))
    end
  end

  # Predictor solution values
  Φ_l = u_l - λ/2*∇f_l
  Φ_r = u_r - λ/2*∇f_r

  # Aproximate cell averages
  Ψr = zeros(N+1,M)
  Ψ = zeros(N,M)
  FΦr = zeros(N+1,M)
  FΦl = zeros(N+1,M)
  for j = 1:N+1
    FΦr[j,:] = Flux(Φ_r[j,:])
    FΦl[j,:] = Flux(Φ_l[j,:])
    if (abs(aa[j]) > 1e-6)
      Ψr[j,:] = 0.5*(uu[j-1,:]+uu[j,:])+(1-λ*aa[j])/4*(∇u[j-1,1:M]-∇u[j,1:M])-1/(2*aa[j])*
      (FΦr[j,:]-FΦl[j,:])
    else
      Ψr[j,:] = 0.5*(uu[j-1,:]+uu[j,:])
    end
  end
  Ψ = zeros(uu)
  for j = 1:N
    Ψ[j,1:M] = uu[j,:] - λ/2*(aa[j+1]-aa[j])*∇u[j,1:M]-λ/(1-λ*(aa[j+1]+aa[j]))*
    (FΦl[j+1,:]-FΦr[j,:])
  end
  # Discrete derivatives
  ∇Ψ = zeros(N+1,M)
  for j = 2:N
    for i = 1:M
      ∇Ψ[j,i]=2.0/dx*minmod(Θ*(Ψr[j,i]-Ψ[j-1,i])/(1+λ*(aa[j]-aa[j-1])),
      (Ψ[j,i]-Ψ[j-1,i])/(2+λ*(2*aa[j]-aa[j-1]-aa[j+1])),
      Θ*(Ψ[j,i]-Ψr[j,i])/(1+λ*(aa[j]-aa[j+1])))
    end
  end

  # Numerical Fluxes
  hh = zeros(N+1,M)
  for j = 1:(N+1)
    hh[j,:] = 0.5*(FΦr[j,:]+FΦl[j,:])-0.5*(uu[j,:]-uu[j-1,:])*aa[j]+
    aa[j]*(1-λ*aa[j])/4*(∇u[j,1:M]+∇u[j-1,1:M]) + λ*dx/2*(aa[j])^2*∇Ψ[j,:]
  end
  if bdtype == :ZERO_FLUX
    hh[1,:] = 0.0_dp; hh[N+1,:] = 0.0_dp
  end
end

function FV_solve{tType,uType,tAlgType,F}(integrator::FVIntegrator{FVKTAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F};kwargs...)
  @fv_deterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack Θ = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @boundary_header
    @kt_rhs_header
    @no_diffusion_term
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end

function FV_solve{tType,uType,tAlgType,F,B}(integrator::FVDiffIntegrator{FVKTAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F,B};kwargs...)
  @fv_diffdeterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack Θ = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @boundary_header
    @kt_rhs_header
    # Diffusion
    pp = zeros(N+1,M)
    ∇u_ap = zeros(uu)
    ∇u_ap[:,:] = ∇u/dx#(uu[2:N,:]-uu[1:N-1,:])/dx
    for j = 1:(N+1)
      pp[j,:] = 0.5*(DiffMat(uu[j,:])+DiffMat(uu[j-1,:]))*∇u_ap[j,1:M]
    end
    if bdtype == :ZERO_FLUX
      pp[1,:] = 0.0_dp; pp[N+1,:] = 0.0_dp
    end
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end
