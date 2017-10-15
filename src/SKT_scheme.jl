# Semidiscrete KT Scheme: Second-Order
# Based on:
# Kurganov, Tadmor. New High Resolution Central Schemes for Non Linear Conservation
# Laws and Convection-Difussion Equations. Journal of Comp Physics 160, pp 241-282. 2000

immutable FVSKTAlgorithm <: AbstractFVAlgorithm
  Θ :: Float64
end

function FVSKTAlgorithm(;Θ=1.0)
  FVSKTAlgorithm(Θ)
end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

@def skt_rhs_header begin
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

  # Numerical Fluxes
  hh = zeros(N+1,M)
  for j = 1:(N+1)
    hh[j,:] = 0.5*(Flux(uplus[j,:])+Flux(uminus[j,:])) -
    aa[j]/2*(uplus[j,:] - uminus[j,:])
  end
  if bdtype == :ZERO_FLUX
    hh[1,:] = 0.0; hh[N+1,:] = 0.0
  end
end

function FV_solve{tType,uType,tAlgType,F}(integrator::FVIntegrator{FVSKTAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F};kwargs...)
  @fv_deterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack Θ = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @boundary_header
    @skt_rhs_header
    @no_diffusion_term
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end

function FV_solve{tType,uType,tAlgType,F,B}(integrator::FVDiffIntegrator{FVSKTAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F,B};kwargs...)
  @fv_diffdeterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack Θ = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @boundary_header
    @skt_rhs_header
    # Diffusion
    pp = zeros(N+1,M)
    ∇u_ap = zeros(uu)
    ∇u_ap[:,:] = ∇u/dx#(uu[2:N,:]-uu[1:N-1,:])/dx
    for j = 1:(N+1)
      pp[j,:] = 0.5*(DiffMat(uu[j,:])+DiffMat(uu[j-1,:]))*∇u_ap[j,1:M]
    end
    if bdtype == :ZERO_FLUX
      pp[1,:] = 0.0; pp[N+1,:] = 0.0
    end
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end
