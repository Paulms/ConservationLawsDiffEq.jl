# Second-Order upwind central scheme
# Based on:
# Kurganov A., Noelle S., Petrova G., Semidiscrete Central-Upwind schemes
# for hyperbolic Conservation Laws and Hamilton-Jacobi Equations. SIAM. Sci Comput,
# Vol 23, No 3m pp 707-740. 2001

immutable FVCUAlgorithm <: AbstractFVAlgorithm
  Θ :: Float64
end

function FVCUAlgorithm(;Θ=1.0)
  FVCUAlgorithm(Θ)
end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

@def cu_rhs_header begin
  #Compute diffusion
  λ = dt/dx
  #update vector
  # 1. Reconstruct approximate derivatives
  ∇u = zeros(uu)
  for i = 1:M
    for j = 1:N
      ∇u[j,i] = minmod(Θ*(uu[j,i]-uu[j-1,i]),(uu[j+1,i]-uu[j-1,i])/2,Θ*(uu[j+1,i]-uu[j,i]))
    end
  end
  # Local speeds of propagation (Assuming convex flux)
  # A second-order piecewise linear interpolant is used
  uminus = zeros(N+1,M);uplus=zeros(N+1,M)
  uminus[:,:] = uu[0:N,1:M]+0.5*∇u[0:N,1:M]
  uplus[:,:] = uu[1:N+1,1:M]-0.5*∇u[1:N+1,1:M]
  aa_plus = zeros(N+1)
  aa_minus = zeros(N+1)
  for j = 1:N+1
    λm = sort(eigvals(Flux(Val{:jac}, uminus[j,:])))
    λp = sort(eigvals(Flux(Val{:jac}, uplus[j,:])))
    aa_plus[j]=maximum((λm[end], λp[end],0))
    aa_minus[j]=minimum((λm[1], λp[1],0))
  end

    # Numerical Fluxes
  hh = zeros(N+1,M)
  for j = 1:(N+1)
    if abs(aa_plus[j]-aa_minus[j]) < 1e-8
      hh[j,:] = 0.0
    else
      hh[j,:] = (aa_plus[j]*Flux(uminus[j,:])-aa_minus[j]*Flux(uplus[j,:]))/(aa_plus[j]-aa_minus[j]) +
      (aa_plus[j]*aa_minus[j])/(aa_plus[j]-aa_minus[j])*(uplus[j,:] - uminus[j,:])
    end
  end
  if bdtype == :ZERO_FLUX
    hh[1,:] = 0.0; hh[N+1,:] = 0.0
  end
end

function FV_solve{tType,uType,tAlgType,F}(integrator::FVIntegrator{FVCUAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F};kwargs...)
  @fv_deterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack Θ = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @boundary_header
    @cu_rhs_header
    @no_diffusion_term
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end

function FV_solve{tType,uType,tAlgType,F,B}(integrator::FVDiffIntegrator{FVCUAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F,B};kwargs...)
  @fv_diffdeterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack Θ = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @boundary_header
    @cu_rhs_header
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
