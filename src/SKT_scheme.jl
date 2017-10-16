# Semidiscrete KT Scheme: Second-Order
# Based on:
# Kurganov, Tadmor. New High Resolution Central Schemes for Non Linear Conservation
# Laws and Convection-Difussion Equations. Journal of Comp Physics 160, pp 241-282. 2000

immutable FVSKTAlgorithm <: AbstractFVAlgorithm
  θ :: Float64
end

function FVSKTAlgorithm(;θ=1.0)
  FVSKTAlgorithm(θ)
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVSKTAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = zeros(u)
    Threads.@threads for j = 1:N
      ul = cellval_at_left(j,u,mesh)
      ur = cellval_at_right(j+1,u,mesh)
      Threads.@threads for i = 1:M
        @inbounds ∇u[j,i] = minmod(θ*(u[j,i]-ul[i]),(ur[i]-ul[i])/2,θ*(ur[i]-u[j,i]))
      end
    end

    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
        @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
        aa = max(fluxρ(uminus,Flux),fluxρ(uplus,Flux))
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(uplus)+Flux(uminus)) - aa/2*(uplus - uminus)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVSKTAlgorithm, ::Type{Val{false}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = zeros(u)
    for j = 1:N
      ul = cellval_at_left(j,u,mesh)
      ur = cellval_at_right(j+1,u,mesh)
      for i = 1:M
        @inbounds ∇u[j,i] = minmod(θ*(u[j,i]-ul[i]),(ur[i]-ul[i])/2,θ*(ur[i]-u[j,i]))
      end
    end

    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
        @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
        aa = max(fluxρ(uminus,Flux),fluxρ(uplus,Flux))
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(uplus)+Flux(uminus)) - aa/2*(uplus - uminus)
    end
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
    #@skt_rhs_header
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
