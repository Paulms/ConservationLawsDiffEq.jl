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

function compute_slopes(u, mesh, θ, N, M, ::Type{Val{true}})
    ∇u = zeros(u)
    Threads.@threads for j = 1:N
      ul = cellval_at_left(j,u,mesh)
      ur = cellval_at_right(j+1,u,mesh)
      Threads.@threads for i = 1:M
        @inbounds ∇u[j,i] = minmod(θ*(u[j,i]-ul[i]),(ur[i]-ul[i])/2,θ*(ur[i]-u[j,i]))
      end
    end
    ∇u
end

function compute_slopes(u, mesh, θ, N, M, ::Type{Val{false}})
    ∇u = zeros(u)
    for j = 1:N
      ul = cellval_at_left(j,u,mesh)
      ur = cellval_at_right(j+1,u,mesh)
      for i = 1:M
        @inbounds ∇u[j,i] = minmod(θ*(u[j,i]-ul[i]),(ur[i]-ul[i])/2,θ*(ur[i]-u[j,i]))
      end
    end
    ∇u
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVSKTAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{true})

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
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{false})

    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
        @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
        aa = max(fluxρ(uminus,Flux),fluxρ(uplus,Flux))
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(uplus)+Flux(uminus)) - aa/2*(uplus - uminus)
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVSKTAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{true})

    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
        @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
        @inbounds ul = cellval_at_left(j,u,mesh)
        @inbounds ur = cellval_at_right(j,u,mesh)
        aa = max(fluxρ(uminus,Flux),fluxρ(uplus,Flux))
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(uplus)+Flux(uminus)) - aa/2*(uplus - uminus) - 0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j,∇u,mesh)/mesh.Δx
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVSKTAlgorithm, ::Type{Val{false}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{false})

    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
        @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
        @inbounds ul = cellval_at_left(j,u,mesh)
        @inbounds ur = cellval_at_right(j,u,mesh)
        aa = max(fluxρ(uminus,Flux),fluxρ(uplus,Flux))
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(uplus)+Flux(uminus)) - aa/2*(uplus - uminus) - 0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j,∇u,mesh)/mesh.Δx
    end
end
