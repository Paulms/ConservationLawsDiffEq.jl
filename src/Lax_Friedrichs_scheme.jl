# Classic Lax-Friedrichs Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

immutable LaxFriedrichsAlgorithm <: AbstractFVAlgorithm end

immutable LocalLaxFriedrichsAlgorithm <: AbstractFVAlgorithm end

mutable struct GlobalLaxFriedrichsAlgorithm{T} <: AbstractFVAlgorithm
  αf :: Function #viscosity coefficient
  α :: T
end

function update_dt(alg::GlobalLaxFriedrichsAlgorithm{T1},u::AbstractArray{T2,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T1,T2}
  alg.α = alg.αf(u,Flux)
  assert(abs(alg.α) > eps(T1))
  CFL*mesh.Δx/alg.α
end

@inline function maxfluxρ(u::AbstractArray,f)
    maxρ = zero(eltype(u))
    N = size(u,1)
    for i in 1:N
      maxρ = max(maxρ, fluxρ(u[i,:],f))
    end
    maxρ
end

function GlobalLaxFriedrichsAlgorithm(;αf = nothing)
    if αf == nothing
        αf = maxfluxρ
    end
    GlobalLaxFriedrichsAlgorithm(αf,0.0)
end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1
"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxFriedrichsAlgorithm, ::Type{Val{true}})
Numerical flux of lax friedrichs algorithm in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxFriedrichsAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxFriedrichsAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul)
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LocalLaxFriedrichsAlgorithm, ::Type{Val{true}})
Numerical flux of local lax friedrichs algorithm in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LocalLaxFriedrichsAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    dx = mesh.Δx
    ul=cellval_at_left(1,u,mesh)
    αl = fluxρ(ul)
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        αr = fluxρ(ur)
        αk = max(αl, αr)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-αk*(ur-ul)
        αl = αr
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LocalLaxFriedrichsAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    dx = mesh.Δx
    ul=cellval_at_left(1,u,mesh)
    αl = fluxρ(ul, Flux)
    #update vector
    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        αr = fluxρ(ur, Flux)
        αk = max(αl, αr)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-αk*(ur-ul)
        αl = αr
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::GlobalLaxFriedrichsAlgorithm, ::Type{Val{true}})
Numerical flux of Global Lax-Friedrichs Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::GlobalLaxFriedrichsAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # update numerical flux
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-alg.α*(ur-ul)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::GlobalLaxFriedrichsAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # update numerical flux
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-alg.α*(ur-ul)
    end
end
