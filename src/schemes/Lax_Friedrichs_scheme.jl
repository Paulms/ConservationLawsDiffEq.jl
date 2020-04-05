# Classic Lax-Friedrichs Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

struct LaxFriedrichsScheme <: AbstractFVAlgorithm end

struct LocalLaxFriedrichsScheme <: AbstractFVAlgorithm end

mutable struct GlobalLaxFriedrichsScheme{T} <: AbstractFVAlgorithm
  αf :: Function #viscosity coefficient
  α :: T
end

function update_dt(alg::GlobalLaxFriedrichsScheme{T1},u,Flux,
    CFL,mesh) where {T1}
  alg.α = alg.αf(u, Flux, mesh)
  @assert (abs(alg.α) > eps(T1))
  dx = cell_volume(mesh, 1)
  CFL*dx/alg.α
end

function GlobalLaxFriedrichsScheme(;αf = nothing)
    if αf == nothing
        αf = maxfluxρ
    end
    GlobalLaxFriedrichsScheme(αf,0.0)
end

function update_flux_value(uold,node_idx,dt,dx,mesh,Flux,alg::LaxFriedrichsScheme)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(node_idx,uold,mesh)
    @inbounds ur=cellval_at_right(node_idx,uold,mesh)
    # Numerical Fluxes
    return 0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul)
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LocalLaxFriedrichsScheme, ::Type{Val{true}})
Numerical flux of local lax friedrichs algorithm in 1D
"""
function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::LocalLaxFriedrichsScheme, nonscalar::Bool,::Type{Val{false}})
    dx = cell_volume(mesh, 1)
    ul=cellval_at_left(1,u,mesh)
    αl = fluxρ(ul, Flux)
    #update vector
    for j in node_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        αr = fluxρ(ur, Flux)
        αk = max(αl, αr)
        # Numerical Fluxes
        if nonscalar
            fluxes[:,j] .= 0.5*(Flux(ul)+Flux(ur))-αk*(ur-ul)
        else
            fluxes[j] = 0.5*(Flux(ul)+Flux(ur))-αk*(ur-ul)
        end
        αl = αr
    end
end

function update_flux_value(uold,node_idx,dt,dx,mesh,Flux, alg::GlobalLaxFriedrichsScheme)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(node_idx,uold,mesh)
    @inbounds ur=cellval_at_right(node_idx,uold,mesh)
    # update numerical flux
    return 0.5*(Flux(ul)+Flux(ur))-alg.α*(ur-ul)
end
