# Lax-Wendroff Scheme

immutable LaxWendroffAlgorithm <: AbstractFVAlgorithm end

function inner_loop!(hh,j,u,dx,dt,mesh,Flux, alg::LaxWendroffAlgorithm)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(j,u,mesh)
    @inbounds ur=cellval_at_right(j,u,mesh)
    aa = fluxρ(0.5*(ul+ur),Flux)
    # Numerical Fluxes
    @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-aa*dt/(2*dx)*(Flux(ur)-Flux(ul))
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxWendroff2sAlgorithm, ::Type{Val{true}})
Numerical flux of second order lax Wendroff scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxWendroffAlgorithm, ::Type{Val{true}})
    dx = mesh.Δx
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh,j,u,dx,dt,mesh,Flux, alg)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxWendroffAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    for j in edge_indices(mesh)
        inner_loop!(hh,j,u,dx,dt,mesh,Flux, alg)
    end
end

# Ritchmeyer Two-step Lax-Wendroff Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

immutable LaxWendroff2sAlgorithm <: AbstractFVAlgorithm end

function inner_loop!(hh,j,u,dx,dt,mesh,Flux, alg::LaxWendroff2sAlgorithm)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(j,u,mesh)
    @inbounds ur=cellval_at_right(j,u,mesh)
    # Numerical Fluxes
    @inbounds hh[j,:] = Flux(0.5*(ul+ur)-dx/(2*dt)*(Flux(ur)-Flux(ul)))
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxWendroff2sAlgorithm, ::Type{Val{true}})
Numerical flux of second order lax Wendroff scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxWendroff2sAlgorithm, ::Type{Val{true}})
    dx = mesh.Δx
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh,j,u,dx,dt,mesh,Flux, alg)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxWendroff2sAlgorithm, ::Type{Val{false}})
    dx = mesh.Δx
    #update vector
    for j in edge_indices(mesh)
        inner_loop!(hh,j,u,dx,dt,mesh,Flux, alg)
    end
end
