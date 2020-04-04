# Lax-Wendroff Scheme

struct LaxWendroffScheme <: AbstractFVAlgorithm end

function update_flux_value(uold,node_idx,dt,dx,mesh,Flux,alg::LaxWendroffScheme)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(node_idx,uold,mesh)
    @inbounds ur=cellval_at_right(node_idx,uold,mesh)
    aa = fluxρ(0.5*(ul+ur),Flux)
    # Numerical Fluxes
    return 0.5*(Flux(ul)+Flux(ur))-aa*dt/(2*dx)*(Flux(ur)-Flux(ul))
end
# Ritchmeyer Two-step Lax-Wendroff Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

struct LaxWendroff2sScheme <: AbstractFVAlgorithm end

function update_flux_value(uold,node_idx,dt,dx,mesh,Flux,alg::LaxWendroff2sScheme)
    # Local speeds of propagation
    @inbounds ul=cellval_at_left(node_idx,uold,mesh)
    @inbounds ur=cellval_at_right(node_idx,uold,mesh)
    # Numerical Fluxes
    return Flux(0.5*(ul+ur)-dx/(2*dt)*(Flux(ur)-Flux(ul)))
end
