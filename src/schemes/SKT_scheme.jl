# Semidiscrete KT Scheme: Second-Order
# Based on:
# Kurganov, Tadmor. New High Resolution Central Schemes for Non Linear Conservation
# Laws and Convection-Difussion Equations. Journal of Comp Physics 160, pp 241-282. 2000

struct FVSKTScheme{ltype <: AbstractSlopeLimiter} <: AbstractFVAlgorithm
  slopeLimiter :: ltype
end

function FVSKTScheme(;slopeLimiter=GeneralizedMinmodLimiter())
  FVSKTScheme(slopeLimiter)
end

function update_flux_value(u,∇u,j,mesh,Flux, alg::FVSKTScheme)
    # Local speeds of propagation
    uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
    uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
    ul = cellval_at_left(j,u,mesh)
    ur = cellval_at_right(j,u,mesh)
    aa = max(fluxρ(uminus,Flux),fluxρ(uplus,Flux))
    # Numerical Fluxes
    0.5*(Flux(uplus)+Flux(uminus)) - aa/2*(uplus - uminus)
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, alg::FVSKTScheme, ::Type{Val{true}})
Numerical flux of Kurkanov Tadmor scheme in 1D
"""
function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::FVSKTScheme, noscalar::Bool, ::Type{Val{true}})
    slopeLimiter = alg.slopeLimiter
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, slopeLimiter, noscalar, Val{true})

    Threads.@threads for j in node_indices(mesh)
        if noscalar
            fluxes[:,j] .= update_flux_value(u,∇u,j,mesh,Flux,alg)
        else
            fluxes[j] = update_flux_value(u,∇u,j,mesh,Flux,alg)
        end
    end
end

function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::FVSKTScheme, noscalar::Bool, ::Type{Val{false}})
    slopeLimiter = alg.slopeLimiter
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, slopeLimiter, noscalar, Val{false})

    #update vector
    for j in node_indices(mesh)
        if noscalar
            fluxes[:,j] .= update_flux_value(u,∇u,j,mesh,Flux,alg)
        else
            fluxes[j] = update_flux_value(u,∇u,j,mesh,Flux,alg)
        end
    end
end
