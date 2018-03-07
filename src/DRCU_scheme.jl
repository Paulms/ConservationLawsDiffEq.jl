# Dissipation Reduced Central upwind Scheme: Second-Order
# Based on:
# Kurganov A., Lin C., On the reduction of Numerical Dissipation in Central-Upwind
# Schemes, Commun. Comput. Phys. Vol 2. No. 1, pp 141-163, Feb 2007.

immutable FVDRCUAlgorithm <: AbstractFVAlgorithm
  θ :: Float64
end

function FVDRCUAlgorithm(;θ=1.0)
  FVDRCUAlgorithm(θ)
end

function inner_loop!(hh,j,u,∇u,mesh,Flux, alg::FVDRCUAlgorithm)
    @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
    @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)

    λm = sort(eigvals(Flux(Val{:jac}, uminus)))
    λp = sort(eigvals(Flux(Val{:jac}, uplus)))
    aa_plus=maximum((λm[end], λp[end],0))
    aa_minus=minimum((λm[1], λp[1],0))
    # Update numerical fluxes
    if abs(aa_plus-aa_minus) < 1e-8
      hh[j,:] = 0.5*(Flux(uminus)+Flux(uplus))
    else
      flm = Flux(uminus); flp = Flux(uplus)
      wint = 1/(aa_plus-aa_minus)*(aa_plus*uplus-aa_minus*uminus-
      (flp-flm))
      qj = minmod.((uplus-wint)/(aa_plus-aa_minus),(wint-uminus)/(aa_plus-aa_minus))
      hh[j,:] = (aa_plus*flm-aa_minus*flp)/(aa_plus-aa_minus) +
      (aa_plus*aa_minus)*((uplus - uminus)/(aa_plus-aa_minus) - qj)
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVDRCUAlgorithm, ::Type{Val{true}})
Numerical flux of Second-Order dissipation reduced upwind central scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVDRCUAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, M, Val{true})

    # Local speeds of propagation (Assuming convex flux)
    # A second-order piecewise linear interpolant is used
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh,j,u,∇u,mesh,Flux, alg)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVDRCUAlgorithm, ::Type{Val{false}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, M, Val{false})

    # Local speeds of propagation (Assuming convex flux)
    # A second-order piecewise linear interpolant is used
    for j in edge_indices(mesh)
        inner_loop!(hh,j,u,∇u,mesh,Flux, alg)
    end
end

function inner_loop!(hh,j,u,∇u,mesh,Flux, DiffMat, alg::FVDRCUAlgorithm)
    @inbounds uminus=cellval_at_left(j,u,mesh)+0.5*cellval_at_left(j,∇u,mesh)
    @inbounds uplus=cellval_at_right(j,u,mesh)-0.5*cellval_at_right(j,∇u,mesh)
    @inbounds ul = cellval_at_left(j,u,mesh)
    @inbounds ur = cellval_at_right(j,u,mesh)

    λm = sort(eigvals(Flux(Val{:jac}, uminus)))
    λp = sort(eigvals(Flux(Val{:jac}, uplus)))
    aa_plus=maximum((λm[end], λp[end],0))
    aa_minus=minimum((λm[1], λp[1],0))
    # Update numerical fluxes
    if abs(aa_plus-aa_minus) < 1e-8
      hh[j,:] = 0.5*(Flux(uminus)+Flux(uplus)) - 0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j,∇u,mesh)/mesh.Δx
    else
      flm = Flux(uminus); flp = Flux(uplus)
      wint = 1/(aa_plus-aa_minus)*(aa_plus*uplus-aa_minus*uminus-
      (flp-flm))
      qj = minmod.((uplus-wint)/(aa_plus-aa_minus),(wint-uminus)/(aa_plus-aa_minus))
      hh[j,:] = (aa_plus*flm-aa_minus*flp)/(aa_plus-aa_minus) +
      (aa_plus*aa_minus)*((uplus - uminus)/(aa_plus-aa_minus) - qj) -
      0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_left(j,∇u,mesh)/mesh.Δx
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVDRCUAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, M, Val{true})

    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh,j,u,∇u,mesh,Flux, DiffMat, alg)
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVDRCUAlgorithm, ::Type{Val{false}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, M, Val{false})

    for j in edge_indices(mesh)
        inner_loop!(hh,j,u,∇u,mesh,Flux, DiffMat, alg)
    end
end
