# Second-Order upwind central scheme
# Based on:
# Kurganov A., Noelle S., Petrova G., Semidiscrete Central-Upwind schemes
# for hyperbolic Conservation Laws and Hamilton-Jacobi Equations. SIAM. Sci Comput,
# Vol 23, No 3m pp 707-740. 2001

immutable FVCUAlgorithm <: AbstractFVAlgorithm
  θ :: Float64
end

function FVCUAlgorithm(;θ=1.0)
  FVCUAlgorithm(θ)
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCUAlgorithm, ::Type{Val{true}})
Numerical flux of Second-Order upwind central scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCUAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{true})

    # Local speeds of propagation (Assuming convex flux)
    # A second-order piecewise linear interpolant is used
    Threads.@threads for j in edge_indices(mesh)
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
        hh[j,:] = (aa_plus*Flux(uminus)-aa_minus*Flux(uplus))/(aa_plus-aa_minus) +
        (aa_plus*aa_minus)/(aa_plus-aa_minus)*(uplus - uminus)
      end
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCUAlgorithm, ::Type{Val{false}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{false})

    # Local speeds of propagation (Assuming convex flux)
    # A second-order piecewise linear interpolant is used
    for j in edge_indices(mesh)
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
        hh[j,:] = (aa_plus*Flux(uminus)-aa_minus*Flux(uplus))/(aa_plus-aa_minus) +
        (aa_plus*aa_minus)/(aa_plus-aa_minus)*(uplus - uminus)
      end
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVCUAlgorithm, ::Type{Val{true}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{true})

    Threads.@threads for j in edge_indices(mesh)
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
          hh[j,:] = (aa_plus*Flux(uminus)-aa_minus*Flux(uplus))/(aa_plus-aa_minus) +
          (aa_plus*aa_minus)/(aa_plus-aa_minus)*(uplus - uminus) -
          0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j,∇u,mesh)/mesh.Δx
        end
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVCUAlgorithm, ::Type{Val{false}})
    @unpack θ = alg
    λ = dt/mesh.Δx
    N = numcells(mesh)
    #update vector
    # 1. slopes
    ∇u = compute_slopes(u, mesh, θ, N, M, Val{false})

    for j in edge_indices(mesh)
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
          hh[j,:] = (aa_plus*Flux(uminus)-aa_minus*Flux(uplus))/(aa_plus-aa_minus) +
          (aa_plus*aa_minus)/(aa_plus-aa_minus)*(uplus - uminus) -
          0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j,∇u,mesh)/mesh.Δx
        end
    end
end