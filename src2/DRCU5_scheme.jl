# Dissipation Reduced Central upwind Scheme: Fifth-Order
# Based on:
# Kurganov A., Lin C., On the reduction of Numerical Dissipation in Central-Upwind
# Schemes, Commun. Comput. Phys. Vol 2. No. 1, pp 141-163, Feb 2007.
# Kurganov, Liu, New adaptive artificial viscosity method for hyperbolic systems
# of conservation laws

struct FVDRCU5Algorithm{ltype <: AbstractSlopeLimiter} <: AbstractFVAlgorithm
  slopeLimiter :: ltype
end

function FVDRCU5Algorithm(;slopeLimiter=GeneralizedMinmodLimiter())
  FVDRCU5Algorithm(slopeLimiter)
end

function inner_loop!(hh,j,u,mesh,slopeLimiter,Flux, alg::FVDRCU5Algorithm)
    # A fifth-order piecewise polynomial reconstruction
    @inbounds ulll=cellval_at_left(j-2,u,mesh)
    @inbounds ull=cellval_at_left(j-1,u,mesh)
    @inbounds ul=cellval_at_left(j,u,mesh)
    @inbounds ur=cellval_at_right(j,u,mesh)
    @inbounds urr=cellval_at_right(j+1,u,mesh)
    @inbounds urrr=cellval_at_right(j+2,u,mesh)

    uminus = 1/60*(2*ulll-13*ull+47*ul+27*ur-3*urr)
    uplus = 1/60*(-3*ull+27*ul+47*ur-13*urr+2*urrr)

    #Remark: some extrange bug wth eigvals force me to use Lapack
    λm = sort(LAPACK.geev!('N','N',Array(Flux(Val{:jac}, uminus)))[1])
    λp = sort(LAPACK.geev!('N','N',Array(Flux(Val{:jac}, uplus)))[1])
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
Numerical flux of Fifth-Order dissipation reduced upwind central scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVDRCU5Algorithm, ::Type{Val{true}})
    slopeLimiter = alg.slopeLimiter
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh,j,u,mesh,slopeLimiter,Flux, alg)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVDRCU5Algorithm, ::Type{Val{false}})
    slopeLimiter = alg.slopeLimiter
    #update vector
    for j in edge_indices(mesh)
        inner_loop!(hh,j,u,mesh,slopeLimiter,Flux, alg)
    end
end

function inner_loop!(hh,j,u,∇u,mesh,slopeLimiter,Flux, DiffMat, alg::FVDRCU5Algorithm)
    # A fifth-order piecewise polynomial reconstruction
    @inbounds ulll=cellval_at_left(j-2,u,mesh)
    @inbounds ull=cellval_at_left(j-1,u,mesh)
    @inbounds ul=cellval_at_left(j,u,mesh)
    @inbounds ur=cellval_at_right(j,u,mesh)
    @inbounds urr=cellval_at_right(j+1,u,mesh)
    @inbounds urrr=cellval_at_right(j+2,u,mesh)

    uminus = 1/60*(2*ulll-13*ull+47*ul+27*ur-3*urr)
    uplus = 1/60*(-3*ull+27*ul+47*ur-13*urr+2*urrr)

    #Remark: some extrange bug wth eigvals force me to use Lapack
    λm = sort(LAPACK.geev!('N','N',Flux(Val{:jac}, uminus))[1])
    λp = sort(LAPACK.geev!('N','N',Flux(Val{:jac}, uplus))[1])
    aa_plus=maximum((λm[end], λp[end],0))
    aa_minus=minimum((λm[1], λp[1],0))
    # Update numerical fluxes
    if abs(aa_plus-aa_minus) < 1e-8
      hh[j,:] = 0.5*(Flux(uminus)+Flux(uplus)) -
      0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j,∇u,mesh)/mesh.Δx
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

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVDRCU5Algorithm, ::Type{Val{true}})
    slopeLimiter = alg.slopeLimiter
    # 1. slopes
    ∇u = compute_slopes(u, mesh, slopeLimiter, M, Val{true})
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh,j,u,∇u,mesh,slopeLimiter,Flux, DiffMat, alg)
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVDRCU5Algorithm, ::Type{Val{false}})
    slopeLimiter = alg.slopeLimiter
    # 1. slopes
    ∇u = compute_slopes(u, mesh, slopeLimiter, M, Val{true})
    #update vector
    for j in edge_indices(mesh)
        inner_loop!(hh,j,u,∇u,mesh,slopeLimiter,Flux, DiffMat, alg)
    end
end
