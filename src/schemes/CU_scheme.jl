# Second-Order upwind central scheme
# Based on:
# Kurganov A., Noelle S., Petrova G., Semidiscrete Central-Upwind schemes
# for hyperbolic Conservation Laws and Hamilton-Jacobi Equations. SIAM. Sci Comput,
# Vol 23, No 3m pp 707-740. 2001

struct FVCUScheme{ltype <: AbstractSlopeLimiter} <: AbstractFVAlgorithm
  slopeLimiter :: ltype
end

function FVCUScheme(;slopeLimiter=GeneralizedMinmodLimiter())
  FVCUScheme(slopeLimiter)
end

function update_flux_value(uold,∇u,j,mesh,Flux, alg::FVCUScheme,nonscalar::Bool)
    uminus=cellval_at_left(j,uold,mesh)+0.5*cellval_at_left(j,∇u,mesh)
    uplus=cellval_at_right(j,uold,mesh)-0.5*cellval_at_right(j,∇u,mesh)

    if nonscalar
      λm = eigvals(Flux.Dflux(uminus))
      λp = eigvals(Flux.Dflux(uplus))
      aa_plus=maximum((maximum(λm), maximum(λp),0))
      aa_minus=minimum((minimum(λm), minimum(λp),0))
    else
      λm = Flux.Dflux(uminus)
      λp = Flux.Dflux(uplus)
      aa_plus=maximum((λm, λp,0))
      aa_minus=minimum((λm, λp,0))
    end
    # Update numerical fluxes
    if abs(aa_plus-aa_minus) < 1e-8
      return 0.5*(Flux(uminus)+Flux(uplus))
    else
      return (aa_plus*Flux(uminus)-aa_minus*Flux(uplus))/(aa_plus-aa_minus) +
      (aa_plus*aa_minus)/(aa_plus-aa_minus)*(uplus - uminus)
    end
end

# Dissipation Reduced Central upwind Scheme: Second-Order
# Based on:
# Kurganov A., Lin C., On the reduction of Numerical Dissipation in Central-Upwind
# Schemes, Commun. Comput. Phys. Vol 2. No. 1, pp 141-163, Feb 2007.

struct FVDRCUScheme{ltype <: AbstractSlopeLimiter} <: AbstractFVAlgorithm
  slopeLimiter :: ltype
end

function FVDRCUScheme(;slopeLimiter=GeneralizedMinmodLimiter())
  FVDRCUScheme(slopeLimiter)
end

function update_flux_value(uold,∇u,j,mesh,Flux, alg::FVDRCUScheme,nonscalar::Bool)
  uminus=cellval_at_left(j,uold,mesh)+0.5*cellval_at_left(j,∇u,mesh)
  uplus=cellval_at_right(j,uold,mesh)-0.5*cellval_at_right(j,∇u,mesh)

  if nonscalar
    λm = eigvals(Flux.Dflux(uminus))
    λp = eigvals(Flux.Dflux(uplus))
    aa_plus=maximum((λm[end], λp[end],0))
    aa_minus=minimum((λm[1], λp[1],0))
  else
    λm = Flux.Dflux(uminus)
    λp = Flux.Dflux(uplus)
    aa_plus=maximum((λm, λp,0))
    aa_minus=minimum((λm, λp,0))
  end

  # Update numerical fluxes
  if abs(aa_plus-aa_minus) < 1e-8
    return 0.5*(Flux(uminus)+Flux(uplus))
  else
    flm = Flux(uminus); flp = Flux(uplus)
    wint = 1/(aa_plus-aa_minus)*(aa_plus*uplus-aa_minus*uminus-(flp-flm))
    qj = minmod.((uplus-wint)/(aa_plus-aa_minus),(wint-uminus)/(aa_plus-aa_minus))
    return (aa_plus*flm-aa_minus*flp)/(aa_plus-aa_minus) + 
            (aa_plus*aa_minus)*((uplus - uminus)/(aa_plus-aa_minus) - qj)
  end
end


##################3
FVCUSchemes = Union{FVCUScheme,FVDRCUScheme}
"""
compute_fluxes!(hh, Flux, uold, mesh, dt, M, alg::FVCUScheme, ::Type{Val{true}})
Numerical flux of Second-Order upwind central scheme in 1D
"""
function compute_fluxes!(fluxes, Flux, uold, mesh, dt, alg::T, nonscalar::Bool, ::Type{Val{true}}) where {T <: FVCUSchemes}
  slopeLimiter = alg.slopeLimiter
  #update vector
  # 1. slopes
  ∇u = compute_slopes(uold, mesh, slopeLimiter, nonscalar, Val{true})

  Threads.@threads for j in node_indices(mesh)
      if nonscalar
          fluxes[:,j] .= update_flux_value(uold,∇u,j,mesh,Flux,alg,nonscalar)
      else
          fluxes[j] = update_flux_value(uold,∇u,j,mesh,Flux,alg,nonscalar)
      end
  end
end

function compute_fluxes!(fluxes, Flux, uold, mesh, dt, alg::T, nonscalar::Bool, ::Type{Val{false}}) where {T <: FVCUSchemes}
  slopeLimiter = alg.slopeLimiter
  #update vector
  # 1. slopes
  ∇u = compute_slopes(uold, mesh, slopeLimiter, nonscalar, Val{false})

  for j in node_indices(mesh)
      if nonscalar
          fluxes[:,j] .= update_flux_value(uold,∇u,j,mesh,Flux,alg,nonscalar)
      else
          fluxes[j] = update_flux_value(uold,∇u,j,mesh,Flux,alg,nonscalar)
      end
  end
end

# Dissipation Reduced Central upwind Scheme: Fifth-Order
# Based on:
# Kurganov A., Lin C., On the reduction of Numerical Dissipation in Central-Upwind
# Schemes, Commun. Comput. Phys. Vol 2. No. 1, pp 141-163, Feb 2007.
# Kurganov, Liu, New adaptive artificial viscosity method for hyperbolic systems
# of conservation laws

struct FVDRCU5Scheme{ltype <: AbstractSlopeLimiter} <: AbstractFVAlgorithm
  slopeLimiter :: ltype
end

function FVDRCU5Scheme(;slopeLimiter=GeneralizedMinmodLimiter())
  FVDRCU5Scheme(slopeLimiter)
end

function compute_fluxes!(fluxes, Flux, uold, mesh, dt, alg::FVDRCU5Scheme, nonscalar::Bool, ::Type{Val{true}})
  #update vector
  Threads.@threads for j in node_indices(mesh)
      if nonscalar
          fluxes[:,j] .= update_flux_value(uold,j,mesh,Flux,alg,nonscalar)
      else
          fluxes[j] = update_flux_value(uold,j,mesh,Flux,alg,nonscalar)
      end
  end
end

function compute_fluxes!(fluxes, Flux, uold, mesh, dt, alg::FVDRCU5Scheme, nonscalar::Bool, ::Type{Val{false}})
  #update vector
  for j in node_indices(mesh)
      if nonscalar
          fluxes[:,j] .= update_flux_value(uold,j,mesh,Flux,alg,nonscalar)
      else
          fluxes[j] = update_flux_value(uold,j,mesh,Flux,alg,nonscalar)
      end
  end
end

function update_flux_value(uold,j,mesh,Flux,alg::FVDRCU5Scheme, nonscalar)
    slopeLimiter = alg.slopeLimiter
    # A fifth-order piecewise polynomial reconstruction
    @inbounds ulll=cellval_at_left(j-2,uold,mesh)
    @inbounds ull=cellval_at_left(j-1,uold,mesh)
    @inbounds ul=cellval_at_left(j,uold,mesh)
    @inbounds ur=cellval_at_right(j,uold,mesh)
    @inbounds urr=cellval_at_right(j+1,uold,mesh)
    @inbounds urrr=cellval_at_right(j+2,uold,mesh)

    uminus = 1/60*(2*ulll-13*ull+47*ul+27*ur-3*urr)
    uplus = 1/60*(-3*ull+27*ul+47*ur-13*urr+2*urrr)

    #Remark: some extrange bug wth eigvals force me to use Lapack
    if nonscalar
      λm = sort(LAPACK.geev!('N','N',Array(Flux.Dflux(uminus)))[1])
      λp = sort(LAPACK.geev!('N','N',Array(Flux.Dflux(uplus)))[1])
      aa_plus=maximum((λm[end], λp[end],0))
      aa_minus=minimum((λm[1], λp[1],0))
    else
      λm = Flux.Dflux(uminus)
      λp = Flux.Dflux(uplus)
      aa_plus=maximum((λm, λp,0))
      aa_minus=minimum((λm, λp,0))
    end
    # Update numerical fluxes
    if abs(aa_plus-aa_minus) < 1e-8
      return 0.5*(Flux(uminus)+Flux(uplus))
    else
      flm = Flux(uminus); flp = Flux(uplus)
      wint = 1/(aa_plus-aa_minus)*(aa_plus*uplus-aa_minus*uminus-
      (flp-flm))
      qj = minmod.((uplus-wint)/(aa_plus-aa_minus),(wint-uminus)/(aa_plus-aa_minus))
      return (aa_plus*flm-aa_minus*flp)/(aa_plus-aa_minus) +
      (aa_plus*aa_minus)*((uplus - uminus)/(aa_plus-aa_minus) - qj)
    end
end