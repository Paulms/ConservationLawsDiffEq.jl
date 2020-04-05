# Component Wise Weno and Mapped Weno schemes
# Flux splitting using local (LLF) or global Lax-Flux splitting (GLF).
#
# Spectral Mapped Weno
#
# Based on:
# C.-W. Shu, “High order ENO and WENO schemes for computational fluid
# dynamics,” in High-Order Methods for Computational Physics, Lecture Notes
#in Computational Science and Engineering vol. 9, 439–582. New York:
# Springer Verlag, 1999

# A. Henrick, T. Aslam, J. Powers, Mapped weighted essentially non-oscillatory
# schemes: Achiving optimal order near critical points

# R. Bürger, R. Donat, P. Mulet, C. Vega, On the implementation of WENO schemes
# for a class of polydisperse sedimentation models. October 21, 2010.


mutable struct FVCompWENOScheme{oType, aType, cType} <: AbstractFVAlgorithm
  order :: oType
  splitting :: Symbol #Splitting strategy local Lax-Friedrichs (LLF) or global
  α :: aType
  rec_scheme :: cType
  αf :: Function
end

function FVCompWENOScheme(;order=5, splitting = :GLF, αf = nothing)
  if αf == nothing
    αf = maxfluxρ
  end
  FVCompWENOScheme(order, splitting,0.0, WENO_Reconstruction(order), αf)
end

mutable struct FVCompMWENOScheme{oType, aType, cType} <: AbstractFVAlgorithm
  order :: oType
  splitting :: Symbol
  α :: aType
  rec_scheme :: cType
  αf :: Function
end

function FVCompMWENOScheme(;order=5, splitting = :GLF, αf = nothing)
  if αf == nothing
    αf = maxfluxρ
  end
  FVCompMWENOScheme(order, splitting, 0.0, MWENO_Reconstruction(order), αf)
end

struct FVSpecMWENOScheme{cType} <: AbstractFVAlgorithm
  order :: Int
  rec_scheme :: cType
end

FVCompWENOSchemes = Union{FVCompWENOScheme,FVCompMWENOScheme}

function update_dt(alg::FVCompWENOScheme{T1,T2,T3},u,Flux,
    CFL,mesh) where {T1,T2,T3}
  alg.α = alg.αf(u,Flux,mesh)
  @assert (abs(alg.α) > eps(T2))
  dx = cell_volume(mesh, 1)
  CFL*dx/alg.α
 end

 function update_dt(alg::FVCompMWENOScheme{T1,T2,T3},u,Flux,
  CFL,mesh) where {T1,T2,T3}
  alg.α = alg.αf(u,Flux,mesh)
  @assert (abs(alg.α) > eps(T2))
  dx = cell_volume(mesh, 1)
  CFL*dx/alg.α
end

"""
FVSpecMWENOScheme(;order=5)
Initialize Spectral Mapped Weno algorithm with
WENO reconstruction of `order` default WENO5
"""
function FVSpecMWENOScheme(;order=5)
  FVSpecMWENOScheme(order, MWENO_Reconstruction(order))
end
##############################################################
#Component Wise WENO Scheme and Mapped Component Wise WENO Scheme
##############################################################
function update_flux_value(j,k,fminus,fplus,mesh,alg::FVCompWENOSchemes,i)
      fm = get_cellvals(fminus,mesh,(i,j-k+1:j+k+1)...)
      fp = get_cellvals(fplus,mesh,(i,j-k:j+k)...)
      return sum(reconstruct(fm, fp, alg.rec_scheme))
end
function update_flux_value(j,k,fminus,fplus,mesh,alg::FVCompWENOSchemes)
  fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,)...)
  fp = get_cellvals(fplus,mesh,(j-k:j+k,)...)
  return sum(reconstruct(fm, fp, alg.rec_scheme))
end
"""
compute_fluxes!(fluxes, Flux, u, mesh, dt, M, alg::FVCompWENOScheme, ::Type{Val{true}})
Numerical flux of Component Wise WENO algorithm Scheme in 1D
"""
function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::FVCompWENOSchemes, nonscalar::Bool, ::Type{Val{true}})
    order = alg.order; splitting = alg.splitting
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, nonscalar, Val{true})
    elseif splitting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux, nonscalar)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    Threads.@threads for j in node_indices(mesh)
      if nonscalar
        for i in 1:size(fluxes,1)
          fluxes[i,j] = update_flux_value(j-1,k,fminus,fplus,mesh,alg,i)
        end
      else
          fluxes[j] = update_flux_value(j-1,k,fminus,fplus,mesh,alg)
      end
    end
end

function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::FVCompWENOSchemes, nonscalar::Bool, ::Type{Val{false}})
  order = alg.order; splitting = alg.splitting
  k = Int((order + 1)/2)-1
  # Flux splitting
  if splitting == :GLF
    fminus, fplus = glf_splitting(u, alg.α, Flux, nonscalar, Val{false})
  elseif splitting == :LLF
    fminus, fplus = llf_splitting(u, mesh, Flux, nonscalar)
  else
    throw("Splitting strategy not supported...")
  end
  #Compute numerical fluxes
  for j in node_indices(mesh)
    if nonscalar
      for i in 1:size(fluxes,1)
        fluxes[i,j] = update_flux_value(j-1,k,fminus,fplus,mesh,alg,i)
      end
    else
        fluxes[j] = update_flux_value(j-1,k,fminus,fplus,mesh,alg)
    end
  end
end


###############################################################
#Characteristic Wise WENO algorithm (Spectral)
#################################################################
"""
compute_fluxes!(fluxes, Flux, u, mesh, dt, M, alg::FVSpecMWENOScheme, ::Type{Val{true}})
Numerical flux of Mapped Spectral WENO algorithm Scheme in 1D
"""
function compute_fluxes!(fluxes, Flux, u, mesh, dt, alg::FVSpecMWENOScheme, nonscalar::Bool, ::Type{Val{false}})
    N = getncells(mesh.mesh)
    order = alg.order; rec_scheme = alg.rec_scheme
    k = Int((order + 1)/2)-1
    if nonscalar
      M = size(u,1)
      save_case = fill(0.0,M,N+1)
      αj = fill(0.0,M,N+1)
      RMats = Vector{typeof(eigvecs(Flux.Dflux(u[:,1])))}(undef,0)
      LMats = Vector{eltype(RMats)}(undef,0)
      gk = fill!(similar(u), zero(eltype(u)))
      for j in node_indices(mesh)
        @inbounds ul = cellval_at_left(j,u,mesh)
        @inbounds ur = cellval_at_right(j,u,mesh)
        MatJf = Flux.Dflux(0.5*(ul+ur))
        Rj = eigvecs(MatJf);  Lj = inv(Rj)
        push!(RMats,Rj); push!(LMats,Lj)
        λl = eigvals(Flux.Dflux(ul)); λr = eigvals(Flux.Dflux(ur))
        αj[:,j] = maximum(abs,[λl λr],dims=2)
        if j < N+1
          gk[:,j] = Flux(u[:,j])
        end
        for i in 1:M
          if λl[i]*λr[i] <= 0
            save_case[i,j] = 1
          else
            if λl[i] > 0 && λr[i] > 0
              save_case[i,j] = 2
            else
              save_case[i,j] = 3
            end
          end
        end
      end
    #WEno Reconstrucion
    gklloc = fill(0.0,M,k*2+1);gkrloc = fill(0.0,M,k*2+1)
    gmloc = fill(0.0,M,k*2+1);gploc = fill(0.0,M,k*2+1)
    for j = 0:N
      for (ll,l) in enumerate((j-k):(j+k))
        gkl = get_cellvals(gk,mesh,(:,l+1)...)
        gkr = get_cellvals(gk,mesh,(:,l)...)
        ul = get_cellvals(u,mesh,(:,l+1)...)
        ur = get_cellvals(u,mesh,(:,l)...)
        gklloc[:,ll] = LMats[j+1]*gkl
        gkrloc[:,ll] =  LMats[j+1]*gkr
        gmloc[:,ll] = 0.5*LMats[j+1]*(gkl-αj[:,j+1].*ul)
        gploc[:,ll] = 0.5*LMats[j+1]*(gkr+αj[:,j+1].*ur)
      end
      for i = 1:M
        if save_case[i,j+1] == 1
          fluxes[i,j+1] = sum(reconstruct(gmloc[i,:],gploc[i,:], rec_scheme))
        elseif save_case[i,j+1] == 2
          fluxes[i,j+1] = reconstruct(gklloc[i,:],gkrloc[i,:],rec_scheme)[2]
        else
          fluxes[i,j+1] = reconstruct(gklloc[i,:],gkrloc[i,:],rec_scheme)[1]
        end
      end
      fluxes[:,j+1] = RMats[j+1]*fluxes[:,j+1]
    end
  else
    save_case = fill(0.0,N+1)
    αj = fill(0.0,N+1)
    gk = fill!(similar(u), zero(eltype(u)))
    for j in node_indices(mesh)
      @inbounds ul = cellval_at_left(j,u,mesh)
      @inbounds ur = cellval_at_right(j,u,mesh)
      λl = Flux.Dflux(ul); λr = Flux.Dflux(ur)
      αj[j] = maximum(abs,[λl,λr])
      if j < N+1
        gk[j] = Flux(u[j])
      end
      if λl*λr <= 0
        save_case[j] = 1
      else
        if λl > 0 && λr > 0
          save_case[j] = 2
        else
          save_case[j] = 3
        end
      end
    end
    #WEno Reconstrucion
    gklloc = fill(0.0,k*2+1);gkrloc = fill(0.0,k*2+1)
    gmloc = fill(0.0,k*2+1);gploc = fill(0.0,k*2+1)
    for j in 0:N
      for (ll,l) in enumerate((j-k):(j+k))
        gkl = get_cellvals(gk,mesh,(l+1,)...)
        gkr = get_cellvals(gk,mesh,(l,)...)
        ul = get_cellvals(u,mesh,(l+1,)...)
        ur = get_cellvals(u,mesh,(l,)...)
        gklloc[ll] = gkl
        gkrloc[ll] = gkr
        gmloc[ll] = 0.5*(gkl-αj[j+1]*ul)
        gploc[ll] = 0.5*(gkr+αj[j+1]*ur)
      end
      if save_case[j+1] == 1
        fluxes[j+1] = sum(reconstruct(gmloc[:],gploc[:], rec_scheme))
      elseif save_case[j+1] == 2
        fluxes[j+1] = reconstruct(gklloc[:],gkrloc[:],rec_scheme)[2]
      else
        fluxes[j+1] = reconstruct(gklloc[:],gkrloc[:],rec_scheme)[1]
      end

    end
  end
end
