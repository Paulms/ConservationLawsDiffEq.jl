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


mutable struct FVCompWENOAlgorithm{oType, aType, cType} <: AbstractFVAlgorithm
  order :: oType
  splitting :: Symbol #Splitting strategy local Lax-Friedrichs (LLF) or global
  α :: aType
  crj :: cType
  αf :: Function
end

function FVCompWENOAlgorithm(;order=5, splitting = :GLF, αf = nothing)
  if αf == nothing
    αf = maxfluxρ
  end
  k = Int((order + 1)/2)-1
  crj = unif_crj(k+1)
  FVCompWENOAlgorithm(order, splitting,0.0, crj, αf)
end

function update_dt(alg::FVCompWENOAlgorithm{T1,T2,T3},u::AbstractArray{T4,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T1,T2,T3,T4}
  alg.α = alg.αf(u,Flux)
  assert(abs(alg.α) > eps(T2))
  CFL*mesh.Δx/alg.α
 end

mutable struct FVCompMWENOAlgorithm{oType, aType, cType} <: AbstractFVAlgorithm
  order :: oType
  splitting :: Symbol
  α :: aType
  crj :: cType
  αf :: Function
end

function FVCompMWENOAlgorithm(;order=5, splitting = :GLF, αf = nothing)
  if αf == nothing
    αf = maxfluxρ
  end
  k = Int((order + 1)/2)-1
  crj = unif_crj(k+1)
  FVCompMWENOAlgorithm(order, splitting, 0.0, crj, αf)
end

function update_dt(alg::FVCompMWENOAlgorithm{T1,T2,T3},u::AbstractArray{T4,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T1,T2,T3,T4}
  alg.α = alg.αf(u,Flux)
  assert(abs(alg.α) > eps(T2))
  CFL*mesh.Δx/alg.α
 end

immutable FVSpecMWENOAlgorithm{cType} <: AbstractFVAlgorithm
  order :: Int
  crj :: cType
end

"""
FVSpecMWENOAlgorithm(;order=5)
Initialize Spectral Mapped Weno algorithm with
WENO reconstruction of `order` default WENO5
"""
function FVSpecMWENOAlgorithm(;order=5)
  k = Int((order + 1)/2)-1
  crj = unif_crj(k+1)
  FVSpecMWENOAlgorithm(order, crj)
end

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

function glf_splitting(u, α, Flux, N)
  # Lax Friedrichs flux splitting
  fminus = zeros(u); fplus = zeros(u)
  for j = 1:N
    fminus[j,:] = 0.5*(Flux(u[j,:])-α*u[j,:])
    fplus[j,:] = 0.5*(Flux(u[j,:])+α*u[j,:])
  end
  fminus, fplus
end

function llf_splitting(u, mesh, Flux)
  # Lax Friedrichs flux splitting
  N = numcells(mesh)
  fminus = zeros(u); fplus = zeros(u)
  ul=cellval_at_left(1,u,mesh)
  αl = fluxρ(ul, Flux)
  for j = 1:N
    ur=cellval_at_right(j,u,mesh)
    αr = fluxρ(ur, Flux)
    αk = max(αl, αr)
    fminus[j,:] = 0.5*(Flux(u[j,:])-αk*u[j,:])
    fplus[j,:] = 0.5*(Flux(u[j,:])+αk*u[j,:])
    αl = αr
  end
  fminus,fplus
end

##############################################################
#Component Wise WENO algorithm
##############################################################
"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompWENOAlgorithm, ::Type{Val{true}})
Numerical flux of Component Wise WENO algorithm Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompWENOAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    @unpack order, splitting, crj = alg
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N)
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    Threads.@threads for j = 0:N
      Threads.@threads for i = 1:M
        fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
        fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
        hh[j+1,i] = sum(WENO_pm_rec(fm,fp,order; crj = crj))
      end
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompWENOAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    @unpack order, splitting, crj = alg
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N)
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    for j = 0:N
      for i = 1:M
        fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
        fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
        hh[j+1,i] = sum(WENO_pm_rec(fm,fp,order; crj = crj))
      end
    end
end

################################################################
#Component Wise Mapped WENO algorithm
##############################################################

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompMWENOAlgorithm, ::Type{Val{true}})
Numerical flux of Mapped Component Wise WENO algorithm Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompMWENOAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    @unpack order, splitting, crj = alg
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N)
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    Threads.@threads for j = 0:N
      Threads.@threads for i = 1:M
        fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
        fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
        hh[j+1,i] = sum(MWENO_pm_rec(fm,fp,order; crj = crj))
      end
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompMWENOAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    @unpack order, splitting, crj = alg
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N)
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    for j = 0:N
      for i = 1:M
        fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
        fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
        hh[j+1,i] = sum(MWENO_pm_rec(fm,fp,order; crj = crj))
      end
    end
end


###############################################################
#Characteristic Wise WENO algorithm (Spectral)
#################################################################
"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVSpecMWENOAlgorithm, ::Type{Val{true}})
Numerical flux of Mapped Spectral WENO algorithm Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVSpecMWENOAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    @unpack order, crj = alg
    k = Int((order + 1)/2)-1
    save_case = zeros(N+1,M)
    αj = zeros(N+1,M)
    RMats = Vector{typeof(Flux(Val{:jac},u[1,:]))}(0)
    LMats = Vector{typeof(Flux(Val{:jac},u[1,:]))}(0)
    gk = zeros(u)
    for j in edge_indices(mesh)
      @inbounds ul = cellval_at_left(j,u,mesh)
      @inbounds ur = cellval_at_right(j,u,mesh)
      MatJf = Flux(Val{:jac},0.5*(ul+ur))
      Rj = eigvecs(MatJf);  Lj = inv(Rj)
      push!(RMats,Rj); push!(LMats,Lj)
      λl = eigvals(Flux(Val{:jac},ul)); λr = eigvals(Flux(Val{:jac},ur))
      αj[j,:] = maximum(abs,[λl λr],2)
      if j < N+1
        gk[j,:] = Flux(u[j,:])
      end
      for i in 1:M
        if λl[i]*λr[i] <= 0
          save_case[j,i] = 1
        else
          if λl[i] > 0 && λr[i] > 0
            save_case[j,i] = 2
          else
            save_case[j,i] = 3
          end
        end
      end
    end
  #WEno Reconstrucion
  gklloc = zeros(k*2+1,M);gkrloc = zeros(k*2+1,M)
  gmloc = zeros(k*2+1,M);gploc = zeros(k*2+1,M)
  for j = 0:N
    for (ll,l) in enumerate((j-k):(j+k))
      gkl = get_cellvals(gk,mesh,(l+1,:)...)
      gkr = get_cellvals(gk,mesh,(l,:)...)
      ul = get_cellvals(u,mesh,(l+1,:)...)
      ur = get_cellvals(u,mesh,(l,:)...)
      gklloc[ll,:] = LMats[j+1]*gkl
      gkrloc[ll,:] =  LMats[j+1]*gkr
      gmloc[ll,:] = 0.5*LMats[j+1]*(gkl-αj[j+1,:].*ul)
      gploc[ll,:] = 0.5*LMats[j+1]*(gkr+αj[j+1,:].*ur)
    end
    for i = 1:M
      if save_case[j+1,i] == 1
        hh[j+1,i] = sum(MWENO_pm_rec(gmloc[:,i],gploc[:,i],order; crj = crj))
      elseif save_case[j+1,i] == 2
        hh[j+1,i] = MWENO_pm_rec(gklloc[:,i],gkrloc[:,i],order; crj = crj)[2]
      else
        hh[j+1,i] = MWENO_pm_rec(gklloc[:,i],gkrloc[:,i],order; crj = crj)[1]
      end
    end
    hh[j+1,:] = RMats[j+1]*hh[j+1,:]
  end
end
