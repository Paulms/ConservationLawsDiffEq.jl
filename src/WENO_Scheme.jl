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
  rec_scheme :: cType
  αf :: Function
end

function FVCompWENOAlgorithm(;order=5, splitting = :GLF, αf = nothing)
  if αf == nothing
    αf = maxfluxρ
  end
  FVCompWENOAlgorithm(order, splitting,0.0, WENO_Reconstruction(order), αf)
end

function update_dt(alg::FVCompWENOAlgorithm{T1,T2,T3},u::AbstractArray{T4,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T1,T2,T3,T4}
  alg.α = alg.αf(u,Flux)
  @assert (abs(alg.α) > eps(T2))
  CFL*mesh.Δx/alg.α
 end

mutable struct FVCompMWENOAlgorithm{oType, aType, cType} <: AbstractFVAlgorithm
  order :: oType
  splitting :: Symbol
  α :: aType
  rec_scheme :: cType
  αf :: Function
end

function FVCompMWENOAlgorithm(;order=5, splitting = :GLF, αf = nothing)
  if αf == nothing
    αf = maxfluxρ
  end
  FVCompMWENOAlgorithm(order, splitting, 0.0, MWENO_Reconstruction(order), αf)
end

function update_dt(alg::FVCompMWENOAlgorithm{T1,T2,T3},u::AbstractArray{T4,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T1,T2,T3,T4}
  alg.α = alg.αf(u,Flux)
  @assert (abs(alg.α) > eps(T2))
  CFL*mesh.Δx/alg.α
 end

struct FVSpecMWENOAlgorithm{cType} <: AbstractFVAlgorithm
  order :: Int
  rec_scheme :: cType
end

"""
FVSpecMWENOAlgorithm(;order=5)
Initialize Spectral Mapped Weno algorithm with
WENO reconstruction of `order` default WENO5
"""
function FVSpecMWENOAlgorithm(;order=5)
  FVSpecMWENOAlgorithm(order, MWENO_Reconstruction(order))
end

##############################################################
#Component Wise WENO algorithm
##############################################################
function inner_loop!(hh,j,k,M,fminus,fplus,mesh,alg::FVCompWENOAlgorithm)
    @inbounds for i = 1:M
      fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
      fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
      hh[j+1,i] = sum(reconstruct(fm, fp, alg.rec_scheme))
    end
end
"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompWENOAlgorithm, ::Type{Val{true}})
Numerical flux of Component Wise WENO algorithm Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompWENOAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    order = alg.order; splitting = alg.splitting
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N, Val{true})
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    Threads.@threads for j = 0:N
        inner_loop!(hh,j,k,M,fminus,fplus,mesh,alg)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompWENOAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    order = alg.order; splitting = alg.splitting
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N,Val{false})
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    for j = 0:N
        inner_loop!(hh,j,k,M,fminus,fplus,mesh,alg)
    end
end

################################################################
#Component Wise Mapped WENO algorithm
##############################################################

function inner_loop!(hh,j,k,M,fminus,fplus,mesh,alg::FVCompMWENOAlgorithm)
    @inbounds for i = 1:M
      fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
      fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
      hh[j+1,i] = sum(reconstruct(fm, fp, alg.rec_scheme))
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompMWENOAlgorithm, ::Type{Val{true}})
Numerical flux of Mapped Component Wise WENO algorithm Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompMWENOAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    order = alg.order; splitting = alg.splitting
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N,Val{true})
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    Threads.@threads for j = 0:N
        inner_loop!(hh,j,k,M,fminus,fplus,mesh,alg)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVCompMWENOAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    order = alg.order; splitting = alg.splitting
    k = Int((order + 1)/2)-1
    # Flux splitting
    if splitting == :GLF
      fminus, fplus = glf_splitting(u, alg.α, Flux, N,Val{false})
    elseif spliting == :LLF
      fminus, fplus = llf_splitting(u, mesh, Flux)
    else
      throw("Splitting strategy not supported...")
    end
    #Compute numerical fluxes
    for j = 0:N
        inner_loop!(hh,j,k,M,fminus,fplus,mesh,alg)
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
    order = alg.order; rec_scheme = alg.rec_scheme
    k = Int((order + 1)/2)-1
    save_case = fill(0.0,N+1,M)
    αj = fill(0.0,N+1,M)
    RMats = Vector{typeof(Flux(Val{:jac},u[1,:]))}(undef,0)
    LMats = Vector{typeof(Flux(Val{:jac},u[1,:]))}(undef,0)
    gk = fill!(similar(u), zero(eltype(u)))
    for j in edge_indices(mesh)
      @inbounds ul = cellval_at_left(j,u,mesh)
      @inbounds ur = cellval_at_right(j,u,mesh)
      MatJf = Flux(Val{:jac},0.5*(ul+ur))
      Rj = eigvecs(MatJf);  Lj = inv(Rj)
      push!(RMats,Rj); push!(LMats,Lj)
      λl = eigvals(Flux(Val{:jac},ul)); λr = eigvals(Flux(Val{:jac},ur))
      αj[j,:] = maximum(abs,[λl λr],dims=2)
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
  gklloc = fill(0.0,k*2+1,M);gkrloc = fill(0.0,k*2+1,M)
  gmloc = fill(0.0,k*2+1,M);gploc = fill(0.0,k*2+1,M)
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
        hh[j+1,i] = sum(reconstruct(gmloc[:,i],gploc[:,i], rec_scheme))
      elseif save_case[j+1,i] == 2
        hh[j+1,i] = reconstruct(gklloc[:,i],gkrloc[:,i],rec_scheme)[2]
      else
        hh[j+1,i] = reconstruct(gklloc[:,i],gkrloc[:,i],rec_scheme)[1]
      end
    end
    hh[j+1,:] = RMats[j+1]*hh[j+1,:]
  end
end
