# Classic Lax-Friedrichs Method
# Reference:
# R. Leveque. Finite Volume Methods for Hyperbolic Problems.Cambridge University
# Press. New York 2002

immutable LaxFriedrichsAlgorithm <: AbstractFVAlgorithm end

immutable LocalLaxFriedrichsAlgorithm <: AbstractFVAlgorithm end

mutable struct GlobalLaxFriedrichsAlgorithm{T} <: AbstractFVAlgorithm
  αf :: Function #viscosity coefficient
  α :: T
end

function update_dt(alg::GlobalLaxFriedrichsAlgorithm{T1},u::AbstractArray{T2,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T1,T2}
  alg.α = alg.αf(u,Flux)
  assert(abs(alg.α) > eps(T1))
  CFL*mesh.Δx/alg.α
end

function GlobalLaxFriedrichsAlgorithm(;αf = nothing)
    if αf == nothing
        αf = maxfluxρ
    end
    GlobalLaxFriedrichsAlgorithm(αf,0.0)
end

mutable struct COMP_GLF_Diff_Algorithm{T, cType, oType} <: AbstractFVAlgorithm
  αf :: Function #viscosity coefficient
  α :: T
  crj :: cType
  order :: oType
end

function update_dt(alg::COMP_GLF_Diff_Algorithm{T0,T1,T2},u::AbstractArray{T3,2},Flux,
    DiffMat,CFL,mesh::Uniform1DFVMesh) where {T0,T1,T2,T3}
  alg.α = alg.αf(u,Flux)
  assert(abs(alg.α) > eps(T0))
  CFL*mesh.Δx/alg.α
end

function COMP_GLF_Diff_Algorithm(;αf = nothing)
    if αf == nothing
        αf = maxfluxρ
    end
    crj = unif_crj(3)
    COMP_GLF_Diff_Algorithm(αf,0.0,crj,5)
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxFriedrichsAlgorithm, ::Type{Val{true}})
Numerical flux of lax friedrichs algorithm in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxFriedrichsAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LaxFriedrichsAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-dx/(2*dt)*(ur-ul)
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LocalLaxFriedrichsAlgorithm, ::Type{Val{true}})
Numerical flux of local lax friedrichs algorithm in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::LocalLaxFriedrichsAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    dx = mesh.Δx
    ul=cellval_at_left(1,u,mesh)
    αl = fluxρ(ul, Flux)
    #update vector
    for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        αr = fluxρ(ur, Flux)
        αk = max(αl, αr)
        # Numerical Fluxes
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-αk*(ur-ul)
        αl = αr
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::GlobalLaxFriedrichsAlgorithm, ::Type{Val{true}})
Numerical flux of Global Lax-Friedrichs Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::GlobalLaxFriedrichsAlgorithm, ::Type{Val{true}})
    N = numcells(mesh)
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # update numerical flux
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-alg.α*(ur-ul)
    end
end

function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::GlobalLaxFriedrichsAlgorithm, ::Type{Val{false}})
    N = numcells(mesh)
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        # Local speeds of propagation
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        # update numerical flux
        @inbounds hh[j,:] = 0.5*(Flux(ul)+Flux(ur))-alg.α*(ur-ul)
    end
end

# Component Wise Global Lax-Friedrichs Scheme
# Based on:
# Raimund Bürger , Rosa Donat , Pep Mulet , Carlos A. Vega,
# On the implementation of WENO schemes for a class of polydisperse sedimentation
# models, Journal of Computational Physics, v.230 n.6, p.2322-2344,
# March, 2011  [doi>10.1016/j.jcp.2010.12.019]
"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::COMP_GLF_Diff_Algorithm, ::Type{Val{false}})
Numerical flux of Component Wise Global Lax-Friedrichs Diffusive Scheme in 1D
"""
function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::COMP_GLF_Diff_Algorithm, ::Type{Val{false}})
    N = numcells(mesh)
    @unpack order, crj = alg
    k = Int((order + 1)/2)-1
    # Flux splitting
    fminus, fplus = glf_splitting(u, alg.α, Flux, N)
    # 1. slopes
    ∇u = compute_slopes(u, mesh, 1.0, N, M, Val{false})
    #Compute numerical fluxes
    for j = 0:N
      for i = 1:M
        fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
        fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
        @inbounds hh[j+1,i] = sum(WENO_pm_rec(fm,fp,order; crj = crj))
      end
      @inbounds ul=cellval_at_left(j+1,u,mesh)
      @inbounds ur=cellval_at_right(j+1,u,mesh)
      @inbounds hh[j+1,:] = hh[j+1,:] - 0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j+1,∇u,mesh)/mesh.Δx
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::COMP_GLF_Diff_Algorithm, ::Type{Val{true}})
Numerical flux of Component Wise Global Lax-Friedrichs Diffusive Scheme in 1D. Parallel
"""
function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::COMP_GLF_Diff_Algorithm, ::Type{Val{true}})
    N = numcells(mesh)
    @unpack order, crj = alg
    k = Int((order + 1)/2)-1
    # Flux splitting
    fminus, fplus = glf_splitting(u, alg.α, Flux, N)
    # 1. slopes
    ∇u = compute_slopes(u, mesh, 1.0, N, M, Val{true})
    #Compute numerical fluxes
    Threads.@threads for j = 0:N
      for i = 1:M
        fm = get_cellvals(fminus,mesh,(j-k+1:j+k+1,i)...)
        fp = get_cellvals(fplus,mesh,(j-k:j+k,i)...)
        @inbounds hh[j+1,i] = sum(WENO_pm_rec(fm,fp,order; crj = crj))
      end
      @inbounds ul=cellval_at_left(j+1,u,mesh)
      @inbounds ur=cellval_at_right(j+1,u,mesh)
      @inbounds hh[j+1,:] = hh[j+1,:] - 0.5*(DiffMat(ur)+DiffMat(ul))*cellval_at_right(j+1,∇u,mesh)/mesh.Δx
    end
end
