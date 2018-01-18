# Based on
# Jerez, Pares, Entropy stable schemes for degenerate convection-difusion
# equations. 2017. Society for Industrial and Applied Mathematics. SIAM. Vol. 55.
# No. 1. pp. 240-264

immutable FVESJPAlgorithm <: AbstractFVAlgorithm
  Nflux :: Function
  Ndiff :: Function #Entropy stable 2 point flux
  ϵ     :: Number # Extra diffusion
end
immutable FVESJPeAlgorithm <: AbstractFVAlgorithm
  Nflux :: Function
  Ndiff :: Function #Entropy stable 2 point flux
  ϵ     :: Number
  ve    :: Function #Entropy variable
end
function FVESJPAlgorithm(Nflux, Ndiff;ϵ=0.0,ve=nothing)
  if ve != nothing
    FVESJPeAlgorithm(Nflux, Ndiff, ϵ, ve)
  else
    FVESJPAlgorithm(Nflux, Ndiff, ϵ)
  end
end

function inner_loop!(hh, j, u, mesh, ϵ, dx, Nflux, Ndiff, alg::FVESJPAlgorithm)
    @inbounds ul = cellval_at_left(j,u,mesh)
    @inbounds ur = cellval_at_right(j,u,mesh)
    hh[j,:] = Nflux(ul, ur) -
    1/dx*(Ndiff(ul, ur)*(ur-ul)+ ϵ*(ur-ul))
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVESJPAlgorithm, ::Type{Val{true}})
Numerical flux of Entropy Stable Schemes in 1D
"""
function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVESJPAlgorithm, ::Type{Val{true}})
    @unpack Nflux,Ndiff,ϵ = alg
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh, j, u, mesh, ϵ, dx, Nflux, Ndiff, alg)
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVESJPAlgorithm, ::Type{Val{false}})
    @unpack Nflux,Ndiff,ϵ = alg
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    for j in edge_indices(mesh)
        inner_loop!(hh, j, u, mesh, ϵ, dx, Nflux, Ndiff, alg)
    end
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVESJPAlgorithm, ::Type{Val{true}})
Numerical flux of Entropy Stable Schemes in entropy variables 1D
"""
function inner_loop!(hh, j, u, mesh, ϵ, dx, Nflux, Ndiff,alg::FVESJPeAlgorithm)
    @inbounds vl = ve(cellval_at_left(j,u,mesh))
    @inbounds vr = ve(cellval_at_right(j,u,mesh))
    hh[j,:] = Nflux(vl, vr) -
    1/dx*(Ndiff(vl, vr)*(vr-vl)+ ϵ*(vr-vl))
end
function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVESJPeAlgorithm, ::Type{Val{true}})
    @unpack Nflux,Ndiff,ϵ,ve = alg
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    Threads.@threads for j in edge_indices(mesh)
        inner_loop!(hh, j, u, mesh, ϵ, dx, Nflux, Ndiff, alg)
    end
end

function compute_Dfluxes!(hh, Flux, DiffMat, u, mesh, dt, M, alg::FVESJPeAlgorithm, ::Type{Val{false}})
    @unpack Nflux,Ndiff,ϵ,ve = alg
    N = numcells(mesh)
    dx = mesh.Δx
    #update vector
    for j in edge_indices(mesh)
        inner_loop!(hh, j, u, mesh, ϵ, dx, Nflux, Ndiff, alg)
    end
end
