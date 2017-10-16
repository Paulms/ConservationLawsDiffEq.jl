mutable struct FVIntegrator{T1,mType,F,cType,T2,tType}
  alg::T1
  mesh::mType
  Flux::F
  CFL :: cType
  M::Int
  fluxes :: T2
  dt :: tType
  use_threads :: Bool
end

mutable struct FVDiffIntegrator{T1,mType,F,B, cType,T2, tType}
  alg::T1
  mesh::mType
  Flux::F
  DiffMat::B
  CFL :: cType
  M::Int
  fluxes::T2
  dt :: tType
  use_threads :: Bool
end

function update_dt(u::AbstractArray{T,2},fv::FVIntegrator) where {T}
  @unpack mesh, alg, Flux, CFL = fv
  update_dt(alg, u, Flux, CFL, mesh)
end

"""
    (fv::FVIntegrator)(t, u, du)

Apply a finite volume semidiscretisation.
"""
function (fv::FVIntegrator)(t, u, du)
  @boundscheck begin
    if length(u) != length(du)
      error("length(u) = $(length(u)) != $(length(du)) = length(du)")
    end
    length(u) != numcells(fv.mesh) && error("length(u) != numcells(fv.mesh)")
  end

  @unpack mesh, alg, Flux, M, fluxes, dt, use_threads = fv
  compute_fluxes!(fluxes, Flux, u, mesh, dt, M, alg, Val{use_threads})
  if isleftzeroflux(mesh);fluxes[1,:] = 0.0; end
  if isrightzeroflux(mesh);fluxes[numedges(mesh),:] = 0.0;end
  compute_du!(du, fluxes, mesh, Val{false}, Val{use_threads})

  nothing
end

function compute_du!(du, fluxes, mesh::AbstractFVMesh1D, ::Type{Val{false}}, ::Type{Val{true}})
    Threads.@threads for cell in cell_indices(mesh)
        @inbounds left = left_edge(cell, mesh)
        @inbounds right = right_edge(cell, mesh)
        @inbounds du[cell] = -( fluxes[right] - fluxes[left] ) / volume(cell, mesh)
    end
end

function compute_du!(du, fluxes, mesh::AbstractFVMesh1D, ::Type{Val{false}}, ::Type{Val{false}})
    for cell in cell_indices(mesh)
        @inbounds left = left_edge(cell, mesh)
        @inbounds right = right_edge(cell, mesh)
        @inbounds du[cell] = -( fluxes[right] - fluxes[left] ) / volume(cell, mesh)
    end
end
