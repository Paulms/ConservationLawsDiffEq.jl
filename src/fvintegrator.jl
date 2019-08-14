mutable struct FVIntegrator{T1,mType,F,T2,tType}
  alg::T1
  mesh::mType
  Flux::F
  numvars::Int
  fluxes :: T2
  dt :: tType
  use_threads :: Bool
end

mutable struct FVDiffIntegrator{T1,mType,F,B,T2,tType}
  alg::T1
  mesh::mType
  Flux::F
  DiffMat::B
  M::Int
  fluxes::T2
  dt :: tType
  use_threads :: Bool
end

function compute_du!(du, fluxes, mesh, noscalar::Bool, ::Type{Val{true}})
    Threads.@threads for cell in cell_indices(mesh)
        @inbounds left = left_node_idx(cell, mesh)
        @inbounds right = right_node_idx(cell, mesh)
        if noscalar
            du[:,cell] = -(fluxes[:,right] - fluxes[:,left] ) / cell_volume(mesh.mesh, cell)
        else
            du[cell] = -(fluxes[right] - fluxes[left] ) / cell_volume(mesh.mesh, cell)
        end
    end
end

function compute_du!(du, fluxes, mesh, noscalar::Bool, ::Type{Val{false}})
    for cell in cell_indices(mesh)
        @inbounds left = left_node_idx(cell, mesh)
        @inbounds right = right_node_idx(cell, mesh)
        if noscalar
            du[:,cell] = -(fluxes[:,right] - fluxes[:,left] ) / cell_volume(mesh.mesh, cell)
        else
            du[cell] = -(fluxes[right] - fluxes[left] ) / cell_volume(mesh.mesh, cell)
        end
    end
end

function (fv::FVIntegrator)(du::AbstractArray{T}, u::AbstractArray{T}, p, t) where {T}
  mesh = fv.mesh; alg = fv.alg; Flux = fv.Flux; numvars = fv.numvars
  fluxes = fv.fluxes; dt = fv.dt; use_threads = fv.use_threads
  compute_fluxes!(fluxes, Flux, u, mesh, dt, alg, numvars > 1, Val{use_threads})
  apply_bc_in_fluxes!(fluxes, mesh)
  compute_du!(du, fluxes, mesh, numvars > 1, Val{use_threads})
  apply_bc_in_du!(du, mesh)
  nothing
end

function (fv::FVDiffIntegrator)(du::AbstractArray{T}, u::AbstractArray{T}, p, t) where {T}
  mesh = fv.mesh; alg = fv.alg; Flux = fv.Flux; M = fv.numvars
  fluxes = fv.fluxes; DiffMat = fv.DiffMat; dt = fv.dt; use_threads = fv.use_threads
  compute_Dfluxes!(fluxes, Flux, DiffMat, u, mesh, dt, alg, M>1, Val{use_threads})
  apply_bc_in_fluxes!(fluxes, mesh)
  compute_du!(du, fluxes, mesh, numvars > 1, Val{use_threads})
  apply_bc_in_du!(du, mesh)
  nothing
end
