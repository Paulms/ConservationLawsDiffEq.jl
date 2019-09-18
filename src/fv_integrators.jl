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

mutable struct FVDiffIntegrator{T1,mType,F,B, cType,T2,tType}
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

function update_dt!(u::AbstractArray{T,2},fv::FVIntegrator) where {T}
  fv.dt = update_dt(fv.alg, u, fv.Flux, fv.CFL, fv.mesh)
  fv.dt
end

function update_dt!(u::AbstractArray{T,2},fv::FVDiffIntegrator) where {T}
  fv.dt = update_dt(fv.alg, u, fv.Flux, fv.DiffMat, fv.CFL, fv.mesh)
  fv.dt
end

function (fv::FVIntegrator)(du::AbstractArray{T,2}, u::AbstractArray{T,2}, p, t) where {T}
  mesh = fv.mesh; alg = fv.alg; Flux = fv.Flux; M = fv.M;
  fluxes = fv.fluxes; dt = fv.dt; use_threads = fv.use_threads
  compute_fluxes!(fluxes, Flux, u, mesh, dt, M, alg, Val{use_threads})
  if isleftzeroflux(mesh);fluxes[1,:] .= zero(T); end
  if isrightzeroflux(mesh);fluxes[numedges(mesh),:] .= zero(T);end
  compute_du!(du, fluxes, mesh, Val{use_threads})
  if isleftdirichlet(mesh);du[1,:] .= zero(T); end
  if isrightdirichlet(mesh);fluxes[numcells(mesh),:] .= zero(T);end
  nothing
end

function (fv::FVDiffIntegrator)(du::AbstractArray{T,2}, u::AbstractArray{T,2}, p, t) where {T}
  mesh = fv.mesh; alg = fv.alg; Flux = fv.Flux; M = fv.M
  fluxes = fv.fluxes; DiffMat = fv.DiffMat; dt = fv.dt; use_threads = fv.use_threads
  compute_Dfluxes!(fluxes, Flux, DiffMat, u, mesh, dt, M, alg, Val{use_threads})
  if isleftzeroflux(mesh);fluxes[1,:] = zero(T); end
  if isrightzeroflux(mesh);fluxes[numedges(mesh),:] = zero(T);end
  compute_du!(du, fluxes, mesh, Val{use_threads})
  if isleftdirichlet(mesh);du[1,:] = zero(T); end
  if isrightdirichlet(mesh);fluxes[numcells(mesh),:] = zero(T);end
  nothing
end


function compute_du!(du, fluxes, mesh::AbstractFVMesh1D, ::Type{Val{true}})
    Threads.@threads for cell in cell_indices(mesh)
        @inbounds left = left_edge(cell, mesh)
        @inbounds right = right_edge(cell, mesh)
        @inbounds du[cell,:] = -( fluxes[right,:] - fluxes[left,:] ) / cell_volume(cell, mesh)
    end
end

function compute_du!(du, fluxes, mesh::AbstractFVMesh1D, ::Type{Val{false}})
    for cell in cell_indices(mesh)
        @inbounds left = left_edge(cell, mesh)
        @inbounds right = right_edge(cell, mesh)
        @inbounds du[cell,:] = -( fluxes[right,:] - fluxes[left,:] ) / cell_volume(cell, mesh)
    end
end
