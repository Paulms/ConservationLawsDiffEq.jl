abstract type AbstractFVMesh end
abstract type AbstractFVBoundaryCondition end
abstract type AbstractFVProbType end
struct GeneralProblem <: AbstractFVProbType end
struct ScalarProblem <: AbstractFVProbType end


# Reference for cell and node indexes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

struct fvmesh1D{PType <: AbstractFVProbType, MType, LBType,RBType} <: AbstractFVMesh
  mesh::MType
  left_boundary::LBType
  right_boundary::RBType
end

function mesh_setup(mesh, dbcs::Vector{T}, PType::AbstractFVProbType) where {T<:AbstractFVBoundaryCondition}
  dim = getdimension(mesh)
  if  dim == 1
    left_bd = nothing
    right_bd = nothing
    for bdc in dbcs
        if isLeftCondition(bdc)
            left_bd = bdc
        end
        if isRightCondition(bdc)
            right_bd = bdc
        end
    end
    return fvmesh1D{typeof(PType), typeof(mesh), typeof(left_bd), typeof(right_bd)}(mesh, left_bd, right_bd)
  else
    error("No methods available for mesh of dimension ", dim)
  end
end

@inline cell_indices(mesh::AbstractFVMesh) = 1:getncells(mesh.mesh)
@inline node_indices(mesh::AbstractFVMesh) = 1:getnnodes(mesh.mesh)
@inline cell_volume(mesh::AbstractFVMesh, cell_idx::Int) = cell_volume(mesh.mesh, cell_idx)
@inline _get_cell_idx(mesh::fvmesh1D{GeneralProblem}, idx) = (:,idx)
@inline _get_cell_idx(mesh::fvmesh1D{ScalarProblem}, idx) = (idx)

@inline value_at_cell(u, j,mesh::fvmesh1D{ScalarProblem}) = u[j]
@inline value_at_cell(u, j,mesh::fvmesh1D{GeneralProblem}) = u[:,j]

"""
    cellval_at_left(node::Int, A::AbstractArray{T,2}, mesh::AbstractFVMesh1D) where {T}

   cell values of variable `A` to the left of `node` in `mesh`.
"""
function cellval_at_left(node::Int, A, mesh::fvmesh1D) where {T}
    idx = _get_cell_idx(mesh, node-1)
    checkbounds(Bool, A, idx...) && return A[idx...]
    return A[_get_cell_idx(mesh, cellidx_at_left(node, getncells(mesh.mesh), mesh.left_boundary))...]
end

"""
    cellval_at_right(edge::Int, A::AbstractArray{T,2}, mesh::AbstractFVMesh1D) where {T}

cell values of variable `A` to the right of `node` in `mesh`.
"""
function cellval_at_right(node::Int, A, mesh::fvmesh1D) where {T}
    idx = _get_cell_idx(mesh,node)
    checkbounds(Bool, A, idx...) && return A[idx...]
    return A[_get_cell_idx(mesh, cellidx_at_right(node, getncells(mesh.mesh), mesh.right_boundary))...]
end

"""
    get_cellvals(A::AbstractArray{T,2}, idx..., mesh::AbstractFVMesh1D) where {T}
   cell values of variable `A` on cells `idx` of `mesh`.
"""
function get_cellvals(A, mesh::fvmesh1D{GeneralProblem}, idx...) where {T}
    checkbounds(Bool, A, idx...) && return A[idx...]
    if (minimum(idx[2]) < 1)
        return A[idx[1],getIndex(idx[2], size(A,2), mesh.left_boundary)]
    elseif (maximum(idx[2]) > getncells(mesh))
        return A[idx[1],getIndex(idx[2], size(A,2), mesh.right_boundary)]
    else
        error("unknown index")
    end
end

function get_cellvals(A, mesh::fvmesh1D{ScalarProblem}, idx...) where {T}
    checkbounds(Bool, A, idx...) && return A[idx...]
    if (minimum(idx[1]) < 1)
        return A[getIndex(idx[1], size(A,1), mesh.left_boundary)]
    elseif (maximum(idx[1]) > getncells(mesh))
        return A[getIndex(idx[1], size(A,1), mesh.right_boundary)]
    else
        error("unknown index")
    end
end

"""
    left_edge(cell::Int, mesh::AbstractFVMesh1D)

The index of the node to the left of `cell` in `mesh`.
"""
function left_node_idx(cell::Int, mesh::fvmesh1D)
    @boundscheck begin
        @assert (1 <= cell <= getncells(mesh.mesh))
    end
    cell
end

"""
    right_edge(cell::Int, mesh::AbstractFVMesh1D)

The index of the node to the right of `cell` in `mesh`.
"""
function right_node_idx(cell::Int, mesh::fvmesh1D)
    @boundscheck begin
        @assert (1 <= cell <= getncells(mesh.mesh))
    end
    cell+1
end

function left_cell_idx(node::Int, mesh::fvmesh1D)
    @boundscheck begin
        @assert (1 <= node <= getnnodes(mesh.mesh))
    end
    cellidx_at_left(node, getncells(mesh.mesh), mesh.left_boundary)
end
function right_cell_idx(node::Int, mesh::fvmesh1D)
    @boundscheck begin
        @assert (1 <= node <= getnnodes(mesh.mesh))
    end
    cellidx_at_right(node, getncells(mesh.mesh), mesh.right_boundary)
end



function apply_bc_in_fluxes!(fluxes, mesh)
    apply_lbc_in_fluxes!(fluxes, mesh, mesh.left_boundary)
    apply_rbc_in_fluxes!(fluxes, mesh, mesh.right_boundary)
end

function apply_bc_in_du!(du, mesh)
    apply_lbc_in_du!(du, mesh, mesh.left_boundary)
    apply_rbc_in_du!(du, mesh, mesh.right_boundary)
end

apply_lbc_in_fluxes!(fluxes, mesh, ::AbstractFVBoundaryCondition) = nothing
apply_rbc_in_fluxes!(fluxes, mesh, ::AbstractFVBoundaryCondition) = nothing

apply_lbc_in_du!(du, mesh, ::AbstractFVBoundaryCondition) = nothing
apply_rbc_in_du!(du, mesh, ::AbstractFVBoundaryCondition) = nothing

################################
# Periodic boundary conditions
###############################
struct Periodic <: AbstractFVBoundaryCondition
    axis::Integer
end
function Periodic(;axis = 1)
    Periodic(axis)
end
isLeftCondition(::Periodic) = true
isRightCondition(::Periodic) = true

function getIndex(I, n::Int, ::Periodic)
    if typeof(I) <: Int
      return mod1(I, n)
    else
      return [mod1(i, n) for i in I]
    end
end

cellidx_at_left(node::Int, n::Int, ::Periodic) = mod1(node-1, n)
cellidx_at_right(node::Int, n::Int, ::Periodic) =  mod1(node, n)

################################
# Zero flux boundary conditions
###############################
struct ZeroFlux
    side::Symbol
end
ZeroFlux(;side=:Both) = ZeroFlux(side)
isLeftCondition(c::ZeroFlux) = c.side==:Both || c.side == :Left
isRightCondition(c::ZeroFlux) = c.side==:Both || c.side == :Right
function apply_lbc_in_fluxes!(fluxes, mesh, ::ZeroFlux)
    fluxes[get_cell_idx(mesh, 1)...] .= zero(eltype(fluxes))
end
function apply_rbc_in_fluxes!(fluxes, mesh, ::ZeroFlux)
    fluxes[get_cell_idx(mesh, getnnodes(mesh.mesh))...] .= zero(eltype(fluxes))
end


################################
# Dirichlet boundary conditions
###############################
struct Dirichlet
    side::Symbol
end
Dirichlet(;side=:Both) = Dirichlet(side)
isLeftCondition(c::Dirichlet) = c.side==:Both || c.side == :Left
isRightCondition(c::Dirichlet) = c.side==:Both || c.side == :Right
function apply_lbc_in_du!(du, mesh, ::ZeroFlux)
    du[get_cell_idx(mesh, 1)...] .= zero(eltype(fluxes))
end
function apply_rbc_in_du!(fluxes, mesh, ::ZeroFlux)
    du[get_cell_idx(mesh, getnnodes(mesh.mesh))...] .= zero(eltype(fluxes))
end
