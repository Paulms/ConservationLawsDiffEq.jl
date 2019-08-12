abstract type AbstractFVMesh end
abstract type AbstractFVBoundaryCondition end

# Reference for cell and node indexes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1

struct fvmesh1D{MType, LBType,RBType} <: AbstractFVMesh
  mesh::MType
  left_boundary::LBType
  right_boundary::RBType
end

function mesh_setup(mesh, dbcs::Vector{T}) where {T<:AbstractFVBoundaryCondition}
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
    return fvmesh1D(mesh, left_bd, right_bd)
  else
    error("No methods available for mesh of dimension ", dim)
  end
end

@inline cell_indices(mesh) = 1:getncells(mesh.mesh)

"""
    cellval_at_left(node::Int, A::AbstractArray{T,2}, mesh::AbstractFVMesh1D) where {T}

   cell values of variable `A` to the left of `node` in `mesh`.
"""
function cellval_at_left(node::Int, A::AbstractArray{T,2}, mesh::fvmesh1D) where {T}
    idx = (:,edge-1)
    checkbounds(Bool, A, idx...) && return A[idx...]
    cellval_at_left(node, A, mesh.mesh, mesh.left_boundary)
end

"""
    cellval_at_right(edge::Int, A::AbstractArray{T,2}, mesh::AbstractFVMesh1D) where {T}

cell values of variable `A` to the right of `node` in `mesh`.
"""
function cellval_at_right(node::Int, A::AbstractArray{T,2}, mesh::fvmesh1D) where {T}
    idx = (:,edge)
    checkbounds(Bool, A, idx...) && return A[idx...]
    cellval_at_right(node, A, mesh.mesh, mesh.right_boundary)
end

"""
    get_cellvals(A::AbstractArray{T,2}, idx..., mesh::AbstractFVMesh1D) where {T}
   cell values of variable `A` on cells `idx` of `mesh`.
"""
function get_cellvals(A::AbstractArray{T,2}, mesh::fvmesh1D, idx...) where {T}
    checkbounds(Bool, A, idx...) && return A[idx...]
    if (minimum(idx[2]) < 1)
        getIndex(A, mesh.left_boundary, idx...)
    elseif (maximum(idx[2]) > getncells(mesh))
        getIndex(A, mesh.right_boundary, idx...)
    else
        error("unknown index")
    end
end

"""
    left_edge(cell::Int, mesh::AbstractFVMesh1D)

The index of the node to the left of `cell` in `mesh`.
"""
@inline function left_node_idx(cell::Int, mesh::fvmesh1D)
    @boundscheck begin
        @assert (1 <= cell <= getncells(mesh))
    end
    cell
end

"""
    right_edge(cell::Int, mesh::AbstractFVMesh1D)

The index of the node to the right of `cell` in `mesh`.
"""
@inline function right_node_idx(cell::Int, mesh::fvmesh1D)
    @boundscheck begin
        @assert (1 <= cell <= getncells(mesh))
    end
    cell+1
end

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

function getIndex(A::AbstractArray{T,2}, ::Periodic, I...) where {T}
    if typeof(I[2]) <: Int
      return A[I[1], mod1(I[2], size(A,2))]
    else
      return A[I[1],[mod1(i, size(A,2)) for i in I[2]]]
    end
end

function cellval_at_left(edge::Int, A::AbstractArray{T,2}, mesh, ::Periodic) where {T}
        A[:,mod1(edge-1, size(A,1))]
end

function cellval_at_right(edge::Int, A::AbstractArray{T,2}, mesh, ::Periodic) where {T}
        A[:,mod1(edge, size(A,1))]
end
