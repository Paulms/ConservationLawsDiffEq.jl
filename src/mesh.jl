abstract type AbstractPolytopalMesh end
abstract type AbstractCell{dim,V,F} end

"""
A `FaceIndex` wraps an (Int, Int) and defines a face by pointing to a (cell, face).
"""
struct FaceIndex
    cellidx::Int
    idx::Int
end

"""
A `EdgeIndex` wraps an (Int, Int) and defines an edge by pointing to a (cell, edge).
"""
struct EdgeIndex
    cellidx::Int
    idx::Int
end

# Vertices
struct Node{dim,T}
    x::Tensors.Vec{dim, T}
end

@inline get_coordinates(node::Node) = node.x

#--------------- Cells
struct Cell{dim, N, M, L}
    nodes::NTuple{N,Int}
end

#Common cell types
const LineCell = Cell{1,2,1,0}
@inline get_cell_name(::LineCell) = "Line"
@inline reference_edge_nodes(::Type{LineCell}) = ((1,2),)

const TriangleCell = Cell{2,3,3,1}
@inline get_cell_name(::TriangleCell) = "Triangle"
@inline reference_edge_nodes(::Type{TriangleCell}) = ((2,3),(3,1),(1,2))

const RectangleCell = Cell{2,4,4,1}
@inline get_cell_name(::RectangleCell) = "Rectangle"
@inline reference_edge_nodes(::Type{RectangleCell}) = ((1,2),(2,3),(3,4),(4,1))

# API
@inline getnnodes(cell::Cell{dim,N}) where {dim,N} = N
@inline getnedges(cell::Cell{dim,N,M}) where {dim,N,M} = M
@inline getnfaces(cell::Cell{dim,N,M,P}) where {dim,N,M,P} = P

# ----------------- Mesh
struct PolytopalMesh{dim,T,C} <: AbstractPolytopalMesh
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{FaceIndex}}
    edgesets::Dict{String,Set{EdgeIndex}}
    nodesets::Dict{String,Set{Int}}
end

function PolytopalMesh(cells,
              nodes;
              cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facesets::Dict{String,Set{FaceIndex}}=Dict{String,Set{FaceIndex}}(),
              edgesets::Dict{String,Set{EdgeIndex}}=Dict{String,Set{EdgeIndex}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}()) where {dim}
    return PolytopalMesh(cells, nodes, cellsets, facesets, edgesets, nodesets)
end

# API
getdimension(mesh::PolytopalMesh{dim}) where {dim} = dim
@inline getncells(mesh::PolytopalMesh) = length(mesh.cells)
@inline getnnodes(mesh::PolytopalMesh) = length(mesh.nodes)
@inline getnodesidx(mesh, cell_idx) = mesh.cells[cell_idx].nodes
@inline getnodeset(mesh::PolytopalMesh, set::String) = mesh.nodesets[set]

"""
function getcoords(mesh, node_idx::Int)
Return a Tensor.Vec with the coordinates of node with index `node_idx`
"""
@inline getnodecoords(mesh::PolytopalMesh, node_idx::Int) = mesh.nodes[node_idx].x

"""
    getnodescoords(mesh::PolytopalMesh, cell_idx)
Return a vector with the coordinates of the nodes of cell number `cell`.
"""
@inline function getnodescoords(mesh::PolytopalMesh{dim,T}, cell_idx::Int) where {dim,T}
    N = getnnodes(mesh.cells[cell_idx])
    coords = Vector{Tensors.Vec{dim,T}}(undef, N)
    for (i,j) in enumerate(mesh.cells[cell_idx].nodes)
        coords[i] = mesh.nodes[j].x
    end
    return coords
end

function get_nodes_matrix(mesh::PolytopalMesh{dim,T,C}) where {dim,T,C}
    nodes_m = Matrix{T}(undef,length(mesh.nodes),dim)
    for (k,node) in enumerate(mesh.nodes)
        nodes_m[k,:] = node.x
    end
    nodes_m
end
function get_conectivity_list(mesh::PolytopalMesh{dim,T,C}) where {dim,T,C}
    cells_m = Vector()
    for k = 1:getncells(mesh)
        push!(cells_m,mesh.cells[k].nodes)
    end
    cells_m
end

function cell_volume(mesh::PolytopalMesh{2}, cell_idx::Int)
    N = getnnodes(mesh.cells[cell_idx])
    verts = getnodescoords(mesh,cell_idx)
    return 0.5*abs(sum(verts[j][1]*verts[mod1(j+1,N)][2]-verts[mod1(j+1,N)][1]*verts[j][2] for j ∈ 1:N))
end

function cell_volume(mesh::PolytopalMesh{1}, cell_idx::Int)
    verts = getnodescoords(mesh,cell_idx)
    return abs(verts[2][1] - verts[1][1])
end

function cell_centroid(mesh::PolytopalMesh{2}, cell_idx::Int)
    verts = getnodescoords(mesh,cell_idx)
    Ve = cell_volume(mesh, cell_idx)
    N = getnnodes(mesh.cells[cell_idx])
    xc = 1/(6*Ve)*sum((verts[j][1]*verts[mod1(j+1,N)][2]-verts[mod1(j+1,N)][1]*verts[j][2])*(verts[j][1]+verts[mod1(j+1,N)][1]) for j ∈ 1:N)
    yc = 1/(6*Ve)*sum((verts[j][1]*verts[mod1(j+1,N)][2]-verts[mod1(j+1,N)][1]*verts[j][2])*(verts[j][2]+verts[mod1(j+1,N)][2]) for j ∈ 1:N)
    return Tensors.Vec{2}((xc,yc))
end

function cell_diameter(mesh::PolytopalMesh{dim,T}, cell_idx::Int) where {dim,T}
    K = mesh.cells[cell_idx]
    verts = getnodescoords(mesh,cell_idx)
    h = zero(T)
     for k in reference_edge_nodes(typeof(K))
        mσ = norm(verts[k[2]] - verts[k[1]])
        h = max(h, mσ)
    end
    h
end
