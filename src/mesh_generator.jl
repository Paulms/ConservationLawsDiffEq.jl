function _generate_2d_nodes!(nodes, nx, ny, LL, LR, UR, UL)
      for i in 0:ny-1
        ratio_bounds = i / (ny-1)

        x0 = LL[1] * (1 - ratio_bounds) + ratio_bounds * UL[1]
        x1 = LR[1] * (1 - ratio_bounds) + ratio_bounds * UR[1]

        y0 = LL[2] * (1 - ratio_bounds) + ratio_bounds * UL[2]
        y1 = LR[2] * (1 - ratio_bounds) + ratio_bounds * UR[2]

        for j in 0:nx-1
            ratio = j / (nx-1)
            x = x0 * (1 - ratio) + ratio * x1
            y = y0 * (1 - ratio) + ratio * y1
            push!(nodes, Node(Tensors.Vec{2}((x, y))))
        end
    end
end

# Check edge orientation consistency
function _check_node_data(nodes, n1,n2,n3)
    a = nodes[n2].x-nodes[n1].x
    b = nodes[n3].x-nodes[n1].x
    if (a[1]*b[2]-a[2]*b[1]) < 0
        #swap nodes 2 and 3
        return (n1,n3,n2)
    end
    return (n1,n2,n3)
end

@inline _mapToGlobalIdx(cells,cellidx,localnodeidx) = cells[cellidx].nodes[localnodeidx]

function _get_nodeset_from_edges(cells,edgeset,CellType)
    nodes = Set{Int}()
    for edge in edgeset
        push!(nodes, _mapToGlobalIdx(cells, edge.cellidx, reference_edge_nodes(CellType)[edge.idx][1]))
        push!(nodes, _mapToGlobalIdx(cells, edge.cellidx, reference_edge_nodes(CellType)[edge.idx][2]))
    end
    return nodes
end

#########################
# Line 1D
###########################
# Line
line_mesh(nel::Int, left::Number=-1.0, right::Number=1.0) =
    line_mesh(LineCell, (nel,), Tensors.Vec{1}((left,)), Tensors.Vec{1}((right,)))
"""
    line_mesh(celltype::Cell, nel::NTuple, [left::Vec, right::Vec)
Return a `mesh` for a line in 1 dimension. `celltype` defined the type of cell,
e.g. `LineCell`. `nel` is a tuple of the number of elements in each direction.
`left` and `right` are optional endpoints of the domain. Defaults to -1 and 1 in all directions.
"""
function line_mesh(::Type{LineCell}, nel::NTuple{1,Int}, left::Tensors.Vec{1,T}=Tensors.Vec{1}((-1.0,)), right::Tensors.Vec{1,T}=Tensors.Vec{1}((1.0,))) where {T}
    nel_x = nel[1]
    n_nodes = nel_x + 1

    # Generate nodes
    coords_x = collect(range(left[1], stop=right[1], length=n_nodes))
    nodes = Node{1,T}[]
    for i in 1:n_nodes
        push!(nodes, Node(Tensors.Vec{1}((coords_x[i],))))
    end

    # Generate cells
    cells = LineCell[]
    for i in 1:nel_x
        push!(cells, LineCell((i, i+1)))
    end

    # Cell node sets
    nodesets = Dict("left"  => Set{Int}([1]),
                    "right" => Set{Int}([nel_x+1]))
    return PolytopalMesh(cells, nodes; nodesets = nodesets)
end

#########################
# Triangle Cells 2D   #
#########################
"""
rectangle_mesh(::Type{RefTetrahedron}, ::Type{Val{2}}, nel::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T})
Generate a rectangular mesh with triangular cells, where `LL` is the low left node
and `UR` is the upper right one. `nel` is a tuple with the number of partions to be
used in each dimension.
"""
function rectangle_mesh(::Type{TriangleCell}, nel::NTuple{2,Int}, LL::Tensors.Vec{2,T}, UR::Tensors.Vec{2,T}) where {T}
    LR = Tensors.Vec{2}((UR[1],LL[2]))
    UL = Tensors.Vec{2}((LL[1],UR[2]))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = TriangleCell[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, TriangleCell((node_array[i,j], node_array[i+1,j], node_array[i,j+1]))) # ◺
        push!(cells, TriangleCell((node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1]))) # ◹
    end

    # Cell edges
    cell_array = reshape(collect(1:nel_tot),(2, nel_x, nel_y))
    boundary = EdgeIndex[[EdgeIndex(cl, 3) for cl in cell_array[1,:,1]];
                           [EdgeIndex(cl, 3) for cl in cell_array[2,end,:]];
                           [EdgeIndex(cl, 1) for cl in cell_array[2,:,end]];
                           [EdgeIndex(cl, 2) for cl in cell_array[1,1,:]]]
    offset = 0
    edgesets = Dict{String,Set{EdgeIndex}}()
    edgesets["bottom"] = Set{EdgeIndex}(boundary[(1:length(cell_array[1,:,1]))   .+ offset]); offset += length(cell_array[1,:,1])
    edgesets["right"]  = Set{EdgeIndex}(boundary[(1:length(cell_array[2,end,:])) .+ offset]); offset += length(cell_array[2,end,:])
    edgesets["top"]    = Set{EdgeIndex}(boundary[(1:length(cell_array[2,:,end])) .+ offset]); offset += length(cell_array[2,:,end])
    edgesets["left"]   = Set{EdgeIndex}(boundary[(1:length(cell_array[1,1,:]))   .+ offset]); offset += length(cell_array[1,1,:])
    edgesets["boundary"] = union(edgesets["bottom"],edgesets["right"],edgesets["top"],edgesets["left"])
    nodesets = Dict{String,Set{Int}}()
    for set in edgesets
        nodesets[set.first] = _get_nodeset_from_edges(cells,set.second, TriangleCell)
    end
    return PolytopalMesh(cells, nodes; edgesets = edgesets, nodesets = nodesets)
end

#########################
# Rectangle Cells 2D   #
#########################
@inline _build_cell(::Type{RectangleCell}, el_nodes, el_faces) = RectangleCell(el_nodes,(el_faces[1],el_faces[2],el_faces[3],el_faces[4]))
"""
rectangle_mesh(::Type{RectangleCell}, nel::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T}) where {T}
Generate a rectangular mesh with triangular cells, where `LL` is the low left node
and `UR` is the upper right one. `nel` is a tuple with the number of partions to be
used in each dimension.
"""
function rectangle_mesh(::Type{RectangleCell}, nel::NTuple{2,Int}, LL::Tensors.Vec{2,T}, UR::Tensors.Vec{2,T}) where {T}
    LR = Tensors.Vec{2}((UR[1],LL[2]))
    UL = Tensors.Vec{2}((LL[1],UR[2]))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = RectangleCell[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, RectangleCell((node_array[i,j], node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(nel_x, nel_y))
    boundary = EdgeIndex[[EdgeIndex(cl, 1) for cl in cell_array[:,1]];
                      [EdgeIndex(cl, 2) for cl in cell_array[end,:]];
                      [EdgeIndex(cl, 3) for cl in cell_array[:,end]];
                      [EdgeIndex(cl, 4) for cl in cell_array[1,:]]]
    # Cell face sets
    offset = 0
    edgesets = Dict{String, Set{EdgeIndex}}()
    edgesets["bottom"] = Set{EdgeIndex}(boundary[(1:length(cell_array[:,1]))   .+ offset]); offset += length(cell_array[:,1])
    edgesets["right"]  = Set{EdgeIndex}(boundary[(1:length(cell_array[end,:])) .+ offset]); offset += length(cell_array[end,:])
    edgesets["top"]    = Set{EdgeIndex}(boundary[(1:length(cell_array[:,end])) .+ offset]); offset += length(cell_array[:,end])
    edgesets["left"]   = Set{EdgeIndex}(boundary[(1:length(cell_array[1,:]))   .+ offset]); offset += length(cell_array[1,:])
    edgesets["boundary"] = union(edgesets["bottom"],edgesets["right"],edgesets["top"],edgesets["left"])
    nodesets = Dict{String,Set{Int}}()
    for set in edgesets
        nodesets[set.first] = _get_nodeset_from_edges(cells,set.second, RectangleCell)
    end
    return PolytopalMesh(cells, nodes; edgesets = edgesets, nodesets = nodesets)
end
