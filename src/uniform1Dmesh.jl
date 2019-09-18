"""
A uniform mesh in one space dimension of `N` cells
"""
struct Uniform1DFVMesh{N,T}
  Δx :: T
  cell_centers :: Vector{T}
  cell_faces :: Vector{T}
end

getncells(mesh::Uniform1DFVMesh{N}) where {N} = N
getnnodes(mesh::Uniform1DFVMesh{N}) where {N} = N+1
getnodecoords(mesh::Uniform1DFVMesh, node_idx::Int) = (mesh.cell_faces[node_idx],)
getdimension(mesh::Uniform1DFVMesh) = 1

Base.summary(::Uniform1DFVMesh{N}) where {N} = string("FV 1D Mesh of ",N," cells")

function Base.show(io::IO, A::Uniform1DFVMesh)
  println(io,summary(A))
  print(io,"cells: ")
  show(io,getncells(A))
  println(io)
  print(io,"Δx: ")
  show(io, A.Δx)
  println(io)
  print(io,"cell centers: ")
  show(io, A.cell_centers)
  println(io)
  print(io,"cell faces: ")
  show(io, A.cell_faces)
end

TreeViews.hastreeview(x::Uniform1DFVMesh) = true
function TreeViews.treelabel(io::IO,x::Uniform1DFVMesh,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io,mime,Text(Base.summary(x)))
end

function Uniform1DFVMesh(N::Int,xinit,xend)
    L = xend - xinit
    dx = L/N
    xx = [i*dx+dx/2+xinit for i in 0:(N-1)]
    faces = [xinit + dx*i for i in 0:N]
    Uniform1DFVMesh{N,typeof(xinit)}(dx,xx,faces)
end

@inline cell_volume(mesh::Uniform1DFVMesh) = mesh.Δx
@inline cell_volume(mesh::Uniform1DFVMesh,cell::Int) = mesh.Δx
cell_volumes(mesh::Uniform1DFVMesh) = mesh.Δx * ones(mesh.N)
get_nodes_matrix(mesh::Uniform1DFVMesh) = mesh.cell_faces
