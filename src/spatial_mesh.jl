type Uniform1DFVMesh <: AbstractUniformFVMesh
  N ::Int
  x :: Vector{Float64}
  dx :: Float64
  bdtype :: Symbol
end

function Uniform1DFVMesh(N::Int,xinit::Real,xend::Real,bdtype=:ZERO_FLUX)
#Compute lenght (1D Mesh)
L = xend - xinit
dx = L/N
xx = [i*dx+dx/2+xinit for i in 0:(N-1)]
Uniform1DFVMesh(N,vec(xx),dx,bdtype)
end
