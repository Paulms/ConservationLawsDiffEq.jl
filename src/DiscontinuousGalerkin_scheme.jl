# DiscontinuousGalerkinScheme
# Based on:
#

mutable struct DiscontinuousGalerkinScheme{T,vType} <: AbstractFEAlgorithm
  basis::PolynomialBasis
  riemann_solver::Function
  max_w_speed :: Function
  D::AbstractArray{T,2}
  Ma::AbstractArray{T,2}    #Mass Matrix on reference element
  S::AbstractArray{T,2}     #Stiffness matrix on reference element
  ν::vType                  #Max wave speed
end

"DiscontinuousGalerkinScheme constructor"
function DiscontinuousGalerkinScheme(basis, riemann_solver;max_w_speed = nothing)
    if max_w_speed == nothing
        max_w_speed = maxfluxρ
    end
    D = basis.dφ*basis.invφ
    Ma = inv(basis.φ*basis.φ')
  DiscontinuousGalerkinScheme(basis, riemann_solver, max_w_speed, D, Ma, Ma*D, 0.0)
end

"Reconstruc solution from basis space"
function reconstruct_u(u::AbstractArray{T,2}, φ::AbstractArray{T2,2}, NC::Int) where {T, T2}
  uh = myblock(φ,NC)*u
  NN = size(φ,1); Nx = size(u,2)
  uₕ = zeros(T, NN*Nx,NC)
  for j in 1:NC
    uₕ[:,j] = uh[(j-1)*NN+1:j*NN,:][:]
  end
  return uₕ
end

"Flat x nodes on u matrix"
function flat_u(u::AbstractArray{T,2}, order::Int, NC::Int) where {T}
  NN = order + 1
  uh = zeros(T, size(u,2)*NN, NC)
  for j in 1:NC
    uh[:,j] = u[(j-1)*NN+1:j*NN,:][:]
  end
  return uh
end

"Get cell face values for each variable"
function get_dg_face_values(u::AbstractArray{T,2}, order::Int, NC::Int) where {T}
    NN = order + 1
    #TODO: number of faces depend on dimension
    us = zeros(T, 2*NC, size(u,2))
    for j in 1:size(u,2)
      for k in 1:NC
          us[(2*k-1):k*2,j] = u[[1+NN*(k-1),NN+NN*(k-1)],j]
      end
    end
    return us
end

"Update dt based on CFL condition"
function update_dt(alg::DiscontinuousGalerkinScheme,u::AbstractArray{T2,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T2}
    alg.ν = alg.max_w_speed(u, Flux)
    dx = maximum(cell_volumes(mesh))
    dx * CFL*min(abs(alg.basis.nodes[1]-alg.basis.nodes[2])) / alg.ν
end

"Compute right hand side for time integration"
function residual!(H, u, basis::PolynomialBasis, mesh::Uniform1DFVMesh, alg::DiscontinuousGalerkinScheme, f, riemann_solver, NC, ::Type{Val{false}})
    NN = basis.order + 1

    us = get_dg_face_values(u, basis.order, NC)
    #Add ghost cells
    us = hcat(zeros(us[:,1]),us,zeros(us[:,end]))
    #Apply boundary conditions TODO: Other boundary types
    apply_boundary(us, mesh)
    q = zeros(u)
    F = zeros(u)
    ur=us[1:2:end,:]
    ul=us[2:2:end,:]
    for i = 1:numcells(mesh)
        # Evaluate edge fluxes
        q[NN:NN:end,i] = riemann_solver(ul[:,i+1],ur[:,i+2], f, alg.ν)
        q[1:NN:end,i] = -riemann_solver(ul[:,i],ur[:,i+1], f, alg.ν)
        # Integrate interior fluxes ∫f(uₕ)φ'(ξ)dξ
        for k = 1:NN
            F[k:NN:end,i] = f(u[k:NN:end,i])
        end
    end

    # Compute right hand size
    ru = myblock(alg.S',NC)*F - q
    h = maximum(cell_volumes(mesh))
    H[:,:] = (h/2*myblock(alg.Ma,NC))\ru;
    if isleftdirichlet(mesh); H[:,1] = 0.0; end
   if isrightdirichlet(mesh); H[:,end] = 0.0; end
end

"Apply boundary conditions on scalar problems"
function apply_boundary(u::AbstractArray{T,2}, mesh::AbstractFVMesh1D) where {T}
  if isleftperiodic(mesh)
      u[:,1] = u[:,end-1]
  elseif  isleftzeroflux(mesh)
      u[:,1] = u[:,2]
  end
  if isrightperiodic(mesh)
      u[:,end] = u[:,2]
  elseif  isrightzeroflux(mesh)
      u[:,end] = u[:,end-1]
  end
end

"build a block diagonal matrix by repeating a matrix N times"
function myblock(A::AbstractArray{T,2},N::Int) where {T}
  M = size(A,1)
  Q = size(A,2)
  B = zeros(T,M*N,Q*N)
  for i = 1:N
    B[(i-1)*M+1:i*M,(i-1)*Q+1:i*Q] = A
  end
  B
end
