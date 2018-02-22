# DiscontinuousGalerkinScheme
# Based on:
#

struct DiscontinuousGalerkinScheme <: AbstractFEAlgorithm
  basis::PolynomialBasis
  riemann_solver::Function
  max_w_speed :: Function
end

"DiscontinuousGalerkinScheme constructor"
function DiscontinuousGalerkinScheme(basis, riemann_solver;max_w_speed = nothing)
    if max_w_speed == nothing
        max_w_speed = maxfluxρ
    end
  DiscontinuousGalerkinScheme(basis, riemann_solver, max_w_speed)
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

"Update dt based on CFL condition"
function update_dt(alg::DiscontinuousGalerkinScheme,u::AbstractArray{T2,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T2}
    ν = alg.max_w_speed(u, Flux)
    dx = maximum(cell_volumes(mesh))
    dx * CFL / (ν * (2 * alg.basis.order + 1))
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

"Calculates residual for DG method
  Inputs:
    H = matrix used to store residuals
    u = coefficients of current finite solution approx.
  residual = M_inv*(Q+F)
         where M_inv is the inverse mass matrix
              Q is the edge fluxes
              F is the interior flux"
@def scalar_1D_residual_common begin
  #Add ghost cells
  uₘ = hcat(zeros(u[:,1]),u,zeros(u[:,end]))

  #Apply boundary conditions TODO: Other boundary types
  apply_boundary(uₘ, mesh)

  #Reconstruct u in finite space: uₕ(ξ)
  uₕ = myblock(basis.φ,NC)*uₘ
  F = zeros(uₕ)
  Fₕ = zeros(uₕ)
  NN = basis.order+1
  for k in 1:size(uₕ,2)
    for j in 1:NN
      Fₕ[j:NN:size(uₕ,1),k] = f(uₕ[j:NN:size(uₕ,1),k])
    end
  end
  # Integrate interior fluxes ∫f(uₕ)φ'(ξ)dξ
  F = A_mul_B!(F,myblock(basis.dφ.*basis.weights,NC)',Fₕ)

  # Evaluate edge fluxes
  uₛ = myblock(basis.ψ,NC)*uₘ
  q = zeros(eltype(u),NC,size(uₛ,2)-1)
  for i = 1:(size(uₛ,2)-1)
    ul = uₛ[2:2:size(uₛ,1),i]; ur = uₛ[1:2:size(uₛ,1),i+1]
    q[:,i] = riemann_solver(ul,ur)
  end
  Q = zeros(F)
  for l in 1:NN
    for j in 1:NC
      Q[(j-1)*NN+l,2:end-1] = q[j,2:end] + (-1)^l*q[j,1:end-1]
    end
  end
end
function residual!(H, u, basis::PolynomialBasis, mesh::AbstractFVMesh1D, f, riemann_solver, M_inv,NC)
  @scalar_1D_residual_common
  H[:,:] = F[:,2:(end-1)]-Q[:,2:(end-1)]
  #Calculate residual
  for k in 1:mesh.N
    H[:,k] = myblock(M_inv[k],NC)*H[:,k]
  end
  if isleftdirichlet(mesh); H[:,1] = 0.0; end
  if isrightdirichlet(mesh); H[:,end] = 0.0; end
end

"Efficient residual computation for uniform problems"
function residual!(H, u, basis::PolynomialBasis, mesh::Uniform1DFVMesh, f, riemann_solver, M_inv,NC)
  @scalar_1D_residual_common
  #Calculate residual
  A_mul_B!(H,myblock(M_inv,NC),F[:,2:(end-1)]-Q[:,2:(end-1)])
  if isleftdirichlet(mesh); H[:,1] = 0.0; end
  if isrightdirichlet(mesh); H[:,end] = 0.0; end
end
