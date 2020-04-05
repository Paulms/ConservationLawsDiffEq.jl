# Slope Limiters
# note that r = (u - ul)/(ur - u)
# and we set a = u - ul, b = ur - u to define the limiters

function minmod(a,b,c)
  if (a > 0 && b > 0 && c > 0)
    min(a,b,c)
  elseif (a < 0 && b < 0 && c < 0)
    max(a,b,c)
  else
    zero(a)
  end
end

struct GeneralizedMinmodLimiter <: AbstractSlopeLimiter
    θ::Float64
end

GeneralizedMinmodLimiter(;θ=1.0) = GeneralizedMinmodLimiter(θ)
(limiter::GeneralizedMinmodLimiter)(a,b) =minmod(limiter.θ*a,0.5*(a+b),limiter.θ*b)

function minmod(a,b)
  0.5*(sign(a)+sign(b))*min(abs(a),abs(b))
end

struct MinmodLimiter <: AbstractSlopeLimiter end
(::MinmodLimiter)(a,b) = minmod(a,b)

struct OsherLimiter <: AbstractSlopeLimiter
    β::Float64
end
OsherLimiter(;β=1.0) = OsherLimiter(β)
(limiter::OsherLimiter)(a,b) = max(zero(a),min(a,limiter.β*b))

struct SuperbeeLimiter <: AbstractSlopeLimiter end
(::SuperbeeLimiter)(a,b) = max(zero(a),min(2*a,b),min(a,2*b))

"""
function compute_slopes(u, mesh, θ, M, ::Type{Val{true}})
  Estimate slopes of the discretization of function u,
      using a generalized minmod limiter
  inputs:
  `u` discrete approx of function u
  'nonscalar' bool
  `θ` parameter of generalized minmod limiter
  `mesh` problem mesh
  `Type{Val}` bool to choose threaded version
"""
function compute_slopes(u::AbstractArray, mesh::AbstractFVMesh, slopeLimiter::AbstractSlopeLimiter, nonscalar::Bool, ::Type{Val{true}})
  ∇u = similar(u)
  Threads.@threads for j in cell_indices(mesh)
      inner_slopes_loop!(∇u,j,u,mesh,slopeLimiter,nonscalar)
  end
  ∇u
end

function compute_slopes(u::AbstractArray, mesh::AbstractFVMesh, slopeLimiter::AbstractSlopeLimiter, nonscalar::Bool, ::Type{Val{false}})
  ∇u = similar(u)
  for j in cell_indices(mesh)
      inner_slopes_loop!(∇u,j,u,mesh,slopeLimiter,nonscalar)
  end
  ∇u
end

function inner_slopes_loop!(∇u,j,u,mesh,slopeLimiter::AbstractSlopeLimiter,nonscalar)
  ul = cellval_at_left(j,u,mesh)
  ur = cellval_at_right(j+1,u,mesh)
  if nonscalar
    @inbounds for i = 1:size(u,1)
      ∇u[i,j] = slopeLimiter(u[i,j]-ul[i], ur[i]-u[i,j])
    end
  else
    ∇u[j] = slopeLimiter(u[j]-ul, ur-u[j])
  end
end