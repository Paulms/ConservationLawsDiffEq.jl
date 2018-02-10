function num_integrate(f,a,b;order=5, method = gausslegendre)
    nodes, weights = method(order);
    t_nodes = 0.5*(b-a)*nodes+0.5(b+a)
    return 0.5*(b-a)*dot(f.(t_nodes),weights)
end

function inner_slopes_loop!(∇u,j,u,mesh,θ,M)
    ul = cellval_at_left(j,u,mesh)
    ur = cellval_at_right(j+1,u,mesh)
    @inbounds for i = 1:M
      ∇u[j,i] = minmod(θ*(u[j,i]-ul[i]),(ur[i]-ul[i])/2,θ*(ur[i]-u[j,i]))
    end
end

function compute_slopes(u, mesh, θ, N, M, ::Type{Val{true}})
    ∇u = zeros(u)
    Threads.@threads for j = 1:N
        inner_slopes_loop!(∇u,j,u,mesh,θ,M)
    end
    ∇u
end

function compute_slopes(u, mesh, θ, N, M, ::Type{Val{false}})
    ∇u = zeros(u)
    for j = 1:N
        inner_slopes_loop!(∇u,j,u,mesh,θ,M)
    end
    ∇u
end

function update_dt(alg::AbstractFVAlgorithm,u::AbstractArray{T,2},Flux,
    DiffMat, CFL,mesh::Uniform1DFVMesh) where {T}
  maxρ = 0
  maxρB = 0
  N = numcells(mesh)
  for i in 1:N
    maxρ = max(maxρ, fluxρ(u[i,:], Flux))
    maxρB = max(maxρB, maximum(abs,eigvals(DiffMat(u[i,:]))))
  end
  CFL/(1/mesh.Δx*maxρ+1/(2*mesh.Δx^2)*maxρB)
end

function update_dt(alg::AbstractFVAlgorithm,u::AbstractArray{T,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T}
  maxρ = 0
  N = numcells(mesh)
  for i in 1:N
    maxρ = max(maxρ, fluxρ(u[i,:], Flux))
  end
  CFL/(1/mesh.Δx*maxρ)
end

@inline function maxfluxρ(u::AbstractArray,f)
    maxρ = zero(eltype(u))
    N = size(u,1)
    for i in 1:N
      maxρ = max(maxρ, fluxρ(u[i,:],f))
    end
    maxρ
end

@inline function fluxρ(uj::Vector,f)
  maximum(abs,eigvals(f(Val{:jac}, uj)))
end

function minmod(a,b,c)
  if (a > 0 && b > 0 && c > 0)
    min(a,b,c)
  elseif (a < 0 && b < 0 && c < 0)
    max(a,b,c)
  else
    zero(a)
  end
end

function minmod(a,b)
  0.5*(sign(a)+sign(b))*min(abs(a),abs(b))
end
