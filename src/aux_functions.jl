"""
num_integrate(f,a,b;order=5, method = gausslegendre)

Numerical integration helper function.
    Integrates function f on interval [a, b], using quadrature rule given by
        `method` of order `order`
"""
function num_integrate(f,a,b;order=5, method = gausslegendre)
    nodes, weights = method(order);
    t_nodes = 0.5*(b-a)*nodes+0.5(b+a)
    M = length(f(a))
    tmp = zeros(M)
    for i in 1:M
        g(x) = f(x)[i]
        tmp[i] = 0.5*(b-a)*dot(g.(t_nodes),weights)
    end
    return tmp
end

function inner_slopes_loop!(∇u,j,u,mesh,θ,M)
    ul = cellval_at_left(j,u,mesh)
    ur = cellval_at_right(j+1,u,mesh)
    @inbounds for i = 1:M
      ∇u[j,i] = minmod(θ*(u[j,i]-ul[i]),(ur[i]-ul[i])/2,θ*(ur[i]-u[j,i]))
    end
end

"""
function compute_slopes(u, mesh, θ, M, ::Type{Val{true}})
    Estimate slopes of the discretization of function u,
        using a generalized minmod limiter
    inputs:
    `u` discrete approx of function u
    `M` number of variables
    `θ` parameter of generalized minmod limiter
    `mesh` problem mesh
    `Type{Val}` bool to choose threaded version
"""
function compute_slopes(u::AbstractArray, mesh::AbstractFVMesh1D, θ, M::Int, ::Type{Val{true}})
    ∇u = similar(u)
    Threads.@threads for j in cell_indices(mesh)
        inner_slopes_loop!(∇u,j,u,mesh,θ,M)
    end
    ∇u
end

function compute_slopes(u::AbstractArray, mesh::AbstractFVMesh1D, θ, M::Int, ::Type{Val{false}})
    ∇u = zeros(u)
    for j in cell_indices(mesh)
        inner_slopes_loop!(∇u,j,u,mesh,θ,M)
    end
    ∇u
end

function update_dt(alg::AbstractFVAlgorithm,u::AbstractArray{T,2},Flux,
    DiffMat, CFL,mesh::Uniform1DFVMesh) where {T}
  maxρ = zero(T)
  maxρB = zero(T)
  for i in cell_indices(mesh)
    maxρ = max(maxρ, fluxρ(u[i,:], Flux))
    maxρB = max(maxρB, maximum(abs,eigvals(DiffMat(u[i,:]))))
  end
  CFL/(1/mesh.Δx*maxρ+1/(2*mesh.Δx^2)*maxρB)
end

function scheme_short_name(alg::AbstractFVAlgorithm)
    b = string(typeof(alg))
    replace(b[search(b, ".")[1]+1:end], r"(Algorithm)", s"")
end

function update_dt(alg::AbstractFVAlgorithm,u::AbstractArray{T,2},Flux,
    CFL,mesh::Uniform1DFVMesh) where {T}
  maxρ = zero(T)
  for i in cell_indices(mesh)
    maxρ = max(maxρ, fluxρ(u[i,:], Flux))
  end
  CFL/(1/mesh.Δx*maxρ)
end

@inline function maxfluxρ(u::AbstractArray{T,2},f) where {T}
    maxρ = zero(T)
    N = size(u,1)
    for i in 1:N
      maxρ = max(maxρ, fluxρ(u[i,:],f))
    end
    maxρ
end

@inline function fluxρ(uj::Vector,f)
  maximum(abs,eigvals(f(Val{:jac}, uj)))
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
