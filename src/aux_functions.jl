"""
num_integrate(f,a,b;order=5, method = gausslegendre)

Numerical integration helper function.
    Integrates function f on interval [a, b], using quadrature rule given by
        `method` of order `order`
"""
function num_integrate(f,a,b;order=5, method = gausslegendre)
    nodes, weights = method(order);
    t_nodes = 0.5*(b-a)*nodes .+ 0.5*(b+a)
    M = length(f(a))
    tmp = fill(0.0,M)
    for i in 1:M
        g(x) = f(x)[i]
        tmp[i] = 0.5*(b-a)*dot(g.(t_nodes),weights)
    end
    return tmp
end

function inner_slopes_loop!(∇u,j,u,mesh,slopeLimiter::AbstractSlopeLimiter,M)
    ul = cellval_at_left(j,u,mesh)
    ur = cellval_at_right(j+1,u,mesh)
    @inbounds for i = 1:M
      ∇u[j,i] = slopeLimiter(u[j,i]-ul[i], ur[i]-u[j,i])
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
function compute_slopes(u::AbstractArray, mesh::AbstractFVMesh1D, slopeLimiter::AbstractSlopeLimiter, M::Int, ::Type{Val{true}})
    ∇u = similar(u)
    Threads.@threads for j in cell_indices(mesh)
        inner_slopes_loop!(∇u,j,u,mesh,slopeLimiter,M)
    end
    ∇u
end

function compute_slopes(u::AbstractArray, mesh::AbstractFVMesh1D, slopeLimiter::AbstractSlopeLimiter, M::Int, ::Type{Val{false}})
    ∇u = similar(u)
    for j in cell_indices(mesh)
        inner_slopes_loop!(∇u,j,u,mesh,slopeLimiter,M)
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
    b
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

function fluxρ(uj::SArray{Tuple{1},Float64,1,1},f) where {T}
    return abs(f(Val{:jac}, uj)[1])
end

function fluxρ(uj::AbstractArray{T,1},f) where {T}
        maximum(abs,eigvals(f(Val{:jac}, uj)))
end

"build a block diagonal matrix by repeating a matrix N times"
function myblock(A::AbstractArray{T,2},N::Int) where {T}
  M = size(A,1)
  Q = size(A,2)
  B = fill(zero(T),M*N,Q*N)
  for i = 1:N
    B[(i-1)*M+1:i*M,(i-1)*Q+1:i*Q] = A
  end
  B
end

#Flux Splittings
function glf_splt_inner_loop!(fminus, fplus, j, u, α, Flux)
    fminus[j,:] = 0.5*(Flux(u[j,:])-α*u[j,:])
    fplus[j,:] = 0.5*(Flux(u[j,:])+α*u[j,:])
end
function glf_splitting(u, α, Flux, N, ::Type{Val{true}})
  # Lax Friedrichs flux splitting
  fminus = similar(u); fplus = similar(u)
  Threads.@threads for j = 1:N
      glf_splt_inner_loop!(fminus, fplus, j, u, α, Flux)
  end
  fminus, fplus
end

function glf_splitting(u, α, Flux, N, ::Type{Val{false}})
  # Lax Friedrichs flux splitting
  fminus = similar(u); fplus = similar(u)
  for j = 1:N
      glf_splt_inner_loop!(fminus, fplus, j, u, α, Flux)
  end
  fminus, fplus
end

function llf_splitting(u, mesh, Flux)
  # Lax Friedrichs flux splitting
  fminus = similar(u); fplus = similar(u)
  ul=cellval_at_left(1,u,mesh)
  αl = fluxρ(ul, Flux)
  αr = zero(αl)
  for j = cell_indices(mesh)
    ur=cellval_at_right(j,u,mesh)
    αr = fluxρ(ur, Flux)
    αk = max(αl, αr)
    fminus[j,:] = 0.5*(Flux(u[j,:])-αk*u[j,:])
    fplus[j,:] = 0.5*(Flux(u[j,:])+αk*u[j,:])
    αl = αr
  end
  fminus,fplus
end
