function num_integrate(f,a,b;order=5, method = FastGaussQuadrature.gausslegendre)
    nodes, weights = method(order);
    t_nodes = 0.5*(b-a)*nodes .+ 0.5*(b+a)
    M = length(f(a))
    if M == 1
        tmp = 0.5*(b-a)*dot(f.(t_nodes),weights)
    else
        tmp = fill(0.0,M)
        for i in 1:M
            g(x) = f(x)[i]
            tmp[i] = 0.5*(b-a)*dot(g.(t_nodes),weights)
        end
    end
    return tmp
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
function glf_splt_inner_loop!(fminus, fplus, j, u, α, Flux, nonscalar::Bool)
    if nonscalar
        fminus[:,j] = 0.5*(Flux(u[:,j])-α*u[:,j])
        fplus[:,j] = 0.5*(Flux(u[:,j])+α*u[:,j])
    else
        fminus[j] = 0.5*(Flux(u[j])-α*u[j])
        fplus[j] = 0.5*(Flux(u[j])+α*u[j])
    end
end

function glf_splitting(u, α, Flux, nonscalar, ::Type{Val{true}})
  # Lax Friedrichs flux splitting
  fminus = similar(u); fplus = similar(u)
  Threads.@threads for j = 1:size(u,2)
      glf_splt_inner_loop!(fminus, fplus, j, u, α, Flux, nonscalar)
  end
  fminus, fplus
end

function glf_splitting(u, α, Flux, nonscalar, ::Type{Val{false}})
  # Lax Friedrichs flux splitting
  fminus = similar(u); fplus = similar(u)
  N = nonscalar ? size(u,2) : size(u,1)
  for j = 1:N
      glf_splt_inner_loop!(fminus, fplus, j, u, α, Flux, nonscalar)
  end
  fminus, fplus
end

function llf_splitting(u, mesh, Flux, nonscalar)
  # Lax Friedrichs flux splitting
  fminus = similar(u); fplus = similar(u)
  ul=cellval_at_left(1,u,mesh)
  αl = fluxρ(ul, Flux)
  αr = zero(αl)
  for j = 1:getncells(mesh)
    ur=cellval_at_right(j,u,mesh)
    αr = fluxρ(ur, Flux)
    αk = max(αl, αr)
    if nonscalar
        fminus[:,j] = 0.5*(Flux(u[:,j])-αk*u[:,j])
        fplus[:,j] = 0.5*(Flux(u[:,j])+αk*u[:,j])
    else
        fminus[j] = 0.5*(Flux(u[j])-αk*u[j])
        fplus[j] = 0.5*(Flux(u[j])+αk*u[j])
    end
    αl = αr
  end
  fminus,fplus
end