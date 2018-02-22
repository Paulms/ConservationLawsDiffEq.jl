#################### Basis for Polynomial Space ####################3
struct PolynomialBasis{T}
  order::Int
  nodes::Vector{T}      #Quadrature nodes for numerical integration
  weights::Vector{T}    #Quadrature weights for numerical integration
  polynomials::Vector{Poly}
  φ::Matrix{T}     #Vandermonde matrix: basis polynomials evaluated on G-L nodes
  ψ::Matrix{T}     #basis polynomials evaluated on faces (-1,1)
  dφ::Matrix{T}
  invφ::Matrix{T}   #inverse of Gen. Vandermonde matrix
  L2M::Matrix{T}    #Legendre to monomial transform matrix
  M2L::Matrix{T}    #Monomial to legendre transform matrix
end

"""compute Legendre polynomials coefficients, normalized to be orthonormal"""
function poly_legendre{T<:Number}(n, ::Type{T}=Float64, var=:x)
    return poly_jacobi(n,0.0,0.0,T,var)
end

"""compute Jacobi polynomials coefficients, normalized to be orthonormal"""
function poly_jacobi{T<:Number}(n, a, b, ::Type{T}=Float64, var=:x)
    ox = one(T)
    zx = zero(T)
    #Compute initial P_0 and P_1
    γ0 = 2^(a+b+1)/(a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+1);
    p0 = Poly{T}([one(T)/sqrt(γ0)], var)
    if n==0; return p0; end
    γ1 = (a+1)*(b+1)/(a+b+3)*γ0
    p1 = Poly{T}([(a-b)/2/sqrt(γ1), (a+b+2)/2/sqrt(γ1)], var)
    if n==1; return p1; end
    px = Poly{T}([zero(T), one(T)], var)
    aold = 2/(2+a+b)*sqrt((a+1)*(b+1)/(a+b+3))
    for i = 1:(n-1)
        h1 = 2*i+a+b;
        anew = 2/(h1+2)*sqrt((i+1)*(i+1+a+b)*(i+1+a)*(i+1+b)/(h1+1)/(h1+3))
        bnew = -(a^2-b^2)/h1/(h1+2);
        p2 = ox/anew*(-aold*p0 + (px-bnew)*p1);
        aold =anew;
        p0 = p1
        p1 = p2
    end
    return p1
end

function legendre_to_monomials(u, basis::PolynomialBasis{T}) where {T}
    basis.L2M*u ./ [factorial(i) for i in 0:basis.order]
end

function legendre_basis{T<:Number}(order, ::Type{T}=Float64; quad_rule = gausslobatto)
  nodes, weights = quad_rule(order+1)
  φ = zeros(T,order+1,order+1)
  dφ = zeros(T,order+1,order+1)
  # TODO: # of faces depend on dimensions
  ψ = zeros(T,2,order+1)
  polynomials = Vector{Poly}(order+1)
  for n = 0:order
    p = poly_legendre(n, T)
    dp = polyder(p)
    polynomials[n+1] = p
    # Eval interior nodes
    φ[:,n+1] = polyval(p, nodes)
    dφ[:,n+1] = polyval(dp, nodes)
    # Eval faces nodes
    ψ[:,n+1] = polyval(p, [-1.0,1.0])
  end
  invφ = inv(φ)
  V = [nodes[i+1]^j/factorial(j) for i=0:order, j=0:order]
  L2M = inv(V)*φ
  M2L = inv(L2M)
  PolynomialBasis{T}(order,nodes,weights,polynomials,φ,ψ,dφ,invφ,L2M,M2L)
end

"Maps reference coordinates (ξ ∈ [-1,1]) to interval coordinates (x)"
function reference_to_interval(ξ,a::Tuple)
   0.5*(a[2]-a[1])*ξ + 0.5*(a[2]+a[1])
end

"Project function f on polynomial space Vₕ"
function project_function(f, basis, interval::Tuple; component=1)
  nodes = reference_to_interval(basis.nodes, interval)
  f_val = zeros(nodes)
  for i in 1:size(nodes,1)
    f_val[i] = f(nodes[i])[component]
  end
  function model(x,p)
    result = zeros(x)
    for i in 1:size(p,1)
      result.+=p[i]*polyval(basis.polynomials[i], x)
    end
    result
  end
  p0 = zeros(eltype(basis.nodes),basis.order+1)
  curve_fit(model, basis.nodes, f_val, p0)
end

#TODO: Dispatch on different basis types
"Get mass matrix: (2l+1)/Δx for legendre polynomials"
function get_local_mass_matrix{T}(basis::PolynomialBasis{T}, mesh)
  diagnal = zeros(T, basis.order+1)
  diagnal[:] = 2.0/(2*(0:basis.order)+1)
  M = Vector{T}(mesh.N)
  m = diagm(diagnal)
  for k in 1:mesh.N
    M[k] = mesh.cell_dx[k]/2.0*m
  end
  return M
end

"Get mass matrix inverse: Δx/(2l+1) for legendre polynomials"
function get_local_inv_mass_matrix{T}(basis::PolynomialBasis{T}, mesh::AbstractFVMesh1D)
  diagnal = zeros(T, basis.order+1)
  diagnal[:] = (2*(0:basis.order)+1) / 2.0
  M_inv = Vector{Matrix{T}}(mesh.N)
  M = diagm(diagnal)
  for k in 1:mesh.N
    M_inv[k] = 2.0./cell_volume(k, mesh)*M
  end
  return M_inv
end

"compute local inverse matrix on 1D uniform problems"
function get_local_inv_mass_matrix{T}(basis::PolynomialBasis{T}, mesh::Uniform1DFVMesh)
  diagnal = zeros(T, basis.order+1)
  diagnal[:] = (2*(0:basis.order)+1) / 2.0
  M_inv = Vector{Matrix{T}}(mesh.N)
  return 2.0/mesh.Δx*diagm(diagnal)
end
