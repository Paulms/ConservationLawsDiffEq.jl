#################### Basis for Polynomial Space ####################3
struct PolynomialBasis{T}
  order::Int
  nodes::Vector{T}      #Quadrature nodes for numerical integration
  weights::Vector{T}    #Quadrature weights for numerical integration
  polynomials::Vector{Poly}
  φ::Matrix{T}     #Vandermonde matrix: basis polynomials evaluated on G-L nodes
  dφ::Matrix{T}
  invφ::Matrix{T}   #inverse of Gen. Vandermonde matrix
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

function legendre_basis{T<:Number}(order, ::Type{T}=Float64)
  nodes, weights = gausslobatto(order+1)
  φ = zeros(T,order+1,order+1)
  dφ = zeros(T,order+1,order+1)
  polynomials = Vector{Poly}(order+1)
  for n = 0:order
    p = poly_legendre(n, T)
    dp = polyder(p)
    polynomials[n+1] = p
    # Eval interior nodes
    φ[:,n+1] = polyval(p, nodes)
    dφ[:,n+1] = polyval(dp, nodes)
  end
  invφ = inv(φ)
  PolynomialBasis{T}(order,nodes,weights,polynomials,φ,dφ,invφ)
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
