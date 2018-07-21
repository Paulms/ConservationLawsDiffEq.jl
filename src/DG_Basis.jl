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

### Displays
Base.summary(basis::PolynomialBasis{T}) where {T} = string("Polynomial Basis of order ",basis.order," with data Type ",T)

function Base.show(io::IO, A::PolynomialBasis)
  println(io,summary(A))
  print(io,"quadrature nodes: ")
  show(io,A.nodes)
  println(io)
  print(io,"quadrature weights: ")
  show(io,A.weights)
  println(io)
  print(io,"polynomials: ")
  show(io, A.polynomials)
end

TreeViews.hastreeview(x::ConservationLawsDiffEq.PolynomialBasis) = true
function TreeViews.treelabel(io::IO,x::ConservationLawsDiffEq.PolynomialBasis,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io,mime,Text(Base.summary(x)))
end

"""compute Legendre polynomials coefficients, normalized to be orthonormal"""
function poly_legendre(n, ::Type{T}=Float64, var=:x) where T<:Number
    return poly_jacobi(n,0.0,0.0,T,var)
end

"""compute Jacobi polynomials coefficients, normalized to be orthonormal"""
function poly_jacobi(n, a, b, ::Type{T}=Float64, var=:x) where T<:Number
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

function legendre_basis(order, ::Type{T}=Float64) where T<:Number
  nodes, weights = gausslobatto(order+1)
  φ = fill(zero(T),order+1,order+1)
  dφ = fill(zero(T),order+1,order+1)
  polynomials = Vector{Poly}(undef,order+1)
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
   0.5*(a[2]-a[1])*ξ .+ 0.5*(a[2]+a[1])
end
