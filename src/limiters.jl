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


"Apply slope limiter πᴺ to u assuming u an Nth polynomial"
mutable struct DGLimiter{pType}
  problem::AbstractConservationLawProblem
  basis::PolynomialBasis
  limiter::AbstractDGLimiter
  params::pType
end

function DGLimiter(problem, basis, limiter)
    DGLimiter(problem, basis, limiter, params_init(limiter, basis, problem))
end

# Trivial parameter initialization
params_init(limiter::AbstractDGLimiter, basis::PolynomialBasis, problem::AbstractConservationLawProblem) = nothing

function (dgLimiter::DGLimiter)(u, f, t)
    @unpack problem, basis, limiter, params = dgLimiter
    apply_limiter!(u,f,t,problem, basis, params, limiter)
end

# Auxiliary functions
"""
get_cell_averages!(uavg, basis, uh, NN)
Compute cell averages of polynomial system u,
using modal values `uh`, a polynomial `basis` and number of
variables `NC`
"""
function get_cell_averages!(uavg::AbstractArray{T,2}, basis::PolynomialBasis, uh::AbstractArray{T,2}, NC::Int) where {T}
    NN = basis.order + 1
    for k = 1:NC
        uavg[k,:] = [basis.φ[1,1]*uh[(k-1)*NN+1,i] for i in 1:size(uh,2)]
    end
end

#MUSCL Limiter Van Leer
#Reference: Hesthaven, Warburton, Nodal Discontinuous Galerkin Methods Algorithms, Analysis and
#applications, pp 150.
immutable Linear_MUSCL_Limiter <: AbstractDGLimiter end

"""
apply_limiter!(u,f,t,mesh, basis, limiter::Linear_MUSCL_Limiter)
    apply Linear MUSCL limiter on function u
  """
function apply_limiter!(u,f,t,prob, basis, params, imiter::Linear_MUSCL_Limiter)
    # Compute modal coefficients
    mesh = prob.mesh; NC = prob.numvars
    NN = basis.order + 1
    uh = myblock(basis.invφ,NC)*u
    @assert basis.order > 1 "MUSCL limiter requires at least linear polynomials"
    #Extract linear polynomial
    ul = uh[:,:];
    for k = 1:NC;ul[((k-1)*NN+3):k*NN,:] = 0.0;end

    # Compute cell averages
    uavg = zeros(eltype(u), NC, numcells(mesh))
    get_cell_averages!(uavg, basis, uh, NC)

    #compute nodes in real coordinates
    x1 = zeros(basis.order+1,numcells(mesh))
    for k in 1:numcells(mesh)
        x1[:,k] = reference_to_interval(basis.nodes, (cell_faces(mesh)[k],cell_faces(mesh)[k+1]))
    end
    x0 = cell_centers(mesh)
    # Compute derivative
    h = cell_volumes(mesh)
    ux = 2*(1./h)'.*(myblock(basis.dφ,NC)*ul)

    ve = apply_boundary(uavg, mesh)
    uavgp1 = ve[:,3:end];  uavgm1 = ve[:,1:end-2]

    #limit nodal cell values
    for i in 1:size(u,2)
        for k = 1:NC
            for j in 1:NN
                u[(k-1)*NN+j,i] = uavg[k,i]+(x1[j,i]-x0[i])*
                minmod(ux[(k-1)*NN+j,i],(uavgp1[k,i]-uavg[k,i])/h[i],(uavg[k,i]-uavgm1[k,i])/h[i])
            end
      end
    end
end

#WENO limiter by Zhong-Shu (2013)
mutable struct WENO_Limiter <: AbstractDGLimiter end

function params_init(WENO_Limiter::WENO_Limiter, basis::PolynomialBasis, problem::AbstractConservationLawProblem)
    return WENOLimiterWeights(basis.order, basis.invφ)
end

"""
function WENOLimiterWeights(m, iV);
Compute operators to enable evaluation of WENO smoothness
indicator and WENO polynomial of order m.
"""
function WENOLimiterWeights(m,iV);
    Q = zeros(m+1,m+1)
    Pmat = zeros(m+1,m+1)
    Xm = Pmat; Xp = Pmat;

    # Compute quadrature points
    (x,w) = gausslegendre(m+1)
    Λ = diagm(w);

    # Initial matrices of Legendre polynomails
    for n = 0:m
      p = poly_legendre(n)
      # Eval nodes
      Pmat[n+1,:] = polyval(p, x)
      Xm[n+1,:] = polyval(p, x-2)
      Xp[n+1,:] = polyval(p, x+2)
    end

    # Compute matrices corresponding to increasing order of derivative
    for l=1:m
        # Set up operator to recover derivaties
        A = zeros(m+2-l,m+2-l); A[1,1] = 1/sqrt((2*l+1)*(2*l-1))
        A[m+2-l,m+2-l] = 1/(sqrt(2*(m+2)+1)*sqrt(2*(m+2)-1))
        for i=2:m-l+1
            Ah = 1/(sqrt(2*(l-1+i)+1)*sqrt(2*(l-1+i)-1))
            A[i,i] = Ah; A[i+1,i-1] = -Ah
        end

        # Recover derivatives at quadrature points
        Ph1 = A\Pmat[l:m+1,:]
        Pmat[1:l,:]=0; Pmat[l+1:m+1,:] = Ph1[1:m-l+1,:]

        # Compute smoothness operator for order l and update
        Qh = Pmat*Λ*Pmat'
        Q = Q + 2^(2*l-1)*Qh
    end

    # Initialize operator for smoothness indicator in nodal space
    Q = basis.invφ'*Q*basis.invφ;

    # Initialize interpolation matrices
    Xp = basis.invφ'*Xp; Xm = basis.invφ'*Xm
    return Q,Xm,Xp
end

"""
apply_limiter!(u,f,t,mesh, basis, limiter::WENO_Limiter)
    apply Zhong Shu WENO limiter on function u
  """
function apply_limiter!(u,f,t,prob, basis, params, limiter::WENO_Limiter)
    mesh = prob.mesh; NC = prob.numvars; NN = basis.order + 1
    (Q, Xp, Xm) = params
    Xp = myblock(Xp,NC)
    Xm = myblock(Xm,NC)
    iV = myblock(basis.invφ,NC)
    V = myblock(basis.φ,NC)
    eps0=1.0e-6;

    # Set constants for limiting
    eps1 = 1e-10; p=1;
    γm1 =0.001; γ0 = 0.998; γp1 = 0.001;

    # Compute cell averages and cell centers
    uh = iV*u
    uavg = zeros(eltype(u), NC, numcells(mesh))
    get_cell_averages!(uavg, basis, uh, NC)

    # Compute extended polynomials with zero cell averages
    ue = apply_boundary(u, mesh);
    Pm = Xp'*ue; Pp = Xm'*ue;
    Ph = iV*Pm; Ph[1:NN:end,:]=0; Pm = V*Ph;
    Ph = iV*Pp; Ph[1:NN:end,:]=0; Pp = V*Ph;

    # Extend cell averages
    ve = apply_boundary(uavg, mesh)

    # extract end values and cell averages for each element
    uel = u[1:NN:end,:]; uer = u[NN:NN:end,:]
    vj = uavg; vjm = ve[:,1:end-2]; vjp = ve[:,3:end]

    for k = 1:NC
        # Find elements that require limiting
        vel = vj[k,:] - minmod.(vj[k,:]-uel[k,:],vj[k,:]-vjm[k,:],vjp[k,:]-vj[k,:])
        ver = vj[k,:] + minmod.(uer[k,:]-vj[k,:],vj[k,:]-vjm[k,:],vjp[k,:]-vj[k,:])
        ids = union(find(x->abs(x)>eps0, vel-uel[k,:]), find(x->abs(x)>eps0, ver-uer[k,:]))

        # Apply limiting when needed
        if (!isempty(ids))
            # Extract local polynomials
            pm1 = Pm[(k-1)*NN+1:k*NN,ids] + ones(NN,1)*vj[k,ids]'
            p0  =  u[(k-1)*NN+1:k*NN,ids]
            pp1 = Pp[(k-1)*NN+1:k*NN,ids+2] + ones(NN,1)*vj[k,ids]'

            # Compute smoothness indicators and WENO weights
            βm1 = diag(pm1'*Q*pm1); αm1 = γm1./(eps1 + βm1).^(2*p);
            β0 = diag(p0'*Q*p0); α0  =  γ0./(eps1 + β0).^(2*p);
            βp1 = diag(pp1'*Q*pp1); αp1 = γp1./(eps1 + βp1).^(2*p);

            αs = αm1 + α0 + αp1;
            omm1 = αm1./αs; om0  = α0./αs; omp1 = αp1./αs;

            # Compute limited function
            u[(k-1)*NN+1:k*NN,ids] = pm1*diagm(omm1) + p0*diagm(om0) + pp1*diagm(omp1)
        end
    end
end


#TODO: Not Working
# immutable MUSCLV2Limiter <: AbstractDGLimiter end
#
# function apply_limiter(u,f,t,mesh, basis, limiter::MUSCLV2Limiter)
#     # Compute cell averages
#     ̄ū = [0.5*dot(basis.φ*u[:,i], basis.weights) for i in 1:size(u,2)]
#
#     # find end values of each element
#     ue1 = u[1,:];ue2=u[end,:]
#
#     #find cell averages
#     vk = ū; vkm1=[ū[1],ū[1:end-1]...];vkp1=[ū[2:end]...,ū[end]]
#
#     #apply reconstruction to find elements in need of limiting
#     ve1 = vk - minmod.(vk-ue1, vk -vkm1,vkp1-vk)
#     ve2 = vk + minmod.(ue2-vk,vk-vkm1,vkp1-vk)
#     tol = 1e-8
#     idx = (abs.(ve1-ue1).>tol)|(abs.(ve2-ue2).>tol)
#     if (!isempty(idx))
#       h2 = 2./diff(cell_faces(mesh))
#       x1 = zeros(basis.order+1,numcells(mesh))
#       for k in 1:numcells(mesh)
#         x1[:,k] = reference_to_interval(basis.nodes, (cell_faces(mesh)[k],cell_faces(mesh)[k+1]))
#       end
#       x0 = cell_centers(mesh)
#       ux = 0.5*h2.*((basis.dφ*u)'*basis.weights)
#       #println("idx: ", idx)
#       for j in 1:size(u,1)
#         u[j,idx] = vk[idx]+(x1[j,idx]-x0[idx]).*
#         minmod.(ux[idx],h2[idx].*(vkp1[idx]-vk[idx]),h2[idx].*(vk[idx]-vkm1[idx]))
#       end
#     end
# end
