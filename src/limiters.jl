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
struct DGLimiter{mType, bType}
  mesh::mType
  basis::bType
  limiter::AbstractDGLimiter
end

function (dgLimiter::DGLimiter)(u, f, t)
    @unpack mesh, basis, limiter = dgLimiter
    apply_limiter!(u,f,t,mesh, basis, limiter)
end

#MUSCL Limiter Van Leer
#Reference: Hesthaven, Warburton, Nodal Discontinuous Galerkin Methods Algorithms, Analysis and
#applications, pp 150.
immutable Linear_MUSCL_Limiter <: AbstractDGLimiter end

""" apply_linear receives u in modal coefficients """
function apply_limiter!(u,f,t,mesh, basis, limiter::Linear_MUSCL_Limiter)
    #Extract linear polynomial
    ul = u[:,:]; ul[3:end,:] = 0.0

    # Compute cell averages
    uavg = [basis.φ[1,1]*u[1,i] for i in 1:size(u,2)]
    #compute nodes in real coordinates
    x1 = zeros(basis.order+1,numcells(mesh))
    for k in 1:numcells(mesh)
        x1[:,k] = reference_to_interval(basis.nodes, (cell_faces(mesh)[k],cell_faces(mesh)[k+1]))
    end
    x0 = cell_centers(mesh)
    # Compute derivative
    h = cell_volumes(mesh)
    ux = 2*(1./h)'.*(basis.dφ*ul)

    uavgp1 = [uavg[2:end]...,uavg[end]]
    uavgm1 = [uavg[1],uavg[1:end-1]...]

    #limit nodal cell values
    N = size(u,1)
    for i in 1:size(u,2)
      for j in 1:N
          u[j,i] = uavg[i]+(x1[j,i]-x0[i])*
          minmod(ux[j,i],(uavgp1[i]-uavg[i])/h[i],(uavg[i]-uavgm1[i])/h[i])
      end
    end
    # convert back to modal values
    u[:,:]  = basis.invφ*u
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
