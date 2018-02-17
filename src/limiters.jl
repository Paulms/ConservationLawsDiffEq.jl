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
mutable struct DGDefaultLimiter{mType, bType}
  mesh::mType
  basis::bType
end

#TODO: Not Working
function (dgLimiter::DGDefaultLimiter)(u, f, t)
    @unpack mesh, basis = dgLimiter
    # Compute cell averages
    uh = basis.L2M*u
    uh = uh ./ [factorial(i) for i in 0:basis.order]
    ū = [0.5*(polyval(polyint(Poly(uh[:,i])),1)-polyval(polyint(Poly(uh[:,i])),-1)) for i in 1:size(uh,2)]

    # find end values of each element
    ue1 = u[1,:];ue2=u[end,:]

    #find cell averages
    vk = ū; vkm1=[ū[1],ū[1:end-1]...];vkp1=[ū[2:end]...,ū[end]]

    #apply reconstruction to find elements in need of limiting
    ve1 = vk - minmod.(vk-ue1, vk -vkm1,vkp1-vk)
    ve2 = vk + minmod.(ue2-vk,vk-vkm1,vkp1-vk)
    tol = 1e-8
    idx = (abs.(ve1-ue1).>tol)|(abs.(ve2-ue2).>tol)
    if (!isempty(idx))
      h2 = 2./diff(cell_faces(mesh))
      x1 = zeros(basis.order+1,numcells(mesh))
      for k in 1:numcells(mesh)
        x1[:,k] = reference_to_interval(basis.nodes, (cell_faces(mesh)[k],cell_faces(mesh)[k+1]))
      end
      x0 = cell_centers(mesh)
      ux = 0.5*h2.*((basis.dφₕ*u)'*basis.weights)
      #println("idx: ", idx)
      for j in 1:size(u,1)
        u[j,idx] = vk[idx]+(x1[j,idx]-x0[idx]).*
        minmod.(ux[idx],h2[idx].*(vkp1[idx]-vk[idx]),h2[idx].*(vk[idx]-vkm1[idx]))
      end
    end
end
