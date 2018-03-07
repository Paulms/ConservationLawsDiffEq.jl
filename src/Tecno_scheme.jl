# Based on
# U. Fjordholm, S. Mishra, E. Tadmor, Arbitrarly high-order accurate entropy
# stable essentially nonoscillatory schemes for systems of conservation laws.
# 2012. SIAM. vol. 50. No 2. pp. 544-573

immutable FVTecnoAlgorithm <: AbstractFVAlgorithm
  order :: Int
  Nflux :: Function #Entropy stable 2 point flux
  ve    :: Function #Entropy variable
end

function FVTecnoAlgorithm(Nflux;order=2.0, ve = u::Vector -> u)
  FVTecnoAlgorithm(order, Nflux, ve)
end

"""
compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVTecnoAlgorithm, ::Type{Val{false}})
Numerical flux of Tecno Scheme in 1D
"""
function compute_fluxes!(hh, Flux, u, mesh, dt, M, alg::FVTecnoAlgorithm, ::Type{Val{false}})
    @unpack order, Nflux, ve = alg
    N = numcells(mesh)
    dx = mesh.Δx
    #Eno Reconstrucion
    @inbounds ul=cellval_at_left(1,u,mesh)
    @inbounds ur=cellval_at_right(1,u,mesh)
    RΛ1 = eigfact(Flux(Val{:jac},0.5*(ul+ur)))
    MatR = Vector{typeof(RΛ1.vectors)}(0)
    MatΛ = Vector{typeof(RΛ1.values)}(0)
    push!(MatR,RΛ1.vectors)
    push!(MatΛ,RΛ1.values)
    for j = 2:(N+1)
      @inbounds ul=cellval_at_left(j,u,mesh)
      @inbounds ur=cellval_at_right(j,u,mesh)
      RΛj = eigfact(Flux(Val{:jac},0.5*(ul+ur)))
      push!(MatR,RΛj.vectors); push!(MatΛ,RΛj.values)
    end
    dd = zeros(N+1,M) #Extra numerical diffusion
    k = order - 1
    weights = unif_crj(order)
    v = zeros(u)
    for j = indices(v, 1)
      v[j,:] = ve(u[j,:]) #entropy variables
    end
    vminus = zeros(N,M); vplus = zeros(N,M)
    wminus = zeros(N,M); wplus = zeros(N,M)
    for j in cell_indices(mesh)
      for i = 1:M
        v_eno = get_cellvals(v,mesh,(j-k:j+k,i)...)
        vminus[j,i],vplus[j,i] = ENO_urec(dx,v_eno,order,weights)
      end
      wminus[j,:] = MatR[j]'*vminus[j,:]
      wplus[j,:] = MatR[j+1]'*vplus[j,:]
    end
    wdiff = zeros(N+1,M)
    for j = 2:N
      wdiff[j,:] = wminus[j,:] - wplus[j-1,:]
    end
    if isleftperiodic(mesh);wdiff[1,:] = wminus[1,:] - wplus[N,:];end
    if isrightperiodic(mesh);wdiff[N+1,:] = wdiff[1,:];end
    for j in edge_indices(mesh)
      dd[j,:] = MatR[j]*[abs(MatΛ[j][i])*wdiff[j,i] for i in 1:M]
    end

    ff = zeros(N+1,M)
    if order == 2
      for j in edge_indices(mesh)
          @inbounds ul=cellval_at_left(j,u,mesh)
          @inbounds ur=cellval_at_right(j,u,mesh)
        ff[j,:] = Nflux(ul,ur)
      end
    elseif order == 3 || order == 4
      for j in edge_indices(mesh)
        @inbounds ull=cellval_at_left(j-1,u,mesh)
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        @inbounds urr=cellval_at_right(j+1,u,mesh)
        ff[j,:] = 4.0/3.0*Nflux(ul,ur)-1.0/6.0*(Nflux(ull,ur)+Nflux(ul,urr))
      end
    elseif order == 5
      for j in edge_indices(mesh)
        @inbounds ulll=cellval_at_left(j-2,u,mesh)
        @inbounds ull=cellval_at_left(j-1,u,mesh)
        @inbounds ul=cellval_at_left(j,u,mesh)
        @inbounds ur=cellval_at_right(j,u,mesh)
        @inbounds urr=cellval_at_right(j+1,u,mesh)
        @inbounds urrr=cellval_at_right(j+2,u,mesh)
        ff[j,:] = 3.0/2.0*Nflux(ul,ur)-3.0/10.0*(Nflux(ull,ur)+Nflux(ul,urr))+
            1.0/30.0*(Nflux(ulll,ur)+Nflux(ull,urr)+Nflux(ul,urrr))
      end
    end
    hh[:,:] = ff - dd
end
