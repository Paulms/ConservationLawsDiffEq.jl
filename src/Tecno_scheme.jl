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

# Numerical Fluxes
#   1   2   3          N-1  N
# |---|---|---|......|---|---|
# 1   2   3   4 ... N-1  N  N+1
@def tecno_order_header begin
  ngc = 0
  if order == 2
    ngc = 1
  elseif order == 3 || order == 4
    ngc = 2
  elseif order == 5
    ngc = 3
  else
    throw("k=$order not available k ∈ {2,3,4,5}")
  end
end
@def tecno_rhs_header begin
  #Eno Reconstrucion
  RΛ1 = eigfact(Flux(Val{:jac},0.5*(uu[0,:]+uu[1,:])))
  MatR = Vector{typeof(RΛ1.vectors)}(0)
  MatΛ = Vector{typeof(RΛ1.values)}(0)
  push!(MatR,RΛ1.vectors)
  push!(MatΛ,RΛ1.values)
  for j = 2:(N+1)
    RΛj = eigfact(Flux(Val{:jac},0.5*(uu[j-1,:]+uu[j,:])))
    push!(MatR,RΛj.vectors); push!(MatΛ,RΛj.values)
  end
  dd = zeros(N+1,M) #Extra numerical diffusion
  k = order - 1
  weights = unif_crj(order)
  v = zeros(uu)
  for j = indices(v, 1)
    v[j,1:M] = ve(uu[j,:]) #entropy variables
  end
  vminus = zeros(N,M)
  vplus = zeros(N,M)
  for j = 1:N
    for i = 1:M
      vminus[j,i],vplus[j,i] = ENO_urec(dx,v[j-k:j+k,i],order,weights)
    end
  end
  wminus = zeros(N,M)
  wplus = zeros(N,M)
  for j = 1:N
    wminus[j,:] = MatR[j]'*vminus[j,:]
    wplus[j,:] = MatR[j+1]'*vplus[j,:]
  end
  wdiff = zeros(N+1,M)
  for j = 2:N
    wdiff[j,:] = wminus[j,:] - wplus[j-1,:]
  end
  if bdtype == :PERIODIC
    wdiff[1,:] = wminus[1,:] - wplus[N,:]
    wdiff[N+1,:] = wdiff[1,:]
  end

  for j = 1:(N+1)
    dd[j,:] = MatR[j]*[abs(MatΛ[j][i])*wdiff[j,i] for i in 1:M]
  end

  ff = zeros(N+1,M)
  if order == 2
    for j = 1:(N+1)
      ff[j,:] = Nflux(uu[j-1,:],uu[j,:])
    end
  elseif order == 3 || order == 4
    for j = 1:(N+1)
      ff[j,:] = 4.0/3.0*Nflux(uu[j-1,:],uu[j,:])-1.0/6.0*(Nflux(uu[j-2,:],uu[j,:])+Nflux(uu[j-1,:],uu[j+1,:]))
    end
  elseif order == 5
    ff[j,:] = 3.0/2.0*Nflux(uu[j-1,:],uu[j,:])-3.0/10.0*(Nflux(uu[j-2,:],uu[j,:])+Nflux(uu[j-1,:],uu[j+1,:]))+
    1.0/30.0*(Nflux(uu[j-3,:],uu[j,:])+Nflux(uu[j-2,:],uu[j+1,:])+Nflux(uu[j-1,:],uu[j+2,:]))
  end

  hh = zeros(N+1,M)
  hh = ff - dd
end

function FV_solve{tType,uType,tAlgType,F}(integrator::FVIntegrator{FVTecnoAlgorithm,
  Uniform1DFVMesh,tType,uType,tAlgType,F};kwargs...)
  @fv_deterministicpreamble
  @fv_uniform1Dmeshpreamble
  @fv_generalpreamble
  @unpack order,Nflux,ve = integrator.alg
  update_dt = cdt
  function rhs!(rhs, uold, N, M, dx, dt, bdtype)
    #SEt ghost Cells
    @tecno_order_header
    @boundary_header
    @tecno_rhs_header
    # Diffusion
    pp = zeros(N+1,M)
    @boundary_update
    @update_rhs
  end
  @fv_timeloop
end
