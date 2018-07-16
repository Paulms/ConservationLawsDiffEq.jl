#Lax shock tube
using ConservationLawsDiffEq

#Euler Equations
const CFL = 0.5
const Tend = 0.2
const γ=1.4 #gas constant
const xdiafragm = 0.0

function f(::Type{Val{:jac}},u::AbstractVector)
  ρ = u[1]; v = u[2]/u[1]; ϵ=u[3]
  p = (ϵ-0.5*ρ*v^2)*(γ-1)
  F =[0.0 1.0 0.0;-v^2*(1+γ)/2 v*(3-γ) (γ-1);v^3*(γ-1)+γ*ϵ*v/ρ 3/2*v^2*(1-γ)+γ*ϵ/ρ γ*v]
  F
end

function f(u::AbstractVector)
  ρ = u[1]; v = u[2]/u[1]; ϵ=u[3]
  p = (ϵ-0.5*ρ*v^2)*(γ-1)
  [u[2];u[2]^2/u[1]+p;(ϵ+p)*v]
end

#initial condition
function f0(x::Real)
  ufis_l = [1.0, 0.0,	1.0]
  ufis_r = [0.125 0.0 0.1]
  if x <= xdiafragm
    ρ,u,p = ufis_l
  else
    ρ,u,p = ufis_r
  end
  return [ρ, ρ*u, 1.0/(γ - 1.0)*p + 0.5*ρ*u*u]
end

positive(x) = max(0,x)
#define max wave speed (generic euler system)
function max_w_speed(u)
  ρ = u[:,1];v = u[:,2]./ρ;E = u[:,3]
  p = fill!(similar(ρ), zero(eltype(ρ))); wave_speed = fill!(similar(ρ), zero(eltype(ρ)))
  @. p = (γ-1)*(E-0.5*ρ*v*v)
  @. wave_speed = abs(v) + sqrt(γ*positive(p/ρ))
  return maximum(wave_speed)
end

function Nflux(ul::Vector, ur::Vector)
  ρl = ul[1]; vl = ul[2]/ul[1]; ϵl=ul[3]
  pl = (ϵl-0.5*ρl*vl^2)*(γ-1)
  ρr = ur[1]; vr = ur[2]/ur[1]; ϵr=ur[3]
  pr = (ϵr-0.5*ρr*vr^2)*(γ-1)
  zl = sqrt(ρl/pl)*[1;vl;pl]
  zr = sqrt(ρr/pr)*[1;vr;pr]
  zm = 0.5*(zl+zr)
  zln = (zr-zl)./(log(zr)-log(zl))
  F = fill!(similar(ul), zero(eltype(ul)))
  F[1] = zm[2]*zln[3]
  F[2] = zm[3]/zm[1]+zm[2]/zm[1]*F[1]
  F[2] = 0.5*zm[2]/zm[1]*((γ+1)/(γ-1)*zln[3]/zln[1]+F[2])
  F
end

function ve(u::Vector)
  ρ = u[1]; v = u[2]/u[1]; ϵ=u[3]
  p = (ϵ-0.5*ρ*v^2)*(γ-1)
  s = log(p)-γ*log(ρ)
  return [(γ-s)/(γ-1)-ρ*v^2/(2*p);ρ*v/p;-ρ/p]
end

function get_problem(N)
    mesh = Uniform1DFVMesh(N,-1.0,1.0,:ZERO_FLUX, :ZERO_FLUX)
    ConservationLawsProblem(f0,f,CFL,Tend,mesh)
end
prob = get_problem(20)

basis=legendre_basis(3)
limiter! = DGLimiter(prob, basis, Linear_MUSCL_Limiter())
@time sol1 = solve(prob, DiscontinuousGalerkinScheme(basis, rusanov_euler_num_flux); TimeIntegrator = SSPRK22(limiter!))

@time sol = solve(prob, FVSKTAlgorithm();progress=true)
#@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;ve = ve, order=3);progress=true)

#Plot
using Plots
plot(sol, tidx=1, vars=1, lab="ρ_0",line=(:dot,2))
plot!(sol, vars=1, lab="KT ρ",line = (:dot,2))
plot!(sol1, vars=1, lab="DG K = 3 ρ",line = (:dot,2))
plot!(sol2, vars=1,lab="Tecno ρ",line=(:dot,3))
