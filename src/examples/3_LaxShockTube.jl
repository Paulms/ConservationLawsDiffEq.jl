#Lax shock tube
using ConservationLawsDiffEq

#Euler Equations
#TODO: Not finished
const CFL = 0.45
const Tend = 1.3
const γ=8.314472 #gas constant

function Jf(u::Vector)
  ρ = u[1]; v = u[2]/u[1]; ϵ=u[3]
  p = (ϵ-0.5*ρ*v^2)*(γ-1)
  F =[0.0 1.0 0.0;-v^2*(1+γ)/2 v*(3-γ) (γ-1);v^3*(γ-1)+γ*ϵ*v/ρ 3/2*v^2*(1-γ)+γ*ϵ/ρ γ*v]
  F
end

function f(u::Vector)
  ρ = u[1]; v = u[2]/u[1]; ϵ=u[3]
  p = (ϵ-0.5*ρ*v^2)*(γ-1)
  [u[2];u[2]^2/u[1]+p;(ϵ+p)*v]
end

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 3)
  for i = 1:N
    if xx[i] < 0.0
      uinit[i,1] = 0.445
      uinit[i,2] = 0.445*0.698
      uinit[i,3] = 3.528/(1-γ)+0.5*0.445*0.698^2
    else
      uinit[i,1] = 0.5
      uinit[i,2] = 0.0
      uinit[i,3] = 0.571/(1-γ)
   end
  end
  return uinit
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
  F = zeros(ul)
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

N = 100
mesh = Uniform1DFVMesh(N,-5.0,5.0,:PERIODIC)
u0 = u0_func(mesh.x)
prob = ConservationLawsProblem(u0,f,CFL,Tend,mesh;Jf=Jf)
@time sol = solve(prob, FVKTAlgorithm();progress=true)
@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;ve = ve, order=3);progress=true)

#Plot
using Plots
plot(mesh.x, sol.u[1][:,1], lab="ho",line=(:dot,2))
plot!(mesh.x, sol.u[end][:,1],lab="KT h")
plot!(mesh.x, sol2.u[end][:,1],lab="Tecno h")
