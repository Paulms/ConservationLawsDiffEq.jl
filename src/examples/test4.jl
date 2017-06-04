# Vehicular traffic problem
# Test based on example 1 of:
# Bürger, Mulet, Villada, A difussion Corrected Multiclass LWR Traffic model
# with Anticipation Lengths and Reaction times, Advances in Applied Mathematics
# and mechanics, 2013

using ConservationLawsDiffEq

# Parameters:
const CFL = 0.25
const Tend = 0.1
const ϕc = exp(-7/e)
const M = 4
const Vmax = [60.0,55.0,50.0,45.0]
const CC = e/7
const κ = 1e-6
const L = 0.03

function Jf(ϕ::Vector)
  M = size(ϕ,1)
  F = zeros(M,M)
  Vϕ = VV(sum(ϕ))
  VPϕ = VP(sum(ϕ))
  for i =  1:M
    for j = 1:M
      F[i,j]=Vmax[i]*(((i==j)? Vϕ:0.0) + ϕ[i]*VPϕ)
    end
  end
  F
end

f(ϕ::Vector) = VV(sum(ϕ))*ϕ.*Vmax
β(ϕ::Number) = -VP(ϕ)*L*ϕ/M*mean(Vmax)
VV(ϕ::Number) = (ϕ < ϕc) ? 1.0 : 1.0-ϕ
VP(ϕ::Number) = (ϕ < ϕc) ? 0.0 : -1.0

function BB(ϕ::AbstractArray)
  if (sum(ϕ) < ϕc)
    0.0
  else
    M = size(ϕ,1)
    B = β(sum(ϕ))*eye(M)
    B
  end
end

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, M)
  for (i,x) in enumerate(xx)
      uinit[i,:] = 0.5*exp.(-(x-3)^2)*[0.2,0.3,0.2,0.3]
  end
  return uinit
end

N = 100
mesh = Uniform1DFVMesh(N,0.0,10.0,:PERIODIC)
u0 = u0_func(mesh.x)
prob = ConservationLawsWithDiffusionProblem(u0,f,BB,CFL,Tend,mesh;Jf=Jf)
@time sol = solve(prob, FVKTAlgorithm();progress=true,saveat=0.01)
@time sol2 = solve(prob, LI_IMEX_RK_Algorithm();progress=true,saveat=0.01)

#Plot
using(Plots)
#Plots.scalefontsizes(1.5)
plot(sol.prob.mesh.x, sol.u[1], line=(:dot,2), ylab="u", xlab = "x")
plot!(sol.prob.mesh.x, [sum(sol.u[1][i,:]) for i=1:sol.prob.mesh.N],lab="u")
plot(sol.prob.mesh.x, sol.u[end], line=(:dot,2), ylab="u", xlab = "x")
plot!(sol.prob.mesh.x, [sum(sol.u[end][i,:]) for i=1:sol.prob.mesh.N],line=(2),lab="u KT")
#savefig("T4KTN500T02.png")

plot(sol2.prob.mesh.x, sol2.u[1], line=(:dot,2), ylab="u", xlab = "x")
plot(sol2.prob.mesh.x, sol2.u[end], line=(:dot,2), ylab="u", xlab = "x")
plot!(sol2.prob.mesh.x, [sum(sol2.u[end][i,:]) for i=1:N],lab="u IMEX")
