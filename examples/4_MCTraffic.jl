# Vehicular traffic problem
# Test based on example 1 of:
# Bürger, Mulet, Villada, A difussion Corrected Multiclass LWR Traffic model
# with Anticipation Lengths and Reaction times, Advances in Applied Mathematics
# and mechanics, 2013

using ConservationLawsDiffEq
using Statistics
using LinearAlgebra

# Parameters:
const CFL = 0.25
const Tend = 0.1
const ϕc = exp(-7/ℯ)
const M = 4
const Vmax = [60.0,55.0,50.0,45.0]
const CC = ℯ/7
const κ = 1e-6
const L = 0.03

function Jf(ϕ::AbstractVector)
  M = size(ϕ,1)
  F = fill(zero(eltype(ϕ)),M,M)
  Vϕ = VV(sum(ϕ))
  VPϕ = VP(sum(ϕ))
  for j = 1:M, i = 1:M
      F[i,j]=Vmax[i]*(((i==j) ? Vϕ : 0.0) + ϕ[i]*VPϕ)
  end
  F
end

f(ϕ::AbstractVector) = VV(sum(ϕ))*ϕ.*Vmax
β(ϕ::Number) = -VP(ϕ)*L*ϕ/M*mean(Vmax)
VV(ϕ::T) where {T<:Number} = (ϕ < ϕc) ? one(T) : one(T)-ϕ
VP(ϕ::T) where {T<:Number} = (ϕ < ϕc) ? zero(T) : -one(T)

function BB(ϕ::AbstractArray)
  M = size(ϕ,1)
  if (sum(ϕ) < ϕc)
    fill(zero(eltype(ϕ)),M,M)
  else
    β(sum(ϕ))*Diagonal(ones(M))
  end
end

f0(x) = 0.5*exp.(-(x-3)^2)*[0.2,0.3,0.2,0.3]

function get_problem(N)
  mesh = Uniform1DFVMesh(N,0.0,10.0,:PERIODIC, :PERIODIC)
  ConservationLawsWithDiffusionProblem(f0,f,BB,CFL,Tend,mesh;jac = Jf)
end

#Run
prob = get_problem(100)
@time sol = solve(prob, FVSKTAlgorithm();progress=true, save_everystep = false)
@time sol2 = solve(prob, LI_IMEX_RK_Algorithm();progress=true, save_everystep = false)
@time sol3 = solve(prob, FVCUAlgorithm();progress=true, save_everystep = false)
@time sol4 = solve(prob, FVDRCUAlgorithm();progress=true, save_everystep = false)

#Plot
using(Plots)
pyplot()
plot(sol, tidx=1, line=(:dot,2), ylab="u", xlab = "x")
plot!(cell_centers(sol.prob.mesh), [sum(sol.u[1][i,:]) for i=1:numcells(sol.prob.mesh)],lab="u")
plot(sol, line=(:dot,2), ylab="u", xlab = "x")
plot!(cell_centers(sol.prob.mesh), [sum(sol.u[end][i,:]) for i=1:numcells(sol.prob.mesh)],line=(2),lab="u KT")

plot(sol2, line=(:dot,2), ylab="u", xlab = "x")
plot!(cell_centers(sol2.prob.mesh), [sum(sol2.u[end][i,:]) for i=1:numcells(sol2.prob.mesh)],lab="u IMEX")

plot(sol3, line=(:dot,2), ylab="u", xlab = "x")
plot!(cell_centers(sol3.prob.mesh), [sum(sol3.u[end][i,:]) for i=1:numcells(sol3.prob.mesh)],lab="u CU")

plot(sol4, line=(:dot,2), ylab="u", xlab = "x")
plot!(cell_centers(sol4.prob.mesh), [sum(sol4.u[end][i,:]) for i=1:numcells(sol4.prob.mesh)],lab="u DRCU")
