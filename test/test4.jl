# Vehicular traffic problem
# Test based on example 1 of:
# Bürger, Mulet, Villada, A difussion Corrected Multiclass LWR Traffic model
# with Anticipation Lengths and Reaction times, Advances in Applied Mathematics
# and mechanics, 2013

using ConservationLawsDiffEq
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

function f(::Type{Val{:jac}}, ϕ::AbstractVector)
  M = size(ϕ,1)
  F = fill(zero(eltype(ϕ)),M,M)
  Vϕ = VV(sum(ϕ))
  VPϕ = VP(sum(ϕ))
  for i =  1:M
    for j = 1:M
      F[i,j]=Vmax[i]*(((i==j) ? Vϕ : 0.0) + ϕ[i]*VPϕ)
    end
  end
  F
end

f(ϕ::AbstractVector) = VV(sum(ϕ))*ϕ.*Vmax
β(ϕ::Number) = -VP(ϕ)*L*ϕ/M*sum(Vmax)/M
VV(ϕ::Number) = (ϕ < ϕc) ? 1.0 : 1.0-ϕ
VP(ϕ::Number) = (ϕ < ϕc) ? 0.0 : -1.0

function BB(ϕ::AbstractArray)
  M = size(ϕ,1)
  if (sum(ϕ) < ϕc)
    fill(zero(eltype(ϕ)),M,M)
  else
    B = β(sum(ϕ))*Matrix(I,M,M)
    B
  end
end

f0(x) = 0.5*exp.(-(x-3)^2)*[0.2,0.3,0.2,0.3]

function get_problem(N)
  mesh = Uniform1DFVMesh(N,0.0,10.0,:PERIODIC, :PERIODIC)
  ConservationLawsWithDiffusionProblem(f0,f,BB,CFL,Tend,mesh)
end
#Compile
prob = get_problem(10)
sol = solve(prob, FVSKTAlgorithm();progress=false, save_everystep = false)
#Get numerical reference
prob = get_problem(200)
solref = solve(prob, FVSKTAlgorithm();progress=false, save_everystep = false)
# Compute errors
prob = get_problem(100)
@time sol = fast_solve(prob, FVSKTAlgorithm();progress=false, save_everystep = false, use_threads=false)
@test approx_L1_error(solref, sol) < 0.046
@time sol1 = fast_solve(prob, FVSKTAlgorithm();progress=false, save_everystep = false, use_threads=true)
@test approx_L1_error(solref, sol1) ≈ approx_L1_error(solref, sol)
@time sol = solve(prob, LI_IMEX_RK_Algorithm();progress=false, save_everystep = false, use_threads=false)
@test approx_L1_error(solref, sol) < 0.28 #TODO: Too high
@time sol1 = solve(prob, LI_IMEX_RK_Algorithm();progress=false, save_everystep = false, use_threads=true)
@test approx_L1_error(solref, sol1) ≈ approx_L1_error(solref, sol)
@time sol = solve(prob, FVCUAlgorithm();progress=false, save_everystep = false, use_threads=false)
@test approx_L1_error(solref, sol) < 0.037
@time sol1 = solve(prob, FVCUAlgorithm();progress=false, save_everystep = false, use_threads=true)
@test approx_L1_error(solref, sol1) ≈ approx_L1_error(solref, sol)
@time sol = solve(prob, FVDRCUAlgorithm();progress=false, save_everystep = false, use_threads=false)
@test approx_L1_error(solref, sol) < 0.037
@time sol1 = solve(prob, FVDRCUAlgorithm();progress=false, save_everystep = false, use_threads=true)
@test approx_L1_error(solref, sol1) ≈ approx_L1_error(solref, sol)
@time sol = solve(prob, FVDRCU5Algorithm();progress=false, save_everystep = false, use_threads=false)
@test approx_L1_error(solref, sol) < 0.06
@time sol1 = solve(prob, FVDRCU5Algorithm();progress=false, save_everystep = false, use_threads=true)
@test approx_L1_error(solref, sol1) ≈ approx_L1_error(solref, sol)
@time sol = solve(prob, COMP_GLF_Diff_Algorithm();progress=false, save_everystep = false, use_threads=false)
@test approx_L1_error(solref, sol) < 0.015
@time sol1 = solve(prob, COMP_GLF_Diff_Algorithm();progress=false, save_everystep = false, use_threads=true)
@test approx_L1_error(solref, sol1) ≈ approx_L1_error(solref, sol)
