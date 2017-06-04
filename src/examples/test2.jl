# One dimensional wave equation
using ConservationLawsDiffEq

const CFL = 0.45
const Tend = 1.0
const cc = 1.0

function Jf(u::Vector)
  F =[0.0 cc;cc 0.0]
  F
end

f(u::Vector) = [0.0 cc;cc 0.0]*u

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 2)
  uinit[:,1] = sin(4*π*xx)
  return uinit
end

Nflux(ϕl::Vector, ϕr::Vector) = 0.5*(f(ϕl)+f(ϕr))
exact_sol(x::Vector, t::Float64) = hcat(0.5*(sin(4*π*(-t+x))+sin(4*π*(t+x))),
0.5*(sin(4*π*(-t+x))-sin(4*π*(t+x))))

N = 500
mesh = Uniform1DFVMesh(N,-1.0,1.0,:PERIODIC)
u0 = u0_func(mesh.x)
prob = ConservationLawsProblem(u0,f,CFL,Tend,mesh;Jf=Jf)
@time sol = solve(prob, FVKTAlgorithm();progress=true)
@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;order=3);progress=true)
@time sol3 = solve(prob, FVCompWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol4 = solve(prob, FVCompMWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol5 = solve(prob, FVSpecMWENOAlgorithm();progress=true, TimeIntegrator = SSPRK33())

#writedlm("test_2_ktreference.txt", [mesh.x sol.u[end]], '\t')
#writedlm("test_2_Tecnoreference.txt", [mesh.x sol2.u[end]], '\t')
#reference = readdlm("test_2_ktreference.txt");

get_L1_errors(sol, exact_sol; nvar = 1) #43.3
get_L1_errors(sol2, exact_sol; nvar = 1) #0.0790
get_L1_errors(sol3, exact_sol; nvar = 1) #0.00545
get_L1_errors(sol4, exact_sol; nvar = 1) #0.00558
get_L1_errors(sol5, exact_sol; nvar = 1) #0.00558
#Plot
using Plots
plot(sol, tidx=1, uvars=1, lab="ho",line=(:dot,2))
plot!(sol, uvars=1, lab="KT h",line = (:dot,2))
plot!(sol2, uvars=1,lab="Tecno h",line=(:dot,3))
plot!(sol3, uvars=1,lab="Comp WENO h",line=(:dot,3))
plot!(sol4, uvars=1,lab="Comp MWENO h",line=(:dot,3))
plot!(sol5, uvars=1,lab="Spec MWENO h",line=(:dot,3))
plot!(mesh.x, exact_sol(mesh.x,Tend)[:,1],lab="Ref")
