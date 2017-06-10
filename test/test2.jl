# One dimensional wave equation
using ConservationLawsDiffEq

const CFL = 0.45
const Tend = 1.0
const cc = 1.0

f(::Type{Val{:jac}},u::Vector) = [0.0 cc;cc 0.0]
f(u::Vector) = [0.0 cc;cc 0.0]*u

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 2)
  uinit[:,1] = sin.(4*π*xx)
  return uinit
end

Nflux(ϕl::Vector, ϕr::Vector) = 0.5*(f(ϕl)+f(ϕr))
exact_sol(x::Vector, t::Float64) = hcat(0.5*(sin.(4*π*(-t+x))+sin.(4*π*(t+x))),
0.5*(sin.(4*π*(-t+x))-sin.(4*π*(t+x))))

function get_problem(N)
  mesh = Uniform1DFVMesh(N,-1.0,1.0,:PERIODIC)
  u0 = u0_func(mesh.x)
  ConservationLawsProblem(u0,f,CFL,Tend,mesh)
end
prob = get_problem(10)

@time sol = solve(prob, FVKTAlgorithm();progress=true)
@time sol2 = solve(prob, FVTecnoAlgorithm(Nflux;order=3);progress=true)
@time sol3 = solve(prob, FVCompWENOAlgorithm();progress=true)
@time sol4 = solve(prob, FVCompMWENOAlgorithm();progress=true)
@time sol5 = solve(prob, FVSpecMWENOAlgorithm();progress=true)
true
