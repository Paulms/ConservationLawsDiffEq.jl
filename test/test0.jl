# 1D Burgers Equation
# u_t+(0.5*u²)_{x}=0

using ConservationLawsDiffEq
include("burgers.jl")

const CFL = 0.5
const Tend = 1.0
const ul = 0.0
const ur = 1.0
const x0 = 0.0
const xl = -3.0
const xr = 3.0

prob1 = RiemannProblem(Burgers(), ul, ur, x0, 0.0)
sol_ana  = solve(prob1)

f(::Type{Val{:jac}},u::Vector) = diagm(u)
f(u::Vector) = u.^2/2

function u0_func(xx)
  N = size(xx,1)
  uinit = zeros(N, 1)
  for (i,x) in enumerate(xx)
      uinit[i,1] = (x < x0) ? ul : ur
  end
  return uinit
end

function get_problem(N)
  mesh = Uniform1DFVMesh(N,xl,xr,:PERIODIC, :PERIODIC)
  u0 = u0_func(cell_centers(mesh))
  ConservationLawsProblem(u0,f,CFL,Tend,mesh)
end
prob = get_problem(50)
@time sol = solve(prob, FVSKTAlgorithm();progress=true, use_threads = true, save_everystep = false)
@time sol1 = fast_solve(prob, FVSKTAlgorithm();progress=true)
@time sol2 = solve(prob, LaxFriedrichsAlgorithm();progress=true, save_everystep = false)
@time sol3 = solve(prob, LocalLaxFriedrichsAlgorithm();progress=true, save_everystep = false)
@time sol4 = solve(prob, GlobalLaxFriedrichsAlgorithm();progress=true, save_everystep = false)
#@time sol5 = solve(prob, LaxWendroff2sAlgorithm();progress=true, save_everystep = false)
@time sol5 = solve(prob, FVCompWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())
@time sol6 = solve(prob, FVCompMWENOAlgorithm();progress=true, TimeAlgorithm = SSPRK33())

#Compute approximate errors at tend
function calc_error(uana, unum::AbstractFVSolution, tend, xl, xr)
    N = numcells(unum.prob.mesh)
    xk = cell_centers(unum.prob.mesh)
    uexact = zeros(N)
    for (i,x) = enumerate(xk)
        uexact[i] = uana(tend, x)
    end
    sum(1.0/N*abs.(unum.u[end] - uexact))
end
@test calc_error(sol_ana, sol, Tend, -2.0, 2.0) < 0.095
@test calc_error(sol_ana, sol1, Tend, -2.0, 2.0) < 0.095
@test calc_error(sol_ana, sol2, Tend, -2.0, 2.0) < 0.14
@test calc_error(sol_ana, sol3, Tend, -2.0, 2.0) < 0.13
@test calc_error(sol_ana, sol4, Tend, -2.0, 2.0) < 0.14
@test calc_error(sol_ana, sol5, Tend, -2.0, 2.0) < 0.095
@test calc_error(sol_ana, sol6, Tend, -2.0, 2.0) < 0.095
