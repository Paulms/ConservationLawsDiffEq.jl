#Compute approximate errors at tend with analytical solution
function get_L1_errors(ref::Function, sol::AbstractFVSolution)
    x = cell_centers(sol.prob.mesh)
    tspan = sol.prob.tspan
    uexact = zeros(sol.u[end])
    for i in 1:numcells(sol.prob.mesh)
        uexact[i,:] = ref(x[i], tspan[end])
    end
    sum(abs,sol.u[end] - uexact)*sol.prob.mesh.Δx
end

function get_relative_L1_error(ref::Function, sol::AbstractFVSolution)
    x = cell_centers(sol.prob.mesh)
    tspan = sol.prob.tspan
    uexact = zeros(sol.u[end])
    for i in 1:numcells(sol.prob.mesh)
        uexact[i,:] = ref(x[i], tspan[end])
    end
    UrefL1norm = sum(abs, uexact)*sol.prob.mesh.Δx
    100.0*sum(abs,sol.u[end] - uexact)*sol.prob.mesh.Δx/UrefL1norm
end

"Compute aproximate L1 errors with numerical reference solution"
function get_L1_error_num(reference,M, uu,N,dx)
  uexact = zeros(uu)
  R = Int(round(M/N))
  for i = 1:N
      uexact[i,:] = 1.0/R*sum(reference[R*(i-1)+1:R*i,:],1)
  end
  sum(dx*abs.(uu - uexact))
end

function approx_L1_error(sol_ref::AbstractFVSolution, sol::AbstractFVSolution)
    M = numcells(sol_ref.prob.mesh)
    N = numcells(sol.prob.mesh)
    return get_L1_error_num(sol_ref.u[end],M, sol.u[end],N,sol.prob.mesh.Δx)
end

function estimate_error_cubic(reference,M, xx,uu,N)
  uexact = zeros(N)
  itp = interpolate(reference[:,2], BSpline(Cubic(Flat())),OnCell())
  i = (M-1)/(reference[M,1]-reference[1,1])*(xx - reference[1,1])+1
  uexact = itp[i]
  sum(1.0/N*abs(uu - uexact))
end
