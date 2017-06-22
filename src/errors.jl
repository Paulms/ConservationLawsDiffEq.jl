function get_L1_errors{T,N,uType,tType,ProbType}(sol::FVSolution{T,N,uType,tType,
  ProbType,Uniform1DFVMesh}, ref::Function; nvar = 0)
    x = sol.prob.mesh.x
    @unpack tspan = sol.prob
    uexact = ref(x, tspan[end])
    if nvar == 0
      return(1/sol.prob.mesh.N*sum(abs,sol.u[end] - uexact))
    else
      return(1/sol.prob.mesh.N*sum(abs,sol.u[end][:,nvar] - uexact[:,nvar]))
    end
end

function estimate_L1_error(reference, M, uu,N)
  uexact = zeros(N)
  R = Int(round(M/N))
  for i = 1:N
      uexact[i] = 1.0/R*sum(reference[R*(i-1)+1:R*i])
  end
  sum(1.0/N*abs(uu - uexact))
end

function estimate_error_cubic(reference,M, xx,uu,N)
  uexact = zeros(N)
  itp = interpolate(reference[:,2], BSpline(Cubic(Flat())),OnCell())
  i = (M-1)/(reference[M,1]-reference[1,1])*(xx - reference[1,1])+1
  uexact = itp[i]
  sum(1.0/N*abs(uu - uexact))
end
