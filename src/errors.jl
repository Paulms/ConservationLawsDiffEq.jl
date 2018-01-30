function get_L1_errors(sol::FVSolution, ref::Function; nvar = 0)
    x = cell_centers(sol.prob.mesh)
    @unpack tspan = sol.prob
    uexact = ref(x, tspan[end])
    N = numcells(sol.prob.mesh)
    if nvar == 0
      return(1/N*sum(abs,sol.u[end] - uexact))
    else
      return(1/N*sum(abs,sol.u[end][:,nvar] - uexact[:,nvar]))
    end
end

function approx_L1_error(uref, uu; nvar = 1)
  Nr = numcells(uref.prob.mesh)
  Ns = numcells(uu.prob.mesh)
  Ms = size(uu.u[end],2)
  R = Int(round(Nr/Ns))
  uexact = zeros(Ns,Ms)
  for i = 1:Ns
      uexact[i,:] = 1.0/R*sum(uref.u[end][R*(i-1)+1:R*i, :],1)
  end
  1.0/Ns*sum(abs,uu.u[end] - uexact)
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

#Compute approximate errors at tend for 1D 1 Variable problems
function get_L1_errors(uana, unum::AbstractFVSolution, tend, xl, xr)
    N = numcells(unum.prob.mesh)
    xk = cell_centers(unum.prob.mesh)
    uexact = zeros(N)
    for (i,x) = enumerate(xk)
        uexact[i] = uana(tend, x)
    end
    1.0/N*sum(abs,(unum.u[end] - uexact))
end
