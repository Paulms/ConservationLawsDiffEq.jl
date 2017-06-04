@recipe function f(sol::AbstractFVSolution; tidx = size(sol.t,1), uvars=0)
    seriestype  :=  :path
    xguide --> "x"
    yguide --> "u"
    labels = String[]
    for i in 1:size(sol.u,2)
      push!(labels,"u$i")
    end
    label --> reshape(labels,1,length(labels))
    yvector = sol.u[tidx]
    if uvars != 0
      yvector = sol.u[tidx][:,uvars]
    end
    sol.prob.mesh.x, yvector
end
