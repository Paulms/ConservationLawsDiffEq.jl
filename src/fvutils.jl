function num_integrate(f,a,b;order=5, method = FastGaussQuadrature.gausslegendre)
    nodes, weights = method(order);
    t_nodes = 0.5*(b-a)*nodes .+ 0.5*(b+a)
    M = length(f(a))
    tmp = fill(0.0,M)
    for i in 1:M
        g(x) = f(x)[i]
        tmp[i] = 0.5*(b-a)*dot(g.(t_nodes),weights)
    end
    return tmp
end
