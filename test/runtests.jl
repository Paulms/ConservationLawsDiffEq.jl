using ConservationLawsDiffEq
using Base.Test

@time @testset "1D Scalar Algorithms" begin include("test0.jl") end
@time @testset "1D Sytems Algorithms" begin include("test2.jl") end
#@time @testset "1D Diffusion System Algorithms" begin include("test4.jl") end
