@testset "Slope Limiter" begin

using ConservationLawsDiffEq

for limiter in (GeneralizedMinmodLimiter(), MinmodLimiter())
    printstyled("Limiter: ", string(typeof(limiter)) ,"\n"; color = :blue)
    @test limiter(1.0,2.0) == 1.0
    @test limiter(-1.0,2.0) == 0.0
    @test limiter(-1.0,-2.0) == -1.0
end

limiter  = OsherLimiter()
    printstyled("Limiter: ", string(typeof(limiter)) ,"\n"; color = :blue)
    @test limiter(1.0,2.0) == 1.0
    @test limiter(-1.0,2.0) == 0.0
    @test limiter(-1.0,-2.0) == 0.0

limiter = SuperbeeLimiter()
    printstyled("Limiter: ", string(typeof(limiter)) ,"\n"; color = :blue)
    @test limiter(1.0,2.0) == 2.0
    @test limiter(-1.0,2.0) == 0.0
    @test limiter(-1.0,-2.0) == 0.0
end
