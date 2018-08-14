@testset "Auxiliar Functions" begin

using ConservationLawsDiffEq
alg = LaxFriedrichsAlgorithm()
@test ConservationLawsDiffEq.scheme_short_name(alg) == "LaxFriedrichsAlgorithm"
end
