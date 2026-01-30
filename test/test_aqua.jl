using Aqua
using Macroeconometrics

@testset "Aqua.jl" begin
    Aqua.test_all(
        Macroeconometrics;
        ambiguities=false,  # Skip ambiguity tests (can have false positives with StatsAPI)
        deps_compat=false,  # Skip deps compat (stdlib packages don't need compat)
    )
end
