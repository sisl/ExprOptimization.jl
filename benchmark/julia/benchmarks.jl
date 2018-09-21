module EOBenchmarks

export Quartic, ApproximatePi, Salustowicz2d, Combine

include("quartic.jl")
include("approximate_pi.jl")
include("salustowicz_2d.jl")
include("combine.jl")

function runall()
    Quartic.main()
    ApproximatePi.main()
    Combine.main()
end

end
