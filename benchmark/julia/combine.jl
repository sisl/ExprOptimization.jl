module Combine

using DataFrames, CSV, PGFPlots

function load_csvs(dir::AbstractString="../")
    fs = filter(x->endswith(x,".csv"), readdir(dir))
    df = vcat([CSV.read(joinpath(dir,f)) for f in fs]...)
    df
end

function main()
    df = load_csvs()
end

end
