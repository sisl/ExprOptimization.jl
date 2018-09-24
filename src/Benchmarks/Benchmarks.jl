module Benchmarks

using ExprOptimization
using Random, CPUTime, DataFrames, CSV, Statistics
using ExprOptimization.GeneticPrograms: RandomInit, TournamentSelection

const RESULTSDIR = joinpath(@__DIR__, "results")
isdir(RESULTSDIR) || mkpath(RESULTSDIR)

#shared functions
protectedDiv(x, y) = iszero(y) ? 1.0 : x / y
protectedSin(x) = isinf(x) ? 0.0 : sin(x)
protectedCos(x) = isinf(x) ? 1.0 : cos(x)

main_koza_1() = main_timing(run_koza_1, 50, [1000], "koza_1", "exproptimization_koza_1.csv")
main_keijzer_9() = main_timing(run_keijzer_9, 50, [1000], "keijzer_9", "exproptimization_keijzer_9.csv")
main_vladislavleva_3() = main_timing(run_vladislavleva_3, 50, [1000], "vladislavleva_3", "exproptimization_vladislavleva_3.csv")
main_approximate_pi() = main_timing(run_approximate_pi, 50, [1000], "approximatE_pi", "exproptimization_approximate_pi.csv")

function log_df()
    df = DataFrame([String,String,Int,Int,Float64,Float64,Float64,Float64], 
                    [:system,:problem,:n_seeds,:n_pop,:mean_time_s,:std_time_s,:mean_fitness,:std_fitness], 0)
end
function main_timing(f::Function, n_seeds, pop_sizes, problem::String, outfile::String)
    f(0,100)  #precompile

    df = log_df()
    for n_pop in pop_sizes
        times = Float64[]
        fitnesses = Float64[]
        for i = 1:n_seeds
            @show n_pop, i
            result, dt = f(i, n_pop)
            push!(times, dt)
            push!(fitnesses, result.loss)
        end
        ts = times .* 1e-6 #convert to seconds
        push!(df, ["ExprOptimization",problem,n_seeds,n_pop,mean(ts),std(ts),mean(fitnesses),std(fitnesses)])
        CSV.write(joinpath(RESULTSDIR,outfile), df)
    end
end

function load_csvs()
    fs = filter(x->endswith(x,".csv"), readdir(RESULTSDIR))
    df = vcat([CSV.read(joinpath(RESULTSDIR,f)) for f in fs]...)
    df
end

function main()
    main_koza_1()
    main_keijzer_9()
    main_vladislavleva_3()
    main_approximate_pi()
end


end #module
