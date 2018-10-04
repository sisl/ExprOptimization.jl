"""
Benchmarks for ExprOptimization.jl
This will run the julia benchmarks:
using ExprOptimization
ExprOptimization.Benchmarks.main()
See subfolders for the benchmarks of baseline packages
Results are placed in RESULTSDIR
Call load_csvs() to load up a merged dataframe.
See Julia notebook for plots.
"""
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

include("koza.jl")
include("keijzer.jl")
include("vladislavleva.jl")
include("approximate_pi.jl")

main_koza_1() = main_timing(run_koza_1, 50, [1000], "koza_1", "exproptimization_koza_1.csv")
main_keijzer_9() = main_timing(run_keijzer_9, 50, [1000], "keijzer_9", "exproptimization_keijzer_9.csv")
main_keijzer_11() = main_timing(run_keijzer_11, 50, [500], "keijzer_11", "exproptimization_keijzer_11.csv")
main_vladislavleva_3() = main_timing(run_vladislavleva_3, 50, [500], "vladislavleva_3", "exproptimization_vladislavleva_3.csv")
main_vladislavleva_6() = main_timing(run_vladislavleva_6, 50, [500], "vladislavleva_6", "exproptimization_vladislavleva_6.csv")
main_approximate_pi() = main_timing(run_approximate_pi, 50, [1000], "approximate_pi", "exproptimization_approximate_pi.csv")

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
    fs = map(f->joinpath(RESULTSDIR,f), fs)
    fs = filter(f->filesize(f)>0, fs)   #ignore empty files
    df = vcat([CSV.read(f) for f in fs]...)
    df
end

function main_deap()
    dir = pwd()
    cd(joinpath(@__DIR__, "deap"))
    success(`./main.sh`)
    cd(dir)
end

function main_julia()
    main_koza_1()
    main_keijzer_9()
    main_vladislavleva_3()
    main_vladislavleva_6()
    main_keijzer_11()
    main_approximate_pi()
end

function main()
    main_julia()
    main_deap()
end


end #module
