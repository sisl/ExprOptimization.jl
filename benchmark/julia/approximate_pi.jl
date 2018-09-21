module ApproximatePi

using ExprOptimization
using Random, Statistics, BenchmarkTools, CPUTime, Statistics, DataFrames, CSV
using ExprOptimization.GeneticPrograms: RandomInit, TournamentSelection

const grammar = @grammar begin
    R = |(1:9)
    R = R + R
    R = R - R
    R = R * R
    R = R / R
end

#quartic
function loss(tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    value = Core.eval(S, ex)
    if isinf(value) || isnan(value)
        return Inf
    end
    Δ = abs(value - π)
    return log(Δ) + length(tree) / 1e4
end

const S = SymbolTable(grammar, ApproximatePi)

function runonce(seed::Int=0, n_pop::Int=500)
    Random.seed!(seed)

    init_method = RandomInit()
    select_method = TournamentSelection(3)
    p = GeneticProgram(n_pop, #pop_size
                       50,  #iterations
                       15,  #max_depth
                       0.4, #p_reproduction
                       0.5, #p_crossover
                       0.1, #p_mutation 
                       init_method=init_method, 
                       select_method=select_method)
    result = optimize(p, grammar, :R, loss)
    result
end

function main()
    runonce(0,100)  #precompile

    df = DataFrame([String,String,Int,Int,Float64,Float64,Float64,Float64], 
                    [:system,:problem,:n_seeds,:n_pop,:mean_time_s,:std_time_s,:mean_fitness,:std_fitness], 0)

    n_seeds = 50
    pop_sizes = [1000]
    for n_pop in pop_sizes
        times = Float64[]
        fitnesses = Float64[]
        for i = 1:n_seeds
            tstart = CPUtime_us()
            result = runonce(i, n_pop)
            push!(times, CPUtime_us() - tstart)
            push!(fitnesses, result.loss)
        end
        ts = times .* 1e-6 #convert to seconds
        push!(df, ["ExprOptimization","approximate_pi",n_seeds,n_pop,mean(ts),std(ts),mean(fitnesses),std(fitnesses)])
        CSV.write("../ExprOptimization_approximate_pi.csv", df)
    end
end

end

