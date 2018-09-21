module Salustowicz2d

using ExprOptimization
using Random, Statistics, BenchmarkTools, CPUTime, Statistics, DataFrames, CSV
using ExprOptimization.GeneticPrograms: RandomInit, TournamentSelection

const grammar = @grammar begin
    R = x | y
    R = R + R
    R = R - R
    R = R * R
    R = protectedDiv(R, R)
    R = -R
    R = cos(R)
    R = sin(R)
    R = _(7.0*rand()) #ephemeral
end
protectedDiv(x, y) = iszero(y) ? 1.0 : x / y

#salustowicz 2d
gt(x,y) = exp(-x)*x^3*cos(x)*sin(x)*(cos(x)*sin(x)^2-1)*(y-5) 
function loss(tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    n = 0 
    for x = 0.0:0.5:7.0, y = 0.0:0.5:7.0
        S[:x] = x 
        S[:y] = y 
        los += abs2(Core.eval(S,ex) - gt(x,y))
        n += 1
    end
    los / n 
end
const S = SymbolTable(grammar, Salustowicz2d)

function runonce(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    init_method = RandomInit()
    select_method = TournamentSelection(3)
    p = GeneticProgram(n_pop, #pop_size
                       40,  #iterations
                       10,  #max_depth
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
    pop_sizes = [500]
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
        push!(df, ["ExprOptimization","salustowicz_2d",n_seeds,n_pop,mean(ts),std(ts),mean(fitnesses),std(fitnesses)])
        CSV.write("../ExprOptimization_salustowicz_2d.csv", df)
    end
end

end

