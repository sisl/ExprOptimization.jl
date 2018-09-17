module Quartic

using ExprOptimization
using Random, Statistics, BenchmarkTools, CPUTime, Statistics
using ExprOptimization.GeneticPrograms: RandomInit, TournamentSelection

const grammar = @grammar begin
    R = x
    R = R + R
    R = R - R
    R = R * R
    R = protectedDiv(R, R)
    R = -R
    R = cos(R)
    R = sin(R)
    R = |(1.0:5.0) 
end
protectedDiv(x, y) = iszero(y) ? 1.0 : x / y

#quartic
gt(x) = x^4 + x^3 + x^2 + x
function loss(tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    for x = -1.0:0.1:1.0
        S[:x] = x 
        los += abs2(Core.eval(S,ex) - gt(x))
    end
    mean(los)
end

const S = SymbolTable(grammar, Quartic)

function runonce(seed::Int=0)
    Random.seed!(seed)

    init_method = RandomInit()
    select_method = TournamentSelection(3)
    p = GeneticProgram(300, #pop_size
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
    runonce()  #precompile
    times = Float64[]
    for i = 1:20
        tstart = CPUtime_us()
        runonce(i)
        push!(times, CPUtime_us() - tstart)
    end
    ts = times .* 1e-6 #convert to seconds
    mean(ts), std(ts), ts
end

end

