module Quartic

using ExprOptimization
using Random, Statistics, BenchmarkTools
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

function main(seed::Int=0)
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

end

