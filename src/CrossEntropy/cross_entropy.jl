
module CrossEntropy

using ExprRules, ProbabilisticExprRules
using ..loss
using ..ExprOptParams
using ..ExprOptResults

import ..optimize

export CrossEntropyParams

abstract type InitializationMethod end 

"""
    CrossEntropyParams

Parameters for Cross Entropy method.
# Arguments
- `pop_size::Int`: population size
- `iiterations::Int`: number of iterations
- `max_depth::Int`: maximum depth of derivation tree
- `top_k::Int`: top k elite samples used in selection
- `init_method::InitializationMethod`: Initialization method
"""
struct CrossEntropyParams <: ExprOptParams
    pop_size::Int                                       
    iterations::Int                                     
    max_depth::Int                                      
    top_k::Int                                          
    init_method::InitializationMethod

    function CrossEntropyParams(
        pop_size::Int,                                  #population size
        iterations::Int,                                #number of iterations
        max_depth::Int,                                 #maximum depth of derivation tree
        top_k::Int,                                     #top k elite samples used in selection
        init_method::InitializationMethod=RandomInit()) #initialization method 

        new(pop_size, iterations, max_depth, top_k, init_method)
    end
end

"""
    RandomInit

Uniformly random initialization method.
"""
struct RandomInit <: InitializationMethod end

"""
    optimize(p::CrossEntropyParams, grammar::Grammar, typ::Symbol)

Expression tree optimization using the cross-entropy method with parameters p, probabilistic grammar 'grammar', and start symbol typ.
"""
optimize(p::CrossEntropyParams, grammar::Grammar, typ::Symbol) = cross_entropy(p, grammar, typ)

"""
    cross_entropy(p::CrossEntropyParams, grammar::Grammar, typ::Symbol)

Expression tree optimization using cross-entropy method with parameters p, probabilistic grammar 'grammar', and start symbol typ.
"""
function cross_entropy(p::CrossEntropyParams, grammar::Grammar, typ::Symbol)
    iseval(grammar) && error("Cross-entropy does not support _() functions in the grammar")

    losses = Vector{Float64}(p.pop_size)

    pcfg = ProbabilisticGrammar(grammar)
    pop = initialize(p.init_method, p.pop_size, pcfg, typ, p.max_depth)
    best_tree, best_loss = evaluate!(pop, losses, RuleNode(0), Inf)
    for iter = 1:p.iterations 
        for i in eachindex(pop)
            pop[i] = rand(RuleNode, pcfg, typ, p.max_depth)
        end
        fit_mle!(pcfg, pop[1:p.top_k])
        best_tree, best_loss = evaluate!(pop, losses, best_tree, best_loss)
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

"""
    initialize(::RandomInit, pop_size::Int, grammar::Grammar, typ::Symbol, max_depth::Int)

Random population initialization.
"""
initialize(::RandomInit, pop_size::Int, pcfg::ProbabilisticGrammar, typ::Symbol, max_depth::Int) = 
    [rand(RuleNode, pcfg, typ, max_depth) for i = 1:pop_size]

"""
    evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, 
    best_loss::Float64)

    losses[:] = loss.(pop)
    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
    end
    (best_tree, best_loss)
end

end #module
