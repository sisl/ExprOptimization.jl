
module CrossEntropys

using ExprRules
using ..ProbabilisticExprRules

using ExprOptimization: ExprOptAlgorithm, ExprOptResult
import ExprOptimization: optimize

export CrossEntropy

abstract type InitializationMethod end 

"""
    CrossEntropy

Cross Entropy method.
# Arguments
- `pop_size::Int`: population size
- `iterations::Int`: number of iterations
- `max_depth::Int`: maximum depth of derivation tree
- `top_k::Int`: top k elite samples used in selection
- `p_init::Float64`: initial value when fitting MLE 
- `init_method::InitializationMethod`: Initialization method
"""
struct CrossEntropy <: ExprOptAlgorithm
    pop_size::Int                                       
    iterations::Int                                     
    max_depth::Int                                      
    top_k::Int                                          
    p_init::Float64
    init_method::InitializationMethod

    function CrossEntropy(
        pop_size::Int,                                  #population size
        iterations::Int,                                #number of iterations
        max_depth::Int,                                 #maximum depth of derivation tree
        top_k::Int,                                     #top k elite samples used in selection
        p_init::Float64=0.0,                            #initial value when fitting MLE
        init_method::InitializationMethod=RandomInit()) #initialization method 

        new(pop_size, iterations, max_depth, top_k, p_init, init_method)
    end
end

"""
    RandomInit

Uniformly random initialization method.
"""
struct RandomInit <: InitializationMethod end

"""
    optimize(p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function)

Expression tree optimization using the cross-entropy method with parameters p, grammar 'grammar', and start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar)

See: Rubinstein, "Optimization of Computer Simulation Models with Rare Events", European Journal of Operations Research, 99, 89-112, 1197
"""
optimize(p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function) = cross_entropy(p, grammar, typ, loss)

"""
    cross_entropy(p::CrossEntropy, grammar::Grammar, typ::Symbol)

Expression tree optimization using cross-entropy method with parameters p, grammar 'grammar', and start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar)

See: Rubinstein, "Optimization of Computer Simulation Models with Rare Events", European Journal of Operations Research, 99, 89-112, 1197
"""
function cross_entropy(p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function)
    iseval(grammar) && error("Cross-entropy does not support _() functions in the grammar")

    dmap = mindepth_map(grammar)
    losses = Vector{Float64}(undef,p.pop_size)
    pcfg = ProbabilisticGrammar(grammar)
    pop = initialize(p.init_method, p.pop_size, pcfg, typ, dmap, p.max_depth)
    best_tree, best_loss = evaluate!(loss, grammar, pop, losses, RuleNode(0), Inf)
    for iter = 1:p.iterations 
        for i in eachindex(pop)
            pop[i] = rand(RuleNode, pcfg, typ, dmap, p.max_depth)
        end
        fit_mle!(pcfg, pop[1:p.top_k], p.p_init)
        best_tree, best_loss = evaluate!(loss, grammar, pop, losses, best_tree, best_loss)
    end
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

"""
    initialize(::RandomInit, pop_size::Int, grammar::Grammar, typ::Symbol, dmap::AbstractVector{Int},
        max_depth::Int)

Random population initialization.
"""
function initialize(::RandomInit, pop_size::Int, pcfg::ProbabilisticGrammar, typ::Symbol, 
            dmap::AbstractVector{Int}, max_depth::Int)
    [rand(RuleNode, pcfg, typ, dmap, max_depth) for i = 1:pop_size]
end

"""
    evaluate!(loss::Function, grammar::Grammar, pop::Vector{RuleNode}, losses::Vector{Float64}, 
        best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(loss::Function, grammar::Grammar, pop::Vector{RuleNode}, losses::Vector{Float64}, 
                   best_tree::RuleNode, best_loss::Float64)

    losses[:] = loss.(pop, Ref(grammar))
    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
    end
    (best_tree, best_loss)
end

end #module
