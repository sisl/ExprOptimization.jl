
module CrossEntropys

using ExprRules
using ..ProbabilisticExprRules

using ExprOptimization: ExprOptAlgorithm, ExprOptResult, BoundedPriorityQueue, enqueue!
import ExprOptimization: optimize

export CrossEntropy

abstract type InitializationMethod end 
abstract type TrackingMethod end

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
- `track_method::TrackingMethod`: additional tracking, e.g., track top k exprs (default: no additional tracking) 
"""
struct CrossEntropy <: ExprOptAlgorithm
    pop_size::Int                                       
    iterations::Int                                     
    max_depth::Int                                      
    top_k::Int                                          
    p_init::Float64
    init_method::InitializationMethod
    track_method::TrackingMethod

    function CrossEntropy(
        pop_size::Int,                                  #population size
        iterations::Int,                                #number of iterations
        max_depth::Int,                                 #maximum depth of derivation tree
        top_k::Int,                                     #top k elite samples used in selection
        p_init::Float64=0.0,                            #initial value when fitting MLE
        init_method::InitializationMethod=RandomInit(); #initialization method 
        track_method::TrackingMethod=NoTracking())   #tracking method 

        new(pop_size, iterations, max_depth, top_k, p_init, init_method, track_method)
    end
end

"""
    RandomInit

Uniformly random initialization method.
"""
struct RandomInit <: InitializationMethod end

"""
    NoTracking

No additional tracking of expressions.
"""
struct NoTracking <: TrackingMethod end

"""
    TopKTracking

Track the top k expressions.
"""
struct TopKTracking <: TrackingMethod 
    k::Int
    q::BoundedPriorityQueue{RuleNode,Float64}

    function TopKTracking(k::Int)
        q = BoundedPriorityQueue{RuleNode,Float64}(k,Base.Order.Reverse) #lower is better
        obj = new(k, q)
        obj
    end
end

"""
    optimize(p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function; kwargs...)

Expression tree optimization using the cross-entropy method with parameters p, grammar 'grammar', and start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar)

See: Rubinstein, "Optimization of Computer Simulation Models with Rare Events", European Journal of Operations Research, 99, 89-112, 1197
"""
function optimize(p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function; kwargs...) 
    cross_entropy(p, grammar, typ, loss; kwargs...)
end

"""
    cross_entropy(p::CrossEntropy, grammar::Grammar, typ::Symbol)

Expression tree optimization using cross-entropy method with parameters p, grammar 'grammar', and start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar)

See: Rubinstein, "Optimization of Computer Simulation Models with Rare Events", European Journal of Operations Research, 99, 89-112, 1197
"""
function cross_entropy(p::CrossEntropy, grammar::Grammar, typ::Symbol, loss::Function;
    verbose::Bool=false)
    iseval(grammar) && error("Cross-entropy does not support _() functions in the grammar")

    dmap = mindepth_map(grammar)
    losses = Vector{Float64}(undef,p.pop_size)
    pcfg = ProbabilisticGrammar(grammar)
    pop = initialize(p.init_method, p.pop_size, pcfg, typ, dmap, p.max_depth)
    best_tree, best_loss = evaluate!(p, loss, grammar, pop, losses, RuleNode(0), Inf)
    for iter = 1:p.iterations 
        verbose && println("iterations: $iter of $(p.iterations)")
        fit_mle!(pcfg, pop[1:p.top_k], p.p_init)
        for i in eachindex(pop)
            pop[i] = rand(RuleNode, pcfg, typ, dmap, p.max_depth)
        end
        best_tree, best_loss = evaluate!(p, loss, grammar, pop, losses, best_tree, best_loss)
    end
    alg_result = Dict{Symbol,Any}()
    _add_result!(alg_result, p.track_method)
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), alg_result)
end

"""
    _add_result!(d::Dict{Symbol,Any}, t::NoTracking)

Add tracking results to alg_result.  No op for NoTracking.
"""
_add_result!(d::Dict{Symbol,Any}, t::NoTracking) = nothing
"""
    _add_result!(d::Dict{Symbol,Any}, t::TopKTracking)

Add tracking results to alg_result. 
"""
function _add_result!(d::Dict{Symbol,Any}, t::TopKTracking)
    d[:top_k] = collect(t.q)
    d
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
    evaluate!(p::CrossEntropy, loss::Function, grammar::Grammar, pop::Vector{RuleNode}, 
        losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(p::CrossEntropy, loss::Function, grammar::Grammar, pop::Vector{RuleNode}, 
    losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

    Threads.@threads for i in 1:length(losses)
        losses[i] = loss(pop[i], grammar)
    end

    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
    end
    _update_tracker!(p.track_method, pop, losses)
    (best_tree, best_loss)
end

"""
    _update_tracker!(t::NoTracking, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}}) 

Update the tracker.  No op for NoTracking.
"""
function _update_tracker!(t::NoTracking, pop::Vector{RuleNode}, losses::Vector{Float64}) 
    nothing
end
"""
    _update_tracker!(t::TopKTracking, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}})

Update the tracker.  Track top k expressions. 
"""
function _update_tracker!(t::TopKTracking, pop::Vector{RuleNode}, losses::Vector{Float64})
    n = 0
    for i = 1:length(pop)
        r = enqueue!(t.q, pop[i], losses[i])
        r >= 0 && (n += 1) #no clash, increment counter
        n >= t.k && break 
    end
end

end #module
