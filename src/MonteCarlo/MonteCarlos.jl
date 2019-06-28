
module MonteCarlos

using ExprRules

using ExprOptimization: ExprOptAlgorithm, ExprOptResult, BoundedPriorityQueue, enqueue!
import ExprOptimization: optimize

export MonteCarlo

abstract type TrackingMethod end

"""
    MonteCarlo

Monte Carlo.
# Arguments:
- `num_samples::Int`: number of samples
- `max_depth::Int`: maximum depth of derivation tree
- `track_method::TrackingMethod`: additional tracking, e.g., track top k exprs (default: no additional tracking) 
"""
struct MonteCarlo <: ExprOptAlgorithm
    num_samples::Int
    max_depth::Int
    track_method::TrackingMethod

    function MonteCarlo(
        num_samples::Int,
        max_depth::Int;
        track_method::TrackingMethod=NoTracking())   #tracking method

        new(num_samples, max_depth, track_method)
    end
end

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
    optimize(p::MonteCarlo, grammar::Grammar, typ::Symbol, loss::Function; kwargs...)

    Expression tree optimization using Monte Carlo with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar).
"""
function optimize(p::MonteCarlo, grammar::Grammar, typ::Symbol, loss::Function; kwargs...) 
    monte_carlo(p, grammar, typ, loss; kwargs...)
end

"""
    monte_carlo(p::MonteCarlo, grammar::Grammar, typ::Symbol, loss::Function)

Expression tree optimization using Monte Carlo with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'.  Loss function has the form : los::Float64=loss(node::RuleNode, grammar::Grammar).  Draw Monte Carlo samples from the grammar and return the one with the best loss.
"""
function monte_carlo(p::MonteCarlo, grammar::Grammar, typ::Symbol, loss::Function; 
    verbose::Bool=false)
    dmap = mindepth_map(grammar)
    best_tree, best_loss = RuleNode(0), Inf
    for i = 1:p.num_samples
        verbose && println("samples: $i of $(p.num_samples)")
        tree = rand(RuleNode, grammar, typ, dmap, p.max_depth)
        los = float(loss(tree, grammar))
        if los < best_loss
            best_tree, best_loss = tree, los
        end
        _update_tracker!(p.track_method, tree, los)
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
    _update_tracker!(t::NoTracking, tree::RuleNode, los::Float64) 

Update the tracker.  No op for NoTracking.
"""
_update_tracker!(t::NoTracking, tree::RuleNode, los::Float64) = nothing
"""
    _update_tracker!(t::TopKTracking, tree::RuleNode, los::Float64) 

Update the tracker.  Track top k expressions. 
"""
_update_tracker!(t::TopKTracking, tree::RuleNode, los::Float64) = enqueue!(t.q, tree, los)

end #module
