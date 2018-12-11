
module MonteCarlos

using ExprRules

using ExprOptimization: ExprOptAlgorithm, ExprOptResult
import ExprOptimization: optimize

export MonteCarlo

"""
    MonteCarlo

Monte Carlo.
# Arguments:
- `num_samples::Int`: number of samples
- `max_depth::Int`: maximum depth of derivation tree
"""
struct MonteCarlo <: ExprOptAlgorithm
    num_samples::Int
    max_depth::Int
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
        los = loss(tree, grammar)
        if los < best_loss
            best_tree, best_loss = tree, los
        end
    end
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

end #module
