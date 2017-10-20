
module MonteCarlo

using ExprRules
using ..ExprOptParams
using ..ExprOptResult

import ..optimize

export MonteCarloParams

"""
    MonteCarloParams

Parameters for Monte Carlo.
# Arguments:
- `num_samples::Int`: number of samples
- `max_depth::Int`: maximum depth of derivation tree
"""
struct MonteCarloParams <: ExprOptParams
    num_samples::Int
    max_depth::Int
end

"""
    optimize(p::MonteCarloParams, grammar::Grammar, typ::Symbol, loss::Function)

    Expression tree optimization using Monte Carlo with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar).
"""
optimize(p::MonteCarloParams, grammar::Grammar, typ::Symbol, loss::Function) = monte_carlo(p, grammar, typ, loss)

"""
    monte_carlo(p::MonteCarloParams, grammar::Grammar, typ::Symbol, loss::Function)

Expression tree optimization using Monte Carlo with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'.  Loss function has the form : los::Float64=loss(node::RuleNode, grammar::Grammar).  Draw Monte Carlo samples from the grammar and return the one with the best loss.
"""
function monte_carlo(p::MonteCarloParams, grammar::Grammar, typ::Symbol, loss::Function)
    best_tree, best_loss = RuleNode(0), Inf
    for i = 1:p.num_samples
        tree = rand(RuleNode, grammar, typ, p.max_depth)
        los = loss(tree, grammar)
        if los < best_loss
            best_tree, best_loss = tree, los
        end
    end
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

end #module
