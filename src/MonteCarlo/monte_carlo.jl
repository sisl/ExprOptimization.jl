
module MonteCarlo

using ExprRules
using ..loss
using ..ExprOptParams
using ..ExprOptResults

import ..optimize

export MonteCarloParams

struct MonteCarloParams <: ExprOptParams
    num_samples::Int
    max_depth::Int
end

optimize(p::MonteCarloParams, grammar::Grammar, typ::Symbol) = monte_carlo(p, grammar, typ)

"""
    monte_carlo(p::MonteCarloParams, grammar::Grammar, typ::Symbol)

Draw Monte Carlo samples from the grammar and return the one with the best loss.
"""
function monte_carlo(p::MonteCarloParams, grammar::Grammar, typ::Symbol)
    best_tree = rand(RuleNode, grammar, typ, p.max_depth)
    best_loss = loss(best_tree)
    for i = 2:p.num_samples
        tree = rand(RuleNode, grammar, typ, p.max_depth)
        los = loss(tree)
        if los < best_loss
            best_tree, best_loss = tree, los
        end
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

end #module