
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

optimize(p::MonteCarloParams, ruleset::RuleSet, typ::Symbol) = monte_carlo(p, ruleset, typ)

"""
    monte_carlo(p::MonteCarloParams, ruleset::RuleSet, typ::Symbol)

TODO
"""
function monte_carlo(p::MonteCarloParams, ruleset::RuleSet, typ::Symbol)
    best_tree = rand(RuleNode, ruleset, typ, p.max_depth)
    best_loss = loss(best_tree)
    for i = 2:p.num_samples
        tree = rand(RuleNode, ruleset, typ, p.max_depth)
        los = loss(tree)
        if los < best_loss
            best_tree = tree
            best_loss = los
        end
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, ruleset), nothing)
end

end #module
