__precompile__()

module ExprOptimization

using ExprRules

export 
        @ruleset,
        RuleSet,
        RuleNode,
        get_executable,

        loss,
        optimize,
        ExprOptResults,
        MonteCarloParams,
        GeneticProgramParams

abstract type ExprOptParams end

struct ExprOptResults
    tree::RuleNode
    loss::Float64
    expr::Any
    alg_results::Any
end

function loss end #implemented by user
function optimize end  #implemented by algorithms

include("MonteCarlo/monte_carlo.jl")
using .MonteCarlo

end # module
