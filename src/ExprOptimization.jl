__precompile__()

module ExprOptimization

export 
        @grammar,
        Grammar,
        RuleNode,
        get_executable,

        optimize,
        ExprOptParams,
        ExprOptResult,

        ProbabilisticExprRules,
        PPT,
        PPTParams,

        MonteCarlo,
        MonteCarloParams,

        GeneticProgram,
        GeneticProgramParams,

        GrammaticalEvolution,
        GrammaticalEvolutionParams,

        CrossEntropy,
        CrossEntropyParams,

        PIPE,
        PIPEParams


using Reexport
@reexport using ExprRules

abstract type ExprOptParams end

"""
    ExprOptResult

Returned by optimize().  Contains the results of the optimization.
"""
struct ExprOptResult
    tree::RuleNode #best tree
    loss::Float64 #best loss
    expr::Any #best expression
    alg_result::Any #algorithm-specific results
end

"""
    optimize(p::ExprOptParams, grammar::Grammar, typ::Symbol, loss::Function)

Main entry for expression optimization.  Use concrete ExprOptParams to specify optimization algorithm. Optimize using grammar and start symbol, typ, and loss function.  Loss function has the form: los::Float64=loss(node::RuleNode).
"""
function optimize end   #implemented by algorithms

#############################################################################
# Common base modules
include("ProbabilisticExprRules/ProbabilisticExprRules.jl")

include("PPT/ppt.jl")
using .PPT: PPTParams

#############################################################################
# Optimization algorithms

include("MonteCarlo/monte_carlo.jl")
using .MonteCarlo

include("GeneticProgram/genetic_program.jl")
using .GeneticProgram

include("GrammaticalEvolution/grammatical_evolution.jl")
using .GrammaticalEvolution

include("CrossEntropy/cross_entropy.jl")
using .CrossEntropy

include("PIPE/pipe.jl")
using .PIPE

end # module
