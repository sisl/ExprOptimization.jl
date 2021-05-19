__precompile__()

module ExprOptimization

export 
        @grammar,
        Grammar,
        RuleNode,
        get_executable,

        optimize,
        ExprOptAlgorithm,
        ExprOptResult,
        get_expr,

        ProbabilisticExprRules,
        PPTs,
        PPT,

        MonteCarlos,
        MonteCarlo,

        GeneticPrograms,
        GeneticProgram,

        GrammaticalEvolutions,
        GrammaticalEvolution,

        CrossEntropys,
        CrossEntropy,

        PIPEs,
        PIPE


using Reexport
@reexport using ExprRules

abstract type ExprOptAlgorithm end

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
    optimize(p::ExprOptAlgorithm, grammar::Grammar, typ::Symbol, loss::Function; kwargs...)

Main entry for expression optimization.  Use concrete ExprOptAlgorithm to specify optimization algorithm. Optimize using grammar and start symbol, typ, and loss function.  Loss function has the form: los::Float64=loss(node::RuleNode).
"""
function optimize end   #implemented by algorithms

"""
    get_expr(result::ExprOptResult) 

Returns the expression in the result
"""
get_expr(result::ExprOptResult) = result.expr
get_expr(x::Nothing) = nothing

#############################################################################
# Common base modules
include("../contrib/BoundedPriorityQueues.jl")
using .BoundedPriorityQueues

include("ProbabilisticExprRules/ProbabilisticExprRules.jl")
include("PPT/PPTs.jl")
using .PPTs: PPT

#############################################################################
# Optimization algorithms

include("MonteCarlo/MonteCarlos.jl")
using .MonteCarlos

include("GeneticProgram/GeneticPrograms.jl")
using .GeneticPrograms

include("GrammaticalEvolution/GrammaticalEvolutions.jl")
using .GrammaticalEvolutions

include("CrossEntropy/CrossEntropys.jl")
using .CrossEntropys

include("PIPE/PIPEs.jl")
using .PIPEs

end # module
