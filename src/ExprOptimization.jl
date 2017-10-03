__precompile__()

module ExprOptimization

using ExprRules

export 
        @grammar,
        Grammar,
        RuleNode,
        get_executable,

        loss,
        optimize,
        ExprOptResults,

        MonteCarlo,
        MonteCarloParams,

        GeneticProgram,
        GeneticProgramParams,

        CrossEntropy,
        CrossEntropyParams,

        PIPE,
        PIPEParams

abstract type ExprOptParams end

"""
    ExprOptResults

Returned by optimize().  Contains the results of the optimization.
"""
struct ExprOptResults
    tree::RuleNode #best tree
    loss::Float64 #best loss
    expr::Any #best expression
    alg_results::Any #algorithm-specific results
end

"""
    loss(tree::RuleNode)

User-defined loss function.  Should be overloaded by user.  Takes an expression tree and returns a real number.  The loss is minimized by the optimization algorithms.
"""
function loss end       #loss function, loss(tree::RuleNode), implemented by user

"""
    optimize(p::ExprOptParams, grammar::Grammar, typ::Symbol)

Main entry for expression optimization.  Use concrete ExprOptParams to specify optimization algorithm. Optimize using grammar and start symbol, typ.
"""
function optimize end   #implemented by algorithms

include("MonteCarlo/monte_carlo.jl")
using .MonteCarlo

include("GeneticProgram/genetic_program.jl")
using .GeneticProgram

include("CrossEntropy/cross_entropy.jl")
using .CrossEntropy

include("PIPE/pipe.jl")
using .PIPE

end # module
