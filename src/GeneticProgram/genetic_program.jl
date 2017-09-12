
module GeneticProgram

using ExprRules
using StatsBase
using ..loss
using ..ExprOptParams
using ..ExprOptResults

import ..optimize

export GeneticProgramParams

const OPERATORS = [:reproduction, :crossover, :mutation]

abstract type InitializationMethod end 
abstract type SelectionMethod end

"""
    GeneticProgramParams(pop_size::Int, iterations::Int, max_depth::Int, 
        p_reproduction::Float64, p_crossover::Float64, p_mutation::Float64;
        init_method::InitializationMethod=RandomInit(), 
        select_method::SelectionMethod=TournamentSelection())

Parameters for Genetic Programming.
    pop_size: population size
    iterations: number of iterations
    max_depth: maximum depth of derivation tree
    p_reproduction: probability of reproduction operator
    p_crossover: probability of crossover operator
    p_mutation: probability of mutation operator
    init_method: Initialization method
    select_method: Initialization method
"""
struct GeneticProgramParams <: ExprOptParams
    pop_size::Int
    iterations::Int
    max_depth::Int
    p_operators::Weights
    init_method::InitializationMethod
    select_method::SelectionMethod

    function GeneticProgramParams(
        pop_size::Int,                          #population size 
        iterations::Int,                        #number of generations 
        max_depth::Int,                         #maximum depth of derivation tree
        p_reproduction::Float64,                #probability of reproduction operator 
        p_crossover::Float64,                   #probability of crossover operator
        p_mutation::Float64;                    #probability of mutation operator 
        init_method::InitializationMethod=RandomInit(),      #initialization method 
        select_method::SelectionMethod=TournamentSelection())   #selection method 

        p_operators = Weights([p_reproduction, p_crossover, p_mutation])
        new(pop_size, iterations, max_depth, p_operators, init_method, select_method)
    end
end

struct RandomInit <: InitializationMethod end
struct TournamentSelection <: SelectionMethod 
    tournament_size::Int
end
TournamentSelection() = TournamentSelection(2)
struct TruncationSelection <: SelectionMethod 
    k::Int #top k to keep
end
TruncationSelection() = TruncationSelection(100)

optimize(p::GeneticProgramParams, grammar::Grammar, typ::Symbol) = genetic_program(p, grammar, typ)

"""
    genetic_program(p::GeneticProgramParams, grammar::Grammar, typ::Symbol)

Strongly-typed genetic programming optimization. See: 

Montana, "Strongly-typed genetic programming", Evolutionary Computation, Vol 3, Issue 2, 1995.
Koza, "Genetic programming: on the programming of computers by means of natural selection", MIT Press, 1992 

Three operators are implemented: reproduction, crossover, and mutation.
"""
function genetic_program(p::GeneticProgramParams, grammar::Grammar, typ::Symbol)

    pop0 = initialize(p.init_method, p.pop_size, grammar, typ, p.max_depth)
    pop1 = Vector{RuleNode}(p.pop_size)
    losses = Vector{Float64}(p.pop_size)

    best_tree, best_loss = evaluate!(pop0, losses, pop0[1], Inf)
    for iter = 1:p.iterations 
        i = 0
        while i < p.pop_size
            op = sample(OPERATORS, p.p_operators)
            if op == :reproduction
                ind1 = select(p.select_method, pop0, losses)
                pop1[i+=1] = deepcopy(ind1)
            elseif op == :crossover
                ind1 = select(p.select_method, pop0, losses)
                ind2 = select(p.select_method, pop0, losses)
                child = crossover(ind1, ind2, grammar)
                pop1[i+=1] = child
            elseif op == :mutation
                ind1 = select(p.select_method, pop0, losses)
                child1 = mutation(ind1, grammar, p.max_depth)
                pop1[i+=1] = child1
            end
        end
        pop0, pop1 = pop1, pop0
        best_tree, best_loss = evaluate!(pop0, losses, best_tree, best_loss)
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

"""
    initialize(::RandomInit, pop_size::Int, grammar::Grammar, typ::Symbol, max_depth::Int)

Random population initialization.
"""
initialize(::RandomInit, pop_size::Int, grammar::Grammar, typ::Symbol, max_depth::Int) = 
    [rand(RuleNode, grammar, typ, max_depth) for i = 1:pop_size]

"""
    select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Float64})

Tournament selection.
"""
function select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Float64})
    ids = StatsBase.seqsample_c!(collect(1:length(pop)), zeros(Int, p.tournament_size)) 
    pop[ids[1]] #assume sorted
end

"""
    select(p::TruncationSelection, pop::Vector{RuleNode}, losses::Vector{Float64})

Truncation selection.
"""
function select(p::TruncationSelection, pop::Vector{RuleNode}, losses::Vector{Float64})
    pop[rand(1:p.k)]
end

"""
    evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, 
    best_loss::Float64)

    losses[:] = loss.(pop)
    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
    end
    (best_tree, best_loss)
end

"""
    crossover(a::RuleNode, b::RuleNode, grammar::Grammar)

Crossover genetic operator.  Pick a random node from 'a', then pick a random node from 'b' that has the same type, then replace the subtree 
"""
function crossover(a::RuleNode, b::RuleNode, grammar::Grammar)
    child = deepcopy(a)
    loc = sample(NodeLoc, child)
    typ = return_type(grammar, get(child, loc).ind)
    if contains_returntype(b, grammar, typ)
        subtree = sample(b, typ, grammar)
        insert!(child, loc, subtree)
    end
    child 
end

"""
    mutation(a::RuleNode, grammar::Grammar, max_depth::Int=3)

Mutation genetic operator.  Pick a random node from 'a', then replace the subtree with a random one.
"""
function mutation(a::RuleNode, grammar::Grammar, max_depth::Int=5)
    child = deepcopy(a)
    loc = sample(NodeLoc, child)
    typ = return_type(grammar, get(child, loc).ind)
    subtree = rand(RuleNode, grammar, typ)
    insert!(child, loc, subtree)
    child
end

end #module
