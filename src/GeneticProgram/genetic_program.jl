
module GeneticProgram

using ExprRules
using StatsBase
using ..loss
using ..ExprOptParams
using ..ExprOptResults

import ..optimize

export GeneticProgramParams

const OPERATORS = [:reproduction, :crossover, :mutation]

abstract type Initializer end 
abstract type Selector end

struct GeneticProgramParams <: ExprOptParams
    pop_size::Int
    iterations::Int
    max_depth::Int
    op_probs::Weights
    initializer::Initializer
    selector::Selector
end
function GeneticProgramParams(
    pop_size::Int,                          #population size 
    iterations::Int,                        #number of generations 
    max_depth::Int,                         #maximum depth of derivation tree
    p_reproduction::Float64,                #probability of reproduction operator 
    p_crossover::Float64,                   #probability of crossover operator
    p_mutation::Float64;                    #probability of mutation operator 
    initializer::Initializer=RandomInit(),      #initialization method 
    selector::Selector=TournamentSelection())   #selection method 

    op_probs = Weights([p_reproduction, p_crossover, p_mutation])
    GeneticProgramParams(pop_size, iterations, max_depth, op_probs, initializer, selector)
end

struct RandomInit <: Initializer end
struct TournamentSelection <: Selector 
    tournament_size::Int
end
TournamentSelection() = TournamentSelection(2)

optimize(p::GeneticProgramParams, ruleset::RuleSet, typ::Symbol) = genetic_program(p, ruleset, typ)

"""
    genetic_program(p::GeneticProgramParams, ruleset::RuleSet, typ::Symbol)

Strongly-typed genetic programming optimization. See: 

Montana, "Strongly-typed genetic programming", Evolutionary Computation, Vol 3, Issue 2, 1995.
Koza, "Genetic programming: on the programming of computers by means of natural selection", MIT Press, 1992 

Three operators are implemented: reproduction, crossover, and mutation.
"""
function genetic_program(p::GeneticProgramParams, ruleset::RuleSet, typ::Symbol)

    pop0 = initialize(p.initializer, p.pop_size, ruleset, typ, p.max_depth)
    pop1 = Vector{RuleNode}(p.pop_size)
    losses = Vector{Float64}(p.pop_size)

    best_tree, best_loss = evaluate!(pop0, losses, pop0[1], Inf)

    for iter = 1:p.iterations 
        i = 0
        while i < p.pop_size
            op = sample(OPERATORS, p.op_probs)
            if op == :reproduction
                ind1 = select(p.selector, pop0, losses)
                pop1[i+=1] = deepcopy(ind1)
            elseif op == :crossover
                ind1 = select(p.selector, pop0, losses)
                ind2 = select(p.selector, pop0, losses)
                child1, child2 = crossover(ind1, ind2, ruleset)
                pop1[i+=1] = child1
                if i < p.pop_size
                    pop1[i+=1] = child2
                end
            elseif op == :mutation
                ind1 = select(p.selector, pop0, losses)
                child1 = mutation(ind1, ruleset, p.max_depth)
                pop1[i+=1] = child1
            end
        end
        pop0, pop1 = pop1, pop0
        best_tree, best_loss = evaluate!(pop0, losses, best_tree, best_loss)
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, ruleset), nothing)
end

"""
initialize(::RandomInit, pop_size::Int, ruleset::RuleSet, typ::Symbol, max_depth::Int)

Random population initialization.
"""
initialize(::RandomInit, pop_size::Int, ruleset::RuleSet, typ::Symbol, max_depth::Int) = 
    [rand(RuleNode, ruleset, typ, max_depth) for i = 1:pop_size]

"""
select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Float64})

Tournament selection.
"""
function select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Float64})
    ids = StatsBase.seqsample_c!(collect(1:length(pop)), zeros(Int, p.tournament_size)) 
    pop[ids[1]] #assume sorted
end

"""
evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, 
    best_loss::Float64)
    for (i, ind) in enumerate(pop)
        losses[i] = loss(ind)
    end
    perm = sortperm(losses)
    losses[:] = losses[perm]
    pop[:] = pop[perm]

    if losses[1] < best_loss
        best_tree, best_loss = pop[1], losses[1]
    end
    (best_tree, best_loss)
end

"""
crossover(a::RuleNode, b::RuleNode, ruleset::RuleSet)

Crossover genetic operator.  Pick a random node from 'a', then pick a random node from 'b' that has the same type, then exchange the subtrees.
"""
function crossover(a::RuleNode, b::RuleNode, ruleset::RuleSet)
    child_a = deepcopy(a)
    child_b = deepcopy(b)

    loc_a = sample(NodeLoc, child_a)
    node_a = get(child_a, loc_a) 

    loc_b = sample(NodeLoc, child_b, ruleset.types[node_a.ind], ruleset) 
    node_b = get(child_b, loc_b)

    insert!(child_a, loc_a, node_b)
    insert!(child_b, loc_b, node_a)

    (child_a, child_b)
end

"""
mutation(a::RuleNode, ruleset::RuleSet, max_depth::Int=3)

Mutation genetic operator.  Pick a random node from 'a', then replace the subtree with a random one.
"""
function mutation(a::RuleNode, ruleset::RuleSet, max_depth::Int=3)
    child = deepcopy(a)
    loc = sample(NodeLoc, child)
    node = get(child, loc)
    insert!(child, loc, rand(RuleNode, ruleset, ruleset.types[node.ind], max_depth))

    child
end

end #module
