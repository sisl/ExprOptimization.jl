
module GrammaticalEvolution

using ExprRules
using StatsBase
using ..loss
using ..ExprOptParams
using ..ExprOptResults

import ..optimize

export GrammaticalEvolutionParams

const OPERATORS = [:reproduction, :crossover, :mutation]

abstract type SelectionMethod end
abstract type MutationMethod end

"""
    GrammaticalEvolutionParams(num_samples::Int, max_depth::Int)

Parameters for Monte Carlo.
    num_samples: Number of samples
    max_depth: maximum depth of derivation tree
"""
struct GrammaticalEvolutionParams <: ExprOptParams
    pop_size::Int
    iterations::Int
    gene_length::Int
    max_depth::Int
    p_operators::Weights
    select_method::SelectionMethod
    mutate_method::MutationMethod

    function GrammaticalEvolutionParams(
        grammar::Grammar,
        typ::Symbol,
        pop_size::Int,                          #population size 
        iterations::Int,                        #number of generations 
        gene_length::Int,                       #length of genotype Int vector
        max_depth::Int,                         #maximum depth of derivation tree
        p_reproduction::Float64,                #probability of reproduction operator 
        p_crossover::Float64,                   #probability of crossover operator
        p_mutation::Float64;                    #probability of mutation operator 
        select_method::SelectionMethod=TournamentSelection(),   #selection method 
        mutate_method::MutationMethod=MultiMutate([
                                                   RandomMutation(0.1),
                                                   GeneDuplication(),
                                                   GenePruning(0.1, grammar, typ) ]))
        
        p_operators = Weights([p_reproduction, p_crossover, p_mutation])
        new(pop_size, iterations, gene_length, max_depth, p_operators, select_method, mutate_method)
    end
end

"""
    TournamentSelection

Tournament selection method with tournament size k.
"""
struct TournamentSelection <: SelectionMethod 
    k::Int
end
TournamentSelection() = TournamentSelection(2)

"""
    TruncationSelection

Truncation selection method keeping the top k individuals 
"""
struct TruncationSelection <: SelectionMethod 
    k::Int 
end
TruncationSelection() = TruncationSelection(100)

struct RandomMutation <: MutationMethod
    p_mutate::Float64
end

struct GeneDuplication <: MutationMethod end

struct GenePruning <: MutationMethod
    p_prune::Float64
    grammar::Grammar
    typ::Symbol
end

struct MultiMutate <: MutationMethod
    Ms::Vector{MutationMethod} 
end

"""
    optimize(p::GrammaticalEvolutionParams, grammar::Grammar, typ::Symbol)

Grammatical Evolution algorithm with parameters p, grammar 'grammar', and start symbol typ.

See: Ryan, O'Neil...
"""
optimize(p::GrammaticalEvolutionParams, grammar::Grammar, typ::Symbol) = grammatical_evolution(p, grammar, typ)

"""
    grammatical_evolution(p::GrammaticalEvolutionParams, grammar::Grammar, typ::Symbol)

Grammatical Evolution algorithm with parameters p, grammar 'grammar', and start symbol typ.

See: Ryan, O'Neil...
"""
function grammatical_evolution(p::GrammaticalEvolutionParams, grammar::Grammar, typ::Symbol)
    iseval(grammar) && error("Grammatical Evolution does not support _() functions in the grammar")

    pop0 = initialize(p.pop_size, p.gene_length) 
    pop1 = [Int[] for i=1:p.pop_size]
    losses = Vector{Float64}(p.pop_size)

    best_tree, best_loss = evaluate!(p, grammar, typ, pop0, losses, RuleNode(0), Inf)
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
                child = crossover(ind1, ind2)
                limit_length!(child, p.gene_length)
                pop1[i+=1] = child
            elseif op == :mutation
                ind1 = select(p.select_method, pop0, losses)
                child1 = mutation(p.mutate_method, ind1)
                limit_length!(child1, p.gene_length)
                pop1[i+=1] = child1
            end
        end
        pop0, pop1 = pop1, pop0
        best_tree, best_loss = evaluate!(p, grammar, typ, pop0, losses, best_tree, best_loss)
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

initialize(pop_size::Int, len::Int) = [rand(Int, len) for i = 1:pop_size]

function limit_length!(ind::Vector{Int}, max_length::Int)
    resize!(ind, min(length(ind), max_length))
    ind
end

"""
    select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Float64})

Tournament selection.
"""
function select(p::TournamentSelection, pop::Vector{Vector{Int}}, losses::Vector{Float64})
    ids = StatsBase.seqsample_c!(collect(1:length(pop)), zeros(Int, p.k)) 
    pop[ids[1]] #assume sorted
end

"""
    select(p::TruncationSelection, pop::Vector{RuleNode}, losses::Vector{Float64})

Truncation selection.
"""
function select(p::TruncationSelection, pop::Vector{Vector{Int}}, losses::Vector{Float64})
    pop[rand(1:p.k)]
end

"""
    crossover(a::Vector{Int}, b::Vector{Int})

Crossover genetic operator.  Pick a random crossover point from 'a', then pick a crossover point from 'b', splice first part of 'a' with second part of 'b' 
"""
function crossover(a::Vector{Int}, b::Vector{Int})
    i = rand(1:length(a))
    return vcat(a[1:i], b[i+1:end])
end

"""
    mutate!(::RandomMutation, a::Vector{Int})

Mutation genetic operator.  Pick a random point from 'a', then...
"""
function mutation(p::RandomMutation, a::Vector{Int})
    child = deepcopy(a)
    for i in eachindex(child)
        if rand() < p.p_mutate
            child[i] = rand(Int)
        end
    end
    return child 
end
function mutation(p::GeneDuplication, a::Vector{Int})
    child = a
    n = length(a)
    i, j = rand(1:n), rand(1:n)
    interval = min(i,j) : max(i,j)
    return vcat(child, child[interval])
end
function mutation(p::GenePruning, a::Vector{Int})
    child = deepcopy(a)
    if rand() < p.p_prune
        c = decode(child, p.grammar, p.typ).n_rules_applied
        if c < length(a)
            child = child[1:c] 
        end
    end
    return child 
end
function mutation(p::MultiMutate, child::Vector{Int})
    for m in p.Ms
        child = mutation(m, child)
    end
    return child 
end

"""
    evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(p::GrammaticalEvolutionParams, grammar::Grammar, typ::Symbol, 
                   pop::Vector{Vector{Int}}, losses::Vector{Float64}, best_tree::RuleNode, 
                   best_loss::Float64)

    for i in eachindex(losses)
        decoded = decode(pop[i], grammar, typ)
        losses[i] = depth(decoded.node) > p.max_depth ?  Inf : loss(decoded.node)
    end
    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_loss = losses[1]
        best_tree = decode(pop[1], grammar, typ).node
    end
    (best_tree, best_loss)
end

struct DecodedExpression
    node::RuleNode
    n_rules_applied::Int
end
function decode(x::Vector{Int}, grammar::Grammar, typ::Symbol, c_max=1000, c=0)
    node, c = _decode(x, grammar, typ, c_max, c)
    DecodedExpression(node, c)
end
function _decode(x::Vector{Int}, grammar::Grammar, typ::Symbol, c_max=1000, c=0)
    types = grammar[typ]
    if length(types) > 1
        g = x[mod1(c+=1, length(x))]
        rule = types[mod1(g, length(types))]
    else
        rule = types[1]
    end
    node = RuleNode(rule)
    childtypes = child_types(grammar, node)
    if !isempty(childtypes) && c < c_max
        for ctyp in childtypes
            cnode, c = _decode(x, grammar, ctyp, c_max, c)
            push!(node.children, cnode)
        end
    end
    return (node, c)
end

end #module
