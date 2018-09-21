
module GrammaticalEvolutions

using ExprRules
using StatsBase

using ExprOptimization: ExprOptAlgorithm, ExprOptResult 
import ExprOptimization: optimize 

export GrammaticalEvolution

const OPERATORS = [:reproduction, :crossover, :mutation]

abstract type SelectionMethod end
abstract type MutationMethod end

"""
    GrammaticalEvolution

Grammatical Evolution.
# Arguments
- `grammar::Grammar`: grammar
- `typ::Symbol`: start symbol
- `pop_size::Int`: population size
- `iterations::Int`: number of iterations
- `init_gene_length::Int`: initial length of genotype integer array
- `max_gene_length::Int`: maximum length of genotype integer array
- `max_depth::Int`: maximum depth of derivation tree
- `p_reproduction::Float64`: probability of reproduction operator
- `p_crossover::Float64`: probability of crossover operator
- `p_mutation::Float64`: probability of mutation operator
- `select_method::SelectionMethod`: selection method (default: tournament selection)
- `mutate_method::InitializationMethod`: mutation method (default: multi-mutate)
"""
struct GrammaticalEvolution <: ExprOptAlgorithm
    pop_size::Int
    iterations::Int
    init_gene_length::Int
    max_gene_length::Int
    max_depth::Int
    p_operators::Weights
    select_method::SelectionMethod
    mutate_method::MutationMethod

    function GrammaticalEvolution(
        grammar::Grammar,
        typ::Symbol,
        pop_size::Int,                          #population size 
        iterations::Int,                        #number of generations 
        init_gene_length::Int,                  #initial length of genotype Int vector
        max_gene_length::Int,                   #maximum length of genotype Int vector
        max_depth::Int,                         #maximum depth of derivation tree
        p_reproduction::Float64,                #probability of reproduction operator 
        p_crossover::Float64,                   #probability of crossover operator
        p_mutation::Float64;                    #probability of mutation operator 
        select_method::SelectionMethod=TournamentSelection(),   #selection method 
        mutate_method::MutationMethod=MultiMutate(grammar, typ))
        
        p_operators = Weights([p_reproduction, p_crossover, p_mutation])
        new(pop_size, iterations, init_gene_length, max_gene_length, max_depth, p_operators, select_method, mutate_method)
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

"""
    RandomMutation

Randomly change each entry in integer array with probability p_mutate
"""
struct RandomMutation <: MutationMethod
    p_mutate::Float64
end
RandomMutation() = RandomMutation(0.1)

"""
    GeneDuplication

Pick a random segment of the gene and duplicate it at the end.
"""
struct GeneDuplication <: MutationMethod end

"""
    GenePruning

With probability p_prune, decode the gene using grammar and start symbol typ and discard the genes not used.
"""
struct GenePruning <: MutationMethod
    p_prune::Float64
    grammar::Grammar
    typ::Symbol
end
GenePruning(grammar::Grammar, typ::Symbol) = GenePruning(0.1, grammar, typ)

"""
    MultiMutate

Apply a sequence of mutation operators.
"""
struct MultiMutate <: MutationMethod
    Ms::Vector{MutationMethod} 
end
MultiMutate(grammar::Grammar, typ::Symbol) = MultiMutate([
    RandomMutation(), 
    GeneDuplication(), 
    GenePruning(grammar, typ)])

"""
    optimize(p::GrammaticalEvolution, grammar::Grammar, typ::Symbol, loss::Function)

Grammatical Evolution algorithm with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'.  Loss function has the form: los::Float64=loss(node::RuleNode, grammar::Grammar).

See: Ryan, Collins, O'Neil, "Grammatical Evolution: Evolving Programs for an Arbitrary Language", 
    in European Conference on Genetic Programming, Spring, 1998, pp. 83-96. 
"""
optimize(p::GrammaticalEvolution, grammar::Grammar, typ::Symbol, loss::Function) = 
    grammatical_evolution(p, grammar, typ, loss)

"""
    grammatical_evolution(p::GrammaticalEvolution, grammar::Grammar, typ::Symbol, loss::Function)

Grammatical Evolution algorithm with parameters p, grammar 'grammar', start symbol typ, and loss function 'loss'.  Loss function has the form los::Float64=loss(node::RuleNode, grammar::Grammar).

See: Ryan, Collins, O'Neil, "Grammatical Evolution: Evolving Programs for an Arbitrary Language", 
    in European Conference on Genetic Programming, Spring, 1998, pp. 83-96. 
"""
function grammatical_evolution(p::GrammaticalEvolution, grammar::Grammar, typ::Symbol, loss::Function)
    iseval(grammar) && error("Grammatical Evolution does not support _() functions in the grammar")

    pop0 = initialize(p.pop_size, p.init_gene_length) 
    pop1 = [Int[] for i=1:p.pop_size]
    losses0 = Vector{Union{Float64,Missing}}(missing,p.pop_size)
    losses1 = Vector{Union{Float64,Missing}}(missing,p.pop_size)

    best_tree, best_loss = evaluate!(p, grammar, typ, loss, pop0, losses0, RuleNode(0), Inf)
    for iter = 1:p.iterations 
        fill!(losses1, missing)
        i = 0
        while i < p.pop_size
            op = sample(OPERATORS, p.p_operators)
            if op == :reproduction
                ind1,j = select(p.select_method, pop0, losses0)
                pop1[i+=1] = ind1
                losses1[i] = losses0[j]
            elseif op == :crossover
                ind1,_ = select(p.select_method, pop0, losses0)
                ind2,_ = select(p.select_method, pop0, losses0)
                child = crossover(ind1, ind2)
                limit_length!(child, p.max_gene_length)
                pop1[i+=1] = child
            elseif op == :mutation
                ind1,_ = select(p.select_method, pop0, losses0)
                child1 = mutation(p.mutate_method, ind1)
                limit_length!(child1, p.max_gene_length)
                pop1[i+=1] = child1
            end
        end
        pop0, pop1 = pop1, pop0
        losses0, losses1 = losses1, losses0
        best_tree, best_loss = evaluate!(p, grammar, typ, loss, pop0, losses0, best_tree, best_loss)
    end
    ExprOptResult(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

"""
    initialize(pop_size::Int, len::Int)

Randomly initialize population.
"""
initialize(pop_size::Int, len::Int) = [rand(Int, len) for i = 1:pop_size]

"""
    limit_length!(ind::Vector{Int}, max_length::Int)

Limit the length of the gene ind to max_length.
"""
function limit_length!(ind::Vector{Int}, max_length::Int)
    resize!(ind, min(length(ind), max_length))
    ind
end

"""
    select(p::TournamentSelection, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}})

Tournament selection.
"""
function select(p::TournamentSelection, pop::Vector{Vector{Int}}, losses::Vector{Union{Float64,Missing}})
    ids = StatsBase.sample(1:length(pop), p.k; replace=false, ordered=true) 
    i = ids[1] #assumes pop is sorted
    pop[i], i
end

"""
    select(p::TruncationSelection, pop::Vector{RuleNode}, losses::Vector{Union{Float64,Missing}})

Truncation selection.
"""
function select(p::TruncationSelection, pop::Vector{Vector{Int}}, losses::Vector{Union{Float64,Missing}})
    i = rand(1:p.k)  #assumes pop is sorted
    pop[i], i
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
    mutation(::RandomMutation, a::Vector{Int})

Mutation the gene using RandomMutation.  Randomly change each entry in integer array with probability p_mutate
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

"""
    mutation(p::GeneDuplication, a::Vector{Int})

Mutate the gene using GeneDuplication.  Pick a random segment of the gene and duplicate it at the end.
"""
function mutation(p::GeneDuplication, a::Vector{Int})
    child = a
    n = length(a)
    i, j = rand(1:n), rand(1:n)
    interval = min(i,j) : max(i,j)
    return vcat(child, child[interval])
end

"""
    mutation(p::GenePruning, a::Vector{Int})

Mutate the gene using GenePruning.  With probability p_prune, decodes the gene and discards the genes not used.
"""
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

"""
    mutation(p::MultiMutate, a::Vector{Int})

Mutate the gene using MultiMutate.  Apply a sequence of mutation operators.
"""
function mutation(p::MultiMutate, child::Vector{Int})
    for m in p.Ms
        child = mutation(m, child)
    end
    return child 
end

"""
    evaluate!(p::GrammaticalEvolution, grammar::Grammar, typ::Symbol, loss::Function, pop::Vector{RuleNode}, losses::Vector{Union{Float64}}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree, if needed.
"""
function evaluate!(p::GrammaticalEvolution, grammar::Grammar, typ::Symbol, loss::Function,
                   pop::Vector{Vector{Int}}, losses::Vector{Union{Float64,Missing}}, best_tree::RuleNode, 
                   best_loss::Float64)

    for i in eachindex(pop) 
        if ismissing(losses[i])
            decoded = decode(pop[i], grammar, typ)
            losses[i] = depth(decoded.node) > p.max_depth ?  Inf : loss(decoded.node, grammar)
        end
    end

    perm = sortperm(losses)
    pop[:], losses[:] = pop[perm], losses[perm]
    if losses[1] < best_loss
        best_loss = losses[1]
        best_tree = decode(pop[1], grammar, typ).node
    end
    (best_tree, best_loss)
end

"""
    DecodedExpression

Results of a decode operation.  The decoded expression tree is stored in node and the number of rule is in n_rule_applied.
"""
struct DecodedExpression
    node::RuleNode
    n_rules_applied::Int
end

"""
    decode(x::Vector{Int}, grammar::Grammar, typ::Symbol, c_max=1000, c=0)

Decode an integer array (genotype) to a derivation tree (phenotype) using given grammar and start symbol typ.  Unlimited wraps.
"""
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
