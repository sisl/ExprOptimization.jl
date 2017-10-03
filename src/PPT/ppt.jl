module PPT

using StatsBase
using ExprRules

export 
        PPTParams,
        PPTNode,
        get_child,
        probabilities,
        probability,
        prune!


"""
    PPTParams

Parameters for Probabilistic Prototype Tree. 
# Arguments:
- `w_terminal::Float64`: probability of selecting a terminal on initialization 
- `w_nonterm::Float64`: probability of selecting a non-terminal on initialization 
"""
struct PPTParams
    w_terminal::Float64
    w_nonterm::Float64
end
PPTParams(w_terminal::Float64=0.6) = PPTParams(w_terminal, 1-w_terminal)

struct PPTNode
    ps::Dict{Symbol,Vector{Float64}}
    children::Vector{PPTNode}
end

"""
    PPTNode(pp::PPTParams, grammar::Grammar)

Node of a PPT.
"""
function PPTNode(pp::PPTParams, grammar::Grammar)
    ps = Dict(typ=>normalize!([isterminal(grammar, i) ?
                               pp.w_terminal : pp.w_nonterm for i in grammar[typ]], 1)
             for typ in nonterminals(grammar))
    PPTNode(ps, PPTNode[])
end

"""
    nchildren(node::PPTNode)

Returns the number of children of a node.
"""
ExprRules.nchildren(node::PPTNode) = length(node.children)

"""
    probabilities(node::PPTNode, typ::Symbol)

Returns the probability vector of a node. 
"""
probabilities(node::PPTNode, typ::Symbol) = node.ps[typ]

"""
    get_child(pp::PPTParams, ppt::PPTNode, grammar::Grammar, i::Int)

Returns child node i of a ppt node.  Will construct and initialize it if it doesn't already exist.
"""
function get_child(pp::PPTParams, ppt::PPTNode, grammar::Grammar, i::Int)
    if i > length(ppt.children)
        push!(ppt.children, PPTNode(pp, grammar))
    end
    ppt.children[i]
end

"""
    rand(pp::PPTParams, ppt::PPTNode, grammar::Grammar, typ::Symbol)

Randomly sample an expression tree from ppt model using parameters pp, grammar 'grammar', and start symbol typ.
"""
function Base.rand(pp::PPTParams, ppt::PPTNode, grammar::Grammar, typ::Symbol)
    rules = grammar[typ]
    rule_index = sample(rules, Weights(ppt.ps[typ]))
    ctypes = child_types(grammar, rule_index)
    node = iseval(grammar, rule_index) ? 
        RuleNode(rule_index, eval(grammar, rule_index), Vector{RuleNode}(length(ctypes))) :
        RuleNode(rule_index, Vector{RuleNode}(length(ctypes)))

    for (i,typ) in enumerate(ctypes)
        node.children[i] = rand(pp, get_child(pp, ppt, grammar, i), grammar, typ)
    end
    node
end

"""
    probability(pp::PPTParams, ppt::PPTNode, grammar::Grammar, expr::RuleNode)

Compute the probability of an expression tree expr given the model ppt using parameters pp and grammar 'grammar'.
"""
function probability(pp::PPTParams, ppt::PPTNode, grammar::Grammar, expr::RuleNode)
    typ = return_type(grammar, expr)
    i = findfirst(grammar[typ], expr.ind)
    retval = ppt.ps[typ][i]
    for (i,c) in enumerate(expr.children)
        retval *= probability(pp, get_child(pp, ppt, grammar, i), grammar, c)
    end
    retval
end

"""
    prune!(ppt::PPTNode, grammar::Grammar, p_threshold::Float64)

Prune the ppt of decisions with probability less than p_threshold.
"""
function prune!(ppt::PPTNode, grammar::Grammar, p_threshold::Float64)
    kmax, pmax = :None, 0.0
    for (k, p) in ppt.ps
        pmax′ = maximum(p)
        if pmax′ > pmax
            kmax, pmax = k, pmax′
        end
    end
    if pmax > p_threshold
        i = indmax(ppt.ps[kmax])
        if isterminal(grammar, i)
            clear!(ppt.children)
        else
            max_arity_for_rule = maximum(nchildren(grammar, r) for
                                         r in grammar[kmax])
            while length(ppt.children) > max_arity_for_rule
                pop!(ppt.children)
            end
        end
    end
    return ppt
end

end #module
