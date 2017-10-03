module ProbabilisticExprRules

using ExprRules
using StatsBase
using AbstractTrees

export 
    @grammar, 
    RuleNode, 

    ProbabilisticGrammar, 
    probabilities, 

    fit_mle!, 
    uniform!

const ProbType = Dict{Symbol,Vector{Float64}}

struct ProbabilisticGrammar
    grammar::Grammar
    probs::ProbType
end
function ProbabilisticGrammar(grammar::Grammar) 
    probs = ProbType() 
    for nt in nonterminals(grammar)
        n = length(grammar[nt])
        probs[nt] = ones(Float64, n) / n 
    end
    ProbabilisticGrammar(grammar, probs)
end

probabilities(pcfg::ProbabilisticGrammar, typ::Symbol) = pcfg.probs[typ]

"""
    rand(::Type{RuleNode}, grammar::ProbabilisticGrammar, typ::Symbol, max_depth::Int=10)

Generates a random RuleNode of return type typ and maximum depth max_depth.
"""
function Base.rand(::Type{RuleNode}, pcfg::ProbabilisticGrammar, typ::Symbol, max_depth::Int=10)
    grammar = pcfg.grammar
    rules = grammar[typ]
    probs = probabilities(pcfg, typ)

    rule_index = if max_depth > 1
        StatsBase.sample(rules, weights(probs))
    else
        inds = find(r->isterminal(grammar, r), rules)   
        rules, probs = rules[inds], probs[inds]
        StatsBase.sample(rules, weights(probs))
    end

    rulenode = grammar.iseval[rule_index] ?
        RuleNode(rule_index, eval(Main, grammar.rules[rule_index].args[2])) :
        RuleNode(rule_index)

    if !grammar.isterminal[rule_index]
        for ch in child_types(grammar, rule_index)
            push!(rulenode.children, rand(RuleNode, pcfg, ch, max_depth-1))
        end
    end
    return rulenode
end

"""
    uniform!(pcfg::ProbabilisticGrammar)

Set all probability vectors to uniform distribution
"""
function uniform!(pcfg::ProbabilisticGrammar)
    fill!(pcfg, 0.0)
    normalize!(pcfg)
end

"""
    fit_mle!(pcfg::ProbabilisticGrammar, pop::AbstractVector{RuleNode}; initial_value::Float64=0.0)

Update the probability vectors based on population using MLE.
"""
function fit_mle!(pcfg::ProbabilisticGrammar, pop::AbstractVector{RuleNode}; initial_value::Float64=0.0)
    fill!(pcfg, initial_value)
    for x in pop
        _fit_mle!(pcfg, x)
    end
    normalize!(pcfg)
    pcfg
end
function _fit_mle!(pcfg::ProbabilisticGrammar, x::RuleNode)
    grammar = pcfg.grammar
    typ = return_type(grammar, x)
    i = findfirst(grammar[typ], x.ind) 
    pcfg.probs[typ][i] += 1.0
    for c in x.children
        _fit_mle!(pcfg, c)
    end
    pcfg
end

"""
    Base.fill!(pcfg::ProbabilisticGrammar, x::Float64)

Fill all probability values to x
"""
function Base.fill!(pcfg::ProbabilisticGrammar, x::Float64)
    for v in values(pcfg.probs)
        fill!(v, x)
    end
end

"""
    Base.normalize!(pcfg::ProbabilisticGrammar)

Normalize all probability vectors so that each vector sums to 1.0
"""
function Base.normalize!(pcfg::ProbabilisticGrammar)
    for v in values(pcfg.probs)
        if sum(v) > 0.0
            normalize!(v, 1)
        else
            fill!(v, 1.0/length(v))
        end
    end
end


end # module
