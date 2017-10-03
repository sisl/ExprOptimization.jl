
module PIPE

using ExprRules, StatsBase
using ..PPT
using ..loss
using ..ExprOptParams
using ..ExprOptResults

import ..optimize

export PPTParams, PIPEParams

"""
    PIPEParams

Parameters for Probabilistic Incremental Program Evolution. Example parameters from paper are indicated in parentheses)
# Arguments:
- `ppt_params::PPTParams`: parameters for PPT  (e.g., [0.8, 0.2])
- `pop_size::Int`: population size 
- `iterations::Int`: number of iterations
- `p_elitist::Float64`: elitist update probability (e.g., 0.2)
- `c::Float64`: learning rate multiplier (e.g., 0.1)
- `α::Float64`: learning rate (e.g., 0.05) 
- `ϵ::Float64`: fitness constant (e.g., 1)
- `p_mutation::Float64`: mutation probability (e.g., 0.2)
- `β::Float64`: mutation rate (e.g., 0.6)
- `p_threshold::Float64`: prune threshold (e.g., 0.999)
- `max_depth::Int`: maximum depth of derivation tree
"""
struct PIPEParams <: ExprOptParams
    ppt_params::PPTParams
    pop_size::Int
    iterations::Int
    p_elitist::Float64
    c::Float64
    α::Float64
    ϵ::Float64
    p_mutation::Float64
    β::Float64
    p_threshold::Float64
    max_depth::Int
end

"""
    optimize(p::PIPEParams, grammar::Grammar, typ::Symbol)

Expression tree optimization using the PIPE algorithm with parameters p, grammar 'grammar', and start symbol typ.
"""
optimize(p::PIPEParams, grammar::Grammar, typ::Symbol) = pipe(p, grammar, typ)

"""
    pipe(p::PIPEParams, grammar::Grammar, typ::Symbol)

Probabilistic Incremental Program Evolution (PIPE) optimization algorithm with parameters p, grammar 'grammar', and start symbol typ..
Reference: R. Salustowicz and J. Schmidhuber, "Probabilistic Incremental Program Evolution", 
    Evolutionary Computation, vol. 5, no. 2, pp. 123-141, 1997.
"""
function pipe(p::PIPEParams, grammar::Grammar, typ::Symbol)
    iseval(grammar) && error("PIPE does not support _() functions in the grammar")

    best_tree, best_loss = RuleNode(0), Inf
    pp = p.ppt_params
    pop = Vector{RuleNode}(p.pop_size)
    losses = Vector{Float64}(p.pop_size)

    ppt = PPTNode(p.ppt_params, grammar)
    for i = 1:p.iterations
        if rand() < p.p_elitist #elitist learning
            update!(p, ppt, grammar, best_tree, best_loss, best_loss) 
        else #generational learning
            for j = 1:p.pop_size
                pop[j] = rand(pp, ppt, grammar, typ)
            end
            best_tree, best_loss = evaluate!(pop, losses, best_tree, best_loss)
            update!(p, ppt, grammar, pop[1], losses[1], best_loss) 
            mutate!(ppt, grammar, pop[1], p.p_mutation, p.β)
            prune!(ppt, grammar, p.p_threshold)
        end
    end
    ExprOptResults(best_tree, best_loss, get_executable(best_tree, grammar), nothing)
end

"""
    p_target(pp::PPTParams, ppt::PPTNode, grammar::Grammar, x_best::RuleNode, y_best::Float64, 
                  y_elite::Float64, α::Float64, ϵ::Float64)

Compute the target probability p_target.  See PIPE paper for description of equation.
"""
function p_target(pp::PPTParams, ppt::PPTNode, grammar::Grammar, x_best::RuleNode, y_best::Float64, 
                  y_elite::Float64, α::Float64, ϵ::Float64)
    p_best = probability(pp, ppt, grammar, x_best)
    return p_best + (1-p_best)*α*(ϵ - y_elite)/(ϵ - y_best)
end

"""
    update!(p::PIPEParams, ppt::PPTNode, grammar::Grammar, x::RuleNode, y::Float64, 
                 y_elite::Float64)

Update the ppt probabilities toward individual x with loss y given elite loss y_elite.
"""
function update!(p::PIPEParams, ppt::PPTNode, grammar::Grammar, x::RuleNode, y::Float64, 
                 y_elite::Float64)
    pp, c, α, ϵ = p.ppt_params, p.c, p.α, p.ϵ
    p_targ = p_target(pp, ppt, grammar, x, y, y_elite, α, ϵ)
    while probability(pp, ppt, grammar, x) < p_targ
        _update!(ppt, grammar, x, c, α)
    end
    return ppt
end
function _update!(ppt::PPTNode, grammar::Grammar, x::RuleNode, c::Float64, α::Float64)
    typ = return_type(grammar, x)
    i = findfirst(grammar[typ], x.ind)
    p = ppt.ps[typ]
    p[i] += c*α*(1-p[i])
    psum = sum(p)
    for j in 1 : length(p)
        if j != i
            p[j] *= (1- (1-psum)/(p[j]-psum))
        end
    end
    for (pptchild,xchild) in zip(ppt.children, x.children)
        _update!(pptchild, grammar, xchild, c, α)
    end
    return ppt
end

"""
    mutate!(ppt::PPTNode, grammar::Grammar, x_best::RuleNode, p_mutation::Float64, β::Float64;
        sqrtlen::Float64=sqrt(length(x_best)))

Mutate ppt node towards individual x_best using mutation probability p_mutation, mutation rate β using the 
square-root length criteria.
"""
function mutate!(ppt::PPTNode, grammar::Grammar, x_best::RuleNode, p_mutation::Float64, β::Float64;
    sqrtlen::Float64=sqrt(length(x_best)))

    typ = return_type(grammar, x_best)
    p = ppt.ps[typ]
    prob = p_mutation/(length(p)*sqrtlen)
    for i in 1 : length(p)
        if rand() < prob
            p[i] += β*(1-p[i])
        end
    end
    normalize!(p, 1)
    for (pptchild,xchild) in zip(ppt.children, x_best.children)
        mutate!(pptchild, grammar, xchild, p_mutation, β,
                sqrtlen=sqrtlen)
    end
    ppt
end

"""
    evaluate!(pop::Vector{RuleNode}, losses::Vector{Float64}, best_tree::RuleNode, best_loss::Float64)

Evaluate the loss function for population and sort.  Update the globally best tree.
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

end #module
