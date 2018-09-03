using ExprOptimization, ExprRules
using Test, Random

let
    grammar = @grammar begin
        R = R + R
        R = |(1:3)
    end

    function loss(node::RuleNode, grammar::Grammar)
        Core.eval(node, grammar)
    end

    Random.seed!(0)
    p = PIPE(PPT(0.8),20,2,0.2,0.1,0.05,1,0.2,0.6,0.999,10)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test Core.eval(res.tree, grammar) == 1
    @test res.loss == 1 

    iter = ExpressionIterator(grammar, 2, :R)
    pop = collect(iter)

    losses = Vector{Float64}(undef,length(pop))
    (best_tree, best_loss) = PIPEs.evaluate!(loss, grammar, pop, losses, pop[1], Inf)
    @test Core.eval(best_tree, grammar) == 1
    @test best_loss == 1

    ppt = PIPEs.PPTNode(p.ppt_params, grammar)
    PIPEs.update!(p, ppt, grammar, pop[1], losses[1], best_loss) 
    PIPEs.p_target(p.ppt_params, ppt, grammar, pop[1], losses[1], best_loss, p.α, p.ϵ)
    PIPEs.mutate!(ppt, grammar, pop[1], p.p_mutation, p.β)
end

