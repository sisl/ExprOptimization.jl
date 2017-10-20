using ExprOptimization, ExprRules
using Base.Test

let
    grammar = @grammar begin
        R = R + R
        R = |(1:3)
    end

    function loss(node::RuleNode)
        eval(node, grammar)
    end

    srand(0)
    p = PIPEParams(PIPE.PPTParams(0.8),20,2,0.2,0.1,0.05,1,0.2,0.6,0.999,10)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test eval(res.tree, grammar) == 1
    @test res.loss == 1 

    iter = ExpressionIterator(grammar, 2, :R)
    pop = collect(iter)

    losses = Vector{Float64}(length(pop))
    (best_tree, best_loss) = PIPE.evaluate!(loss, pop, losses, pop[1], Inf)
    @test eval(best_tree, grammar) == 1
    @test best_loss == 1

    ppt = PIPE.PPTNode(p.ppt_params, grammar)
    PIPE.update!(p, ppt, grammar, pop[1], losses[1], best_loss) 
    PIPE.p_target(p.ppt_params, ppt, grammar, pop[1], losses[1], best_loss, p.α, p.ϵ)
    PIPE.mutate!(ppt, grammar, pop[1], p.p_mutation, p.β)
end

