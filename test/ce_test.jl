using ExprOptimization, ExprRules
using Test, Random

let
    grammar = @grammar begin
        R = R + R
        R = |(1:2)
    end

    function loss(node::RuleNode, grammar::Grammar)
        Core.eval(node, grammar)
    end

    Random.seed!(0)
    p = CrossEntropy(10, 5, 4, 5)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test Core.eval(res.tree, grammar) == 1
    @test res.loss == 1 

    iter = ExpressionIterator(grammar, 2, :R)
    pop = collect(iter)

    losses = Vector{Float64}(undef,length(pop))
    (best_tree, best_loss) = CrossEntropys.evaluate!(p, loss, grammar, pop, losses, pop[1], Inf)
    @test Core.eval(best_tree, grammar) == 1
    @test best_loss == 1

    p = CrossEntropy(10, 5, 4, 5; track_method=CrossEntropys.TopKTracking(3))
    res = optimize(p, grammar, :R, loss)
    res.alg_result[:top_k]
end

