using ExprOptimization
using Test, Random

let
    grammar = @grammar begin
        R = |(1:3)
        R = R + R
    end

    function loss(node::RuleNode, grammar::Grammar)
        Core.eval(node, grammar)
    end

    Random.seed!(0)
    p = MonteCarlo(20, 5)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test Core.eval(res.tree, grammar) == 1
    @test res.loss == 1 

    p = MonteCarlo(20, 5; track_method=MonteCarlos.TopKTracking(3))
    res = optimize(p, grammar, :R, loss)
    res.alg_result[:top_k]
end

