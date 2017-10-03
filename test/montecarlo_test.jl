using ExprOptimization
using Base.Test

let
    grammar = @grammar begin
        R = |(1:3)
        R = R + R
    end

    function ExprOptimization.loss(node::RuleNode)
        eval(node, grammar)
    end

    srand(0)
    p = MonteCarloParams(20, 5)
    res = optimize(p, grammar, :R)
    @test res.expr == 1
    @test eval(res.tree, grammar) == 1
    @test res.loss == 1 
end

