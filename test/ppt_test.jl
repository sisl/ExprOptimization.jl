using ExprOptimization
using Base.Test

let
    grammar = @grammar begin
        R = |(1:3)
        R = R + R
    end

    pp = PIPE.PPTParams(0.5)
    ppt = PIPE.PPTNode(pp, grammar)
    expr = RuleNode(1) 
    @test PIPE.probability(pp, ppt, grammar, expr) == 0.25
    expr = RuleNode(4, [RuleNode(1), RuleNode(3)])
    @test PIPE.probability(pp, ppt, grammar, expr) == 0.25^3
end

