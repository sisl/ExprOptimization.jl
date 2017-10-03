using ExprOptimization, ExprRules
using Base.Test

let
    grammar = @grammar begin
        R = R + R
        R = |(1:3)
    end

    p = PPTParams(0.75)
    ppt = PPT.PPTNode(p, grammar)
    
    @test PPT.nchildren(ppt) == 0

    PPT.get_child(p, ppt, grammar, 1)
    @test PPT.nchildren(ppt) == 1
    PPT.get_child(p, ppt, grammar, 2)
    @test PPT.nchildren(ppt) == 2

    @test all(isapprox.(PPT.probabilities(ppt, :R), [0.1, 0.3, 0.3, 0.3]; atol=0.01))

    r = rand(p, ppt, grammar, :R)
    PPT.probability(p, ppt, grammar, r)
    PPT.prune!(ppt, grammar, 0.999)
end

