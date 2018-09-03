using ExprOptimization, ExprRules
using Test

let
    grammar = @grammar begin
        R = R + R
        R = |(1:3)
    end

    p = PPT(0.75)
    ppt = PPTs.PPTNode(p, grammar)
    
    @test PPTs.nchildren(ppt) == 0

    PPTs.get_child(p, ppt, grammar, 1)
    @test PPTs.nchildren(ppt) == 1
    PPTs.get_child(p, ppt, grammar, 2)
    @test PPTs.nchildren(ppt) == 2

    @test all(isapprox.(PPTs.probabilities(ppt, :R), [0.1, 0.3, 0.3, 0.3]; atol=0.01))

    r = rand(p, ppt, grammar, :R)
    PPTs.probability(p, ppt, grammar, r)
    PPTs.prune!(ppt, grammar, 0.999)
end

