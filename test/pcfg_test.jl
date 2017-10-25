using ExprOptimization, ExprRules
using Base.Test

let
    grammar = @grammar begin
        R = R + R
        R = |(1:3)
    end

    pcfg = ProbabilisticExprRules.ProbabilisticGrammar(grammar)
    @test ProbabilisticExprRules.probabilities(pcfg, :R) == normalize(ones(length(grammar[:R])), 1)

    dmap = mindepth_map(grammar)
    r = rand(RuleNode, pcfg, :R, dmap)

    iter = ExpressionIterator(grammar, 2, :R)
    pop = collect(iter)

    ProbabilisticExprRules.fit_mle!(pcfg, pop)
    @test all(isapprox.(ProbabilisticExprRules.probabilities(pcfg, :R), [0.3, 0.23333, 0.23333, 0.23333]; atol=0.001))

    ProbabilisticExprRules.uniform!(pcfg)
    @test ProbabilisticExprRules.probabilities(pcfg, :R) == normalize(ones(length(grammar[:R])), 1)
end

