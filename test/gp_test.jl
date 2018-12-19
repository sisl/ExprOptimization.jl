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
    p = GeneticProgram(10, 5, 4, 0.3, 0.3, 0.3)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test Core.eval(res.tree, grammar) == 1
    @test res.loss == 1 

    iter = ExpressionIterator(grammar, 2, :R)
    pop = collect(iter)

    losses = Vector{Union{Float64,Missing}}(undef,length(pop))
    (best_tree, best_loss) = GeneticPrograms.evaluate!(p, loss, grammar, pop, losses, pop[1], Inf)
    @test Core.eval(best_tree, grammar) == 1
    @test best_loss == 1

    dmap = mindepth_map(grammar)
    GeneticPrograms.select(p.select_method, pop, losses)
    GeneticPrograms.crossover(pop[1], pop[2],  grammar)
    GeneticPrograms.mutation(pop[3], grammar, dmap)

    p = GeneticProgram(10, 5, 4, 0.3, 0.3, 0.3; track_method=GeneticPrograms.TopKTracking(3))
    res = optimize(p, grammar, :R, loss)
    res.alg_result[:top_k]
end

