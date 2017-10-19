using ExprOptimization, ExprRules
using Base.Test

let
    grammar = @grammar begin
        R = R + R
        R = |(1:2)
    end

    function ExprOptimization.loss(node::RuleNode)
        eval(node, grammar)
    end

    srand(0)
    p = GrammaticalEvolutionParams(grammar, :R, 10, 5, 5, 4, 0.2, 0.4, 0.4)
    res = optimize(p, grammar, :R)
    @test res.expr == 1
    @test eval(res.tree, grammar) == 1
    @test res.loss == 1 

    pop = GrammaticalEvolution.initialize(p.pop_size, p.gene_length)
    node = GrammaticalEvolution.decode(pop[1], grammar, :R).node

    losses = Vector{Float64}(length(pop))
    (best_tree, best_loss) = GrammaticalEvolution.evaluate!(p, grammar, :R, pop, losses, node, Inf)
    @test eval(best_tree, grammar) == 1
    @test best_loss == 1

    GrammaticalEvolution.select(p.select_method, pop, losses)
    GrammaticalEvolution.crossover(pop[1], pop[2])
    GrammaticalEvolution.mutation(GrammaticalEvolution.MultiMutate(grammar, :R), pop[3])
end

