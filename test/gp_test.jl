using ExprOptimization, ExprRules
using Base.Test

let
    grammar = @grammar begin
        R = R + R
        R = |(1:2)
    end

    function loss(node::RuleNode)
        eval(node, grammar)
    end

    srand(0)
    p = GeneticProgramParams(10, 5, 4, 0.3, 0.3, 0.3)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test eval(res.tree, grammar) == 1
    @test res.loss == 1 

    iter = ExpressionIterator(grammar, 2, :R)
    pop = collect(iter)

    losses = Vector{Float64}(length(pop))
    (best_tree, best_loss) = GeneticProgram.evaluate!(loss, pop, losses, pop[1], Inf)
    @test eval(best_tree, grammar) == 1
    @test best_loss == 1

    GeneticProgram.select(p.select_method, pop, losses)
    GeneticProgram.crossover(pop[1], pop[2],  grammar)
    GeneticProgram.mutation(pop[3], grammar)
end

