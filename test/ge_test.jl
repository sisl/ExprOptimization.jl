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
    p = GrammaticalEvolution(grammar, :R, 10, 5, 5, 5, 4, 0.2, 0.4, 0.4)
    res = optimize(p, grammar, :R, loss)
    @test res.expr == 1
    @test Core.eval(res.tree, grammar) == 1
    @test res.loss == 1 

    pop = GrammaticalEvolutions.initialize(p.pop_size, p.init_gene_length)
    node = GrammaticalEvolutions.decode(pop[1], grammar, :R).node

    losses = Vector{Union{Float64,Missing}}(undef,length(pop))
    (best_tree, best_loss) = GrammaticalEvolutions.evaluate!(p, grammar, :R, loss, pop, losses, node, Inf)
    @test Core.eval(best_tree, grammar) == 1
    @test best_loss == 1

    GrammaticalEvolutions.select(p.select_method, pop, losses)
    GrammaticalEvolutions.crossover(pop[1], pop[2])
    GrammaticalEvolutions.mutation(GrammaticalEvolutions.MultiMutate(grammar, :R), pop[3])
end

