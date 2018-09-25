
function grammar_simple_calculator(; erc=false)
    grammar = erc ? 
        (@grammar begin
            R = R + R
            R = R - R
            R = R * R
            R = R / R
            R = _(rand(1:9))
        end) : 
        (@grammar begin
            R = R + R
            R = R - R
            R = R * R
            R = R / R
            R = |(1:9)
        end)
end


function loss_approximate_pi(S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    value = Core.eval(S, ex)
    if isinf(value) || isnan(value)
        return Inf
    end
    Δ = abs(value - π)
    return log(Δ) + length(tree) / 1e4
end

function run_approximate_pi(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_simple_calculator(erc=false)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_approximate_pi(S, tree, grammar)

    init_method = RandomInit()
    select_method = TournamentSelection(3)
    p = GeneticProgram(n_pop, #pop_size
                       40,  #iterations
                       10,  #max_depth
                       0.4, #p_reproduction
                       0.5, #p_crossover
                       0.1, #p_mutation 
                       init_method=init_method, 
                       select_method=select_method)
    tstart = CPUtime_us()
    result = optimize(p, grammar, :R, loss)
    dt = CPUtime_us() - tstart
    result, dt
end
