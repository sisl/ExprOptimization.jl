function grammar_vladislavleva_c(; erc=false)
    Base.eval(Main, :(protectedDiv(x, y) = Benchmarks.protectedDiv(x, y)))
    Base.eval(Main, :(protectedSin(x) = Benchmarks.protectedSin(x)))
    Base.eval(Main, :(protectedCos(x) = Benchmarks.protectedCos(x)))
    grammar = erc ? 
        (@grammar begin
            R = x
            R = R + R
            R = R - R
            R = R * R
            R = protectedDiv(R,R) 
            R = R^2
            R = exp(R)
            R = exp(-R)
            R = protectedSin(R)
            R = protectedCos(R)
            R = abs(R)^E
            R = R + E
            R = R * E
            E = _(5*rand())
        end) : 
        (@grammar begin
            R = x
            R = R + R
            R = R - R
            R = R * R
            R = protectedDiv(R,R) 
            R = R^2
            R = exp(R)
            R = exp(-R)
            R = protectedSin(R)
            R = protectedCos(R)
            R = abs(R)^E
            R = R + E
            R = R * E
            E = |(-5.0:1.0:5.0) 
        end)
    grammar
end


gt_vladislavleva_3(x,y) = exp(-x)*x^3*(protectedCos(x)*protectedSin(x))*(protectedCos(x)*protectedSin(x)^2-1)*(y-5)
function loss_vladislavleva_3(S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    n = 0
    for x in 0.05:0.1:10.0, y in 0.05:2.0:10.05
        S[:x] = x
        S[:y] = y
        los += abs2(Core.eval(S,ex) - gt_vladislavleva_3(x))
        n += 1
    end
    los / n
end

function run_vladislavleva_3(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_vladislavleva_c(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_vladislavleva_3(S, tree, grammar)

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
