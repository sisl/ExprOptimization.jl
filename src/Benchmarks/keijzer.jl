
function grammar_keijzer(; erc=false)
    Base.eval(Main, :(protectedDiv(x, y) = Benchmarks.protectedDiv(x, y)))
    Base.eval(Main, :(protectedSin(x) = Benchmarks.protectedSin(x)))
    Base.eval(Main, :(protectedCos(x) = Benchmarks.protectedCos(x)))
    grammar = erc ? 
        (@grammar begin
            R = x
            R = R + R
            R = R * R
            R = protectedDiv(R,R) 
            R = -R
            R = sqrt(abs(R))
            R = _(5*randn())
        end) : 
        (@grammar begin
            R = x
            R = R + R
            R = R * R
            R = protectedDiv(R,R) 
            R = -R
            R = sqrt(abs(R))
            R = |(-5.0:1.0:5.0) 
        end)
    grammar
end

gt_keijzer_9(x) = log(x + sqrt(x^2+1)) #arcsinh(x)
function loss_keijzer_9(S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    n = 0
    for x in 0.0:1.0:100.0
        S[:x] = x
        los += abs2(Core.eval(S,ex) - gt_keijzer_9(x))
        n += 1
    end
    los / n
end

function run_keijzer_9(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_keijzer(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_keijzer_9(S, tree, grammar)

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
