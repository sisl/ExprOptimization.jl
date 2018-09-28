
function grammar_koza(; erc=false)
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
            R = protectedSin(R)
            R = protectedCos(R)
            R = exp(R)
            R = log(abs(R))
            R = _(2*rand()-1)
        end) : 
        (@grammar begin
            R = x
            R = R + R
            R = R - R
            R = R * R
            R = protectedDiv(R,R) 
            R = protectedSin(R)
            R = protectedCos(R)
            R = exp(R)
            R = log(abs(R))
            R = |(-1.0:0.25:1.0) 
        end)
    grammar
end

gt_koza_1(x::Float64) = x^4 + x^3 + x^2 + x 
gt_koza_2(x::Float64) = x^5 - 2x^3 + x 
gt_koza_3(x::Float64) = x^6 - 2x^4 + x^2 

function loss_koza(gt::Function, S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    rng = range(-1.0,stop=1.0,length=20) 
    for x in rng
        S[:x] = x
        los += abs2(Core.eval(S,ex)::Float64 - gt(x)::Float64)
    end
    los / length(rng)
end

function run_koza_1(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_koza(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_koza(gt_koza_1, S, tree, grammar)

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
