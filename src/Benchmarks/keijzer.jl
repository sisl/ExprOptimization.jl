
function grammar_keijzer_1d(; erc=false)
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
function grammar_keijzer_2d(; erc=false)
    Base.eval(Main, :(protectedDiv(x, y) = Benchmarks.protectedDiv(x, y)))
    Base.eval(Main, :(protectedSin(x) = Benchmarks.protectedSin(x)))
    Base.eval(Main, :(protectedCos(x) = Benchmarks.protectedCos(x)))
    grammar = erc ? 
        (@grammar begin
            R = x
            R = y
            R = R + R
            R = R * R
            R = protectedDiv(R,R) 
            R = -R
            R = sqrt(abs(R))
            R = _(5*randn())
        end) : 
        (@grammar begin
            R = x
            R = y
            R = R + R
            R = R * R
            R = protectedDiv(R,R) 
            R = -R
            R = sqrt(abs(R))
            R = |(-5.0:1.0:5.0) 
        end)
    grammar
end

gt_keijzer_9(x::Float64) = log(x + sqrt(x^2+1)) #arcsinh(x)
const KEIJZER_9_XS = 0.0:1.0:100.0
const GT_KEIJZER_9 = (x=map(gt_keijzer_9, KEIJZER_9_XS); SVector{length(x)}(x))
const keijzer_9_scratch = MVector{length(GT_KEIJZER_9)}(GT_KEIJZER_9)

function loss_keijzer_9(S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    for i = 1:length(KEIJZER_9_XS) 
        S[:x] = GT_KEIJZER_9[i] 
        keijzer_9_scratch[i] = Core.eval(S,ex)::Float64
    end
    los = sum(abs.(keijzer_9_scratch .- GT_KEIJZER_9))
    los
end

function randexpr_keijzer_9()
    grammar = grammar_keijzer_1d(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    tree = rand(RuleNode, grammar, :R, 10)
    loss(tree::RuleNode, grammar::Grammar) = loss_keijzer_9(S, tree, grammar)
    return (tree=tree, grammar=grammar, S=S, loss=loss)
end

function run_keijzer_9(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_keijzer_1d(erc=true)
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

gt_keijzer_11(x::Float64, y::Float64) = x*y + sin((x-1)*(y-1))
const KEIJZER_11_XS = range(-3.0,stop=3.0,length=20)
const KEIJZER_11_YS = range(-3.0,stop=3.0,length=20)
const GT_KEIJZER_11 = let
    A = zeros(length(KEIJZER_11_XS)*length(KEIJZER_11_YS))
    i = 0
    for x in KEIJZER_11_XS
        for y in KEIJZER_11_YS
            A[i+=1] = gt_keijzer_11(x,y) 
        end
    end
    A
end
const VARS_KEIJZER_11 = let
    A = []
    for x in KEIJZER_11_XS 
        for y in KEIJZER_11_YS 
            push!(A, (x=x, y=y))
        end
    end
    A
end
using TimerOutputs
const to = TimerOutput() 
const keijzer_11_scratch = similar(GT_KEIJZER_11)
function loss_keijzer_11(S::SymbolTable, tree::RuleNode, grammar::Grammar)
    @timeit to "loss" begin
    @timeit to "getex" ex = get_executable(tree, grammar)
    @timeit to "eval loop" for i = 1:length(VARS_KEIJZER_11)
        keijzer_11_scratch[i] = Core.eval(S, ex, VARS_KEIJZER_11[i])::Float64
    end
    @timeit to "abs" keijzer_11_scratch .= abs.(keijzer_11_scratch .- GT_KEIJZER_11)
    @timeit to "sum" los = sum(keijzer_11_scratch)
end
    los 
end

function randexpr_keijzer_11()
    grammar = grammar_keijzer_2d(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    tree = rand(RuleNode, grammar, :R, 10)
    loss(tree::RuleNode, grammar::Grammar) = loss_keijzer_11(S, tree, grammar)
    return (tree=tree, grammar=grammar, S=S, loss=loss)
end

function run_keijzer_11(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_keijzer_2d(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_keijzer_11(S, tree, grammar)

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
