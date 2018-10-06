
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
        end)
    grammar
end

gt_koza_1(x::Float64) = x^4 + x^3 + x^2 + x 
gt_koza_2(x::Float64) = x^5 - 2x^3 + x 
gt_koza_3(x::Float64) = x^6 - 2x^4 + x^2 
const KOZA_XS = range(-1.0,stop=1.0,length=20) 
const GT_KOZA_1 = map(gt_koza_1, KOZA_XS)
const GT_KOZA_2 = map(gt_koza_2, KOZA_XS)
const GT_KOZA_3 = map(gt_koza_3, KOZA_XS)
const koza_scratch = similar(GT_KOZA_1)

loss_koza_1(S::SymbolTable, tree::RuleNode, grammar::Grammar) = loss_koza(GT_KOZA_1, S, tree, grammar)
loss_koza_2(S::SymbolTable, tree::RuleNode, grammar::Grammar) = loss_koza(GT_KOZA_2, S, tree, grammar)
loss_koza_3(S::SymbolTable, tree::RuleNode, grammar::Grammar) = loss_koza(GT_KOZA_3, S, tree, grammar)
function loss_koza(gt_vec::Vector{Float64}, S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    for i = 1:length(KOZA_XS)
        S[:x] = KOZA_XS[i] 
        koza_scratch[i] = Core.eval(S, ex)::Float64
    end
    los = sum(abs.(koza_scratch .- gt_vec))
    los
end

function run_koza_1(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_koza(erc=false)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_koza_1(S, tree, grammar)

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
