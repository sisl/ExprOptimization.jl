module Benchmarks

using ExprOptimization
using Random, CPUTime, DataFrames, CSV, Statistics
using ExprOptimization.GeneticPrograms: RandomInit, TournamentSelection

const RESULTSDIR = joinpath(@__DIR__, "results")
isdir(RESULTSDIR) || mkpath(RESULTSDIR)

protectedDiv(x, y) = iszero(y) ? 1.0 : x / y
protectedSin(x) = isinf(x) ? 0.0 : sin(x)
protectedCos(x) = isinf(x) ? 1.0 : cos(x)

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
            R = R^E
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
            R = R^E
            R = R + E
            R = R * E
            R = |(-5.0:1.0:5.0) 
        end)
    grammar
end

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

gt_koza_1(x) = x^4 + x^3 + x^2 + x 
gt_koza_2(x) = x^5 - 2x^3 + x 
gt_koza_3(x) = x^6 - 2x^4 + x^2 

function loss_koza(gt::Function, S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    n = 0
    for x in range(-1.0,stop=1.0,length=20) 
        S[:x] = x
        los += abs2(Core.eval(S,ex) - gt(x))
        n += 1
    end
    los / n
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

gt_vladislavleva_3(x,y) = exp(-x)*x^3*(protectedCos(x)*protectedSin(x))*(protectedCos(x)*protectedSin(x)^2-1)*(y-5)
function loss_vladislavleva_3(S::SymbolTable, tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    los = 0.0
    n = 0
    for x in 0.05:0.1:10.0, y in 0.05:2.0:10.05
        S[:x] = x
        S[:y] = y
        los += abs2(Core.eval(S,ex) - gt_keijzer_9(x))
        n += 1
    end
    los / n
end

function loss_approximate_pi(tree::RuleNode, grammar::Grammar)
    ex = get_executable(tree, grammar)
    value = Core.eval(S, ex)
    if isinf(value) || isnan(value)
        return Inf
    end
    Δ = abs(value - π)
    return log(Δ) + length(tree) / 1e4
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
function run_vladislavleva_3(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_vladislavleva_c(erc=true)
    S = SymbolTable(grammar, ExprOptimization.Benchmarks)
    loss(tree::RuleNode, grammar::Grammar) = loss_vladislavleva(S, tree, grammar)

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
function run_approximate_pi(seed::Int=0, n_pop::Int=300)
    Random.seed!(seed)

    grammar = grammar_simple_calculator(erc=true)
    loss(tree::RuleNode, grammar::Grammar) = loss_approximate_pi(tree, grammar)

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

function log_df()
    df = DataFrame([String,String,Int,Int,Float64,Float64,Float64,Float64], 
                    [:system,:problem,:n_seeds,:n_pop,:mean_time_s,:std_time_s,:mean_fitness,:std_fitness], 0)
end

main_koza_1() = main_timing(run_koza_1, 50, [1000], "koza_1", "exproptimization_koza_1.csv")
main_keijzer_9() = main_timing(run_keijzer_9, 50, [1000], "keijzer_9", "exproptimization_keijzer_9.csv")
main_vladislavleva_3() = main_timing(run_vladislavleva_3, 50, [1000], "vladislavleva_3", "exproptimization_vladislavleva_3.csv")
main_approximate_pi() = main_timing(run_approximate_pi, 50, [1000], "approximatE_pi", "exproptimization_approximate_pi.csv")

function main_timing(f::Function, n_seeds, pop_sizes, problem::String, outfile::String)
    f(0,100)  #precompile

    df = log_df()
    for n_pop in pop_sizes
        times = Float64[]
        fitnesses = Float64[]
        for i = 1:n_seeds
            @show n_pop, i
            result, dt = f(i, n_pop)
            push!(times, dt)
            push!(fitnesses, result.loss)
        end
        ts = times .* 1e-6 #convert to seconds
        push!(df, ["ExprOptimization",problem,n_seeds,n_pop,mean(ts),std(ts),mean(fitnesses),std(fitnesses)])
        CSV.write(joinpath(RESULTSDIR,outfile), df)
    end
end

function load_csvs()
    fs = filter(x->endswith(x,".csv"), readdir(RESULTSDIR))
    df = vcat([CSV.read(joinpath(RESULTSDIR,f)) for f in fs]...)
    df
end

function main()
    main_koza_1()
    main_keijzer_9()
    main_vladislavleva_3()
    main_approximate_pi()
end


end #module
