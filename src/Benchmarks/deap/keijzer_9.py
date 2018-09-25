import operator
import math
import random
import time
import csv

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    return 1.0 if right < 1e-10 else left / right 

def sqrtAbs(x):
    return math.sqrt(math.fabs(x))

pset = gp.PrimitiveSetTyped("main", [float], float)
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(sqrtAbs, [float], float)
pset.addEphemeralConstant("randn5", lambda: random.gauss(0.0,5.0), float)
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def gt(x):
    return math.log(x + math.sqrt(x**2+1)) 

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    los = 0.0
    n = 0
    for x in numpy.arange(0.0,100.0,1.0):
        los += (func(x) - gt(x))**2 
        n += 1
    return los / n, 

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

def runonce(i, n_pop):
    random.seed(i)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    
    #stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    #stats_size = tools.Statistics(len)
    #mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    #mstats.register("avg", numpy.mean)
    #mstats.register("std", numpy.std)
    #mstats.register("min", numpy.min)
    #mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, #stats=mstats,
                                   halloffame=hof, verbose=False)

    print str(hof[0])
    # print log
    return pop, log, hof

def main():
    n_seeds = 50
    with open('../results/deap_keijzer_9.csv', 'wb') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['system', 'problem', 'n_seeds', 'n_pop', 'mean_time_s', 'std_time_s', 'mean_fitness', 'std_fitness'])

        pop_sizes = [1000]
        for n_pop in pop_sizes:
            ts = []
            fitnesses = []
            for i in range(0, n_seeds):
                print "(n_pop, i) = " + "(" + str(n_pop) + ", " + str(i) + ")"
                tstart = time.clock()  #seconds
                pop, log, hof = runonce(i, n_pop)
                ts.append(time.clock() - tstart)
                fitnesses.append(hof[0].fitness.getValues()[0])
            mean_time_s = numpy.mean(ts)
            std_time_s = numpy.std(ts)
            mean_fitness = numpy.mean(fitnesses)
            std_fitness= numpy.std(fitnesses)
            w.writerow(['deap', 'keijzer_9', n_seeds, n_pop, mean_time_s, std_time_s, mean_fitness, std_fitness])

if __name__ == "__main__":
    main()
