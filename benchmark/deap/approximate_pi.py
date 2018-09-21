#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

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
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSetTyped("main", [], float)
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)
pset.addTerminal(1.0, float)
pset.addTerminal(2.0, float)
pset.addTerminal(3.0, float)
pset.addTerminal(4.0, float)
pset.addTerminal(5.0, float)
pset.addTerminal(6.0, float)
pset.addTerminal(7.0, float)
pset.addTerminal(8.0, float)
pset.addTerminal(9.0, float)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    value = toolbox.compile(expr=individual)
    d = numpy.abs(value - math.pi) 
    return math.log(d) + 1e-4 * len(individual),

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))

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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50, #stats=mstats,
                                   halloffame=hof, verbose=False)

    print str(hof[0])
    # print log
    return pop, log, hof

def main():
    n_seeds = 50
    with open('../deap_approximate_pi.csv', 'wb') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['system', 'problem', 'n_seeds', 'n_pop', 'mean_time_s', 'std_time_s', 'mean_fitness', 'std_fitness'])

        pop_sizes = [1000]
        for n_pop in pop_sizes:
            ts = []
            fitnesses = []
            for i in range(0, n_seeds):
                tstart = time.clock()  #seconds
                pop, log, hof = runonce(i, n_pop)
                ts.append(time.clock() - tstart)
                fitnesses.append(hof[0].fitness.getValues()[0])
            mean_time_s = numpy.mean(ts)
            std_time_s = numpy.std(ts)
            mean_fitness = numpy.mean(fitnesses)
            std_fitness= numpy.std(fitnesses)
            w.writerow(['deap', 'approximate_pi', n_seeds, n_pop, mean_time_s, std_time_s, mean_fitness, std_fitness])

if __name__ == "__main__":
    main()
