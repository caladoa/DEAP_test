# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:23:30 2022

@author: andre
"""
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import algorithms
import pickle
import math
#parameters for genetic algorithm
Npopulation = 2500 #size of pop.
Ngeneration = 50 #number of generations

x_reg_ooc = np.load('x_reg_unscaled.npy')
angles = np.load('X_SR.npy')
points = np.hstack((angles, np.repeat(x_reg_ooc,100,axis=0)))
CL_data = np.load('CL_data.npy')
CL_vec = CL_data.flatten(order='F')

#%% Choose primitive set

#terminal set  -  phi, phid, phidd, alpha, alphad, alphadd
pset = gp.PrimitiveSet("MAIN", 11)
pset.renameArguments(ARG0='phi',
                     ARG1='phid',
                     ARG2='phidd',
                     ARG3='alpha',
                     ARG4='alphad', 
                     ARG5='alphadd', 
                     ARG6='Re', 
                     ARG7 = 'k', 
                     ARG8='A_alpha',
                     ARG9='K_phi',
                     ARG10='K_alpha')

'''
try: 
    evaluate ind 
except Nan: 
    fitness = -1e5


'''

#terminal set - add ephemeral constant
# name = 'rand'+str(random.randint(0,10000))
# pset.addEphemeralConstant(name,lambda: random.uniform(-1,1))

#terminal set - add a constant of value 1.0
#pset.addTerminal(1.0)

# #function set - add operators
# pset.addPrimitive(operator.add, 2) #addition
# pset.addPrimitive(operator.sub, 2) #subtraction
# pset.addPrimitive(operator.mul, 2) #multiplication
#
# pset.addPrimitive(np.sin, 1) #sin
# pset.addPrimitive(np.cos, 1) #cos
# pset.addPrimitive(np.power, 2) #power

#protected divide by zero
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.tanh, 1)
# pset.addPrimitive(math.pow, 1)
#pset.addPrimitive(protectedDiv, 2)

#%% Define individual

# This is the way DEAP defines individuals for genetic programing
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #you define this as a minimisation problem
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #the individual are primite trees


toolbox = base.Toolbox()


# -> here you can change how the population is initialized try genFull or genGrow for example (slide 13)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5) #  <- here

#specific code for gp in deap. basically copy and paste from deap tutorial
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

#%% Define fitness function

def MSE_coef(individual, points, y): #take individual function, reg. inputs and weights to find
    #Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    sqerr = np.zeros(len(points))
    #for ii, pt in enumerate(points):
    try:
        for ii in range(len(points)):
            #sqerr[ii] = (func(points[ii,0],points[ii,1],points[ii,2],points[ii,3],points[ii,4],points[ii,5],points[ii,6],points[ii,7],points[ii,8],points[ii,9],points[ii,10])-y[ii])**2
            sqerr[ii] = (func(*points[ii,:])-y[ii])**2

            #sqerr[ii] = (func(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5])-y[ii])**2
        fitness = np.sum(sqerr)/len(points)
    except (ValueError, RuntimeError) or math.isnan(fitness) or math.isinf(fitness):
        fitness = 1e2

    return fitness,




# we apply the fitness in an operation called evaluate where we compute the error for each points 
toolbox.register("evaluate", MSE_coef, points=points, y=CL_vec)

#%% Define selection and mutation parameters

#selection tournament 
# -> here you can change the selection process (slide 14)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=3, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# -> here you can change the probability for the cross-over (slide 15)
# toolbox.register("mate", gp.cxOnePoint)
cxpb = 0.4 # <- here 

# -> here you can change the probability for the mutation (slide 16) 
# toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
mutpb = 0.2 # <- here  

# -> here you can change the probability for the shrink (slide 17) 
# toolbox.register("shrink", gp.mutShrink)
mutsh = 0.05 # <- here 

# -> here you can change the probability for the mutation of ephemeral constants
toolbox.register("mut_eph", gp.mutEphemeral, mode='all') 
mut_eph = 0.05 # <- here

    
pop = toolbox.population(n=Npopulation) #generate initial population
hof = tools.HallOfFame(1) #generate elite (save best individual of the population)


if __name__ == '__main__':
    N_PROCESSES = 4
    pool = mp.Pool(processes=N_PROCESSES)
    toolbox.register("map", pool.map)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", np.min)
    mstats.register("mean", np.mean)
    mstats.register("std", np.std)
    stats=mstats

    pop, log = algorithms.eaMuCommaLambda(pop, toolbox, int(0.4*Npopulation), int(0.5*Npopulation), cxpb, mutpb,
                                         Ngeneration, halloffame=hof,stats=stats, verbose=True)

    #%% Display end solution
    bestfunct = toolbox.compile(expr=hof[0])

    print(hof[0])
    # --- Saving the best individual, hof to be compiled and used afterwards
    with open('./hof.pickle', 'wb') as fp:
        pickle.dump(hof, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./logs.pickle', 'wb') as fp1:
        pickle.dump(log, fp1, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pop.pickle', 'wb') as fp2:
        pickle.dump(pop, fp2, protocol=pickle.HIGHEST_PROTOCOL)

    pool.close()
    #%%
    y_pred = bestfunct(points[:,0],points[:,1],points[:,2],points[:,3],points[:,4],points[:,5],points[:,6],points[:,7],points[:,8],points[:,9],points[:,10])
    plt.plot(CL_vec,CL_vec,'--')
    plt.plot(CL_vec, y_pred,'.')
