# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:23:30 2022

@author: andre
"""
import operator
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import algorithms

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
name = 'rand'+str(random.randint(0,10000))
pset.addEphemeralConstant(name,lambda: random.uniform(-1,1))

#terminal set - add a constant of value 1.0
#pset.addTerminal(1.0)

#function set - add operators
pset.addPrimitive(operator.add, 2) #addition
pset.addPrimitive(operator.sub, 2) #subtraction
pset.addPrimitive(operator.mul, 2) #multiplication

pset.addPrimitive(np.sin, 1) #sin
pset.addPrimitive(np.cos, 1) #cos
pset.addPrimitive(np.power, 2) #power

#protected divide by zero
def protectedDiv(left, right):
    try:
        return left / right
    
    except ZeroDivisionError:
        return 1

#pset.addPrimitive(protectedDiv, 2)

#%% Define individual

# This is the way DEAP defines individuals for genetic programing
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #you define this as a minimisation problem
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #the individual are primite trees


toolbox = base.Toolbox()


# -> here you can change how the population is initialized try genFull or genGrow for example (slide 13)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5) #  <- here 

#specific code for gp in deap. basically copy and paste from deap tutorial
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

#%% Define fitness function

def MSE_coef(individual, points, y): #take individual function, reg. inputs and weights to find
    #Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    #print(func)
    
    #try:
        
    #evaluate mean squared error between the expression
    sqerr = np.zeros(len(points))
    #for ii, pt in enumerate(points):
    for ii in range(len(points)):
        #sqerr[ii] = (func(points[ii,0],points[ii,1],points[ii,2],points[ii,3],points[ii,4],points[ii,5],points[ii,6],points[ii,7],points[ii,8],points[ii,9],points[ii,10])-y[ii])**2
        sqerr[ii] = (func(*points[ii,:])-y[ii])**2

        #sqerr[ii] = (func(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5])-y[ii])**2
    fitness = np.sum(sqerr)/len(points)


        
    if np.isnan(fitness) or np.isinf(fitness):
        fitness = 1e2
            
    # except np.isnan(fitness):
    #     fitness = 1e2
    #    
                
    return fitness,




# we apply the fitness in an operation called evaluate where we compute the error for each points 
toolbox.register("evaluate", MSE_coef, points=points, y=CL_vec)

#%% Define selection and mutation parameters

#selection tournament 
# -> here you can change the selection process (slide 14)
toolbox.register("select", tools.selTournament, tournsize=5)# <- here 

# -> here you can change the probability for the cross-over (slide 15)
toolbox.register("mate", gp.cxOnePoint)
cxpb = 0.4 # <- here 

# -> here you can change the probability for the mutation (slide 16) 
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2) 
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
mutpb = 0.2 # <- here  

# -> here you can change the probability for the shrink (slide 17) 
toolbox.register("shrink", gp.mutShrink) 
mutsh = 0.05 # <- here 

# -> here you can change the probability for the mutation of ephemeral constants
toolbox.register("mut_eph", gp.mutEphemeral, mode='all') 
mut_eph = 0.05 # <- here






#%% GA loop

import multiprocessing

# Initial population size


#try eaMuPlusLambda , need to define mu (# ind to select for next gen) and lambda (# children to produce each gen)
#generally mu < lambda for convergence of optimal solution if using mu,lambda 


#pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, Ngeneration, halloffame=hof,stats=stats, verbose=True)

# if __name__ == '__main__':
    
#     pool = multiprocessing.Pool(4)
#     toolbox.register("map", pool.map)
    
pop = toolbox.population(n=Npopulation) #generate initial population
hof = tools.HallOfFame(1) #generate elite (save best individual of the population)

mu = int(0.4*Npopulation)
lambda_ = int(0.6*Npopulation)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("min", np.min)
mstats.register("mean", np.mean)
mstats.register("std", np.std)
stats=mstats

pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, cxpb, mutpb, Ngeneration, halloffame=hof,stats=stats, verbose=True)
    
    # pool.close()



# pool.close()
    
# # this is some esthetic copy and paste from the DEAP example to compute the stat at each generation


# #first evaluation
# logbook = tools.Logbook()
# logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

# # Evaluate the individuals with an invalid fitness 
# # this is a small trick to only evaluate individuals that have been mutated
# invalid_ind = [ind for ind in pop if not ind.fitness.valid]
# fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
# for ind, fit in zip(invalid_ind, fitnesses):
#     ind.fitness.values = fit
    
# if hof is not None:
#     hof.update(pop)

# # this is some esthetic copy and paste from the DEAP example to show the stats at each generation
# record = stats.compile(pop) if stats else {}
# logbook.record(gen=0, nevals=len(invalid_ind), **record)
# print(logbook.stream)

# # evolution of elites we record the best individual at each generation 
# save_elite = []

# # Begin the generational process
# for gen in range(1, Ngeneration + 1):
#     # Select the next generation individuals
#     select = toolbox.select(pop, len(pop)-1)
#       # Select the best individual and save it in elite. This is elitism
#     elites = tools.selBest(pop, k=1)
#     save_elite.append(toolbox.clone(elites))
    
#     #clone offspring
#     off = [toolbox.clone(ind) for ind in select]
    
#     # Apply mutation
#     #this is to make sure we apply only one mutation to each individual (slide 18)
#     cumpb = np.cumsum([cxpb,mutsh,mutpb,mut_eph])
#     for i in range(len(off)):
#         pb = random.random()
#         if pb<cumpb[0] and i>0:
#             off[i-1], off[i] = toolbox.mate(off[i-1],off[i])
#             del off[i-1].fitness.values, off[i].fitness.values
#             # print('off N=%i, crossover'%(i))
#         elif pb<cumpb[1]:
#             off[i], = toolbox.shrink(off[i])
#             del off[i].fitness.values
#             #print('off N=%i, schrink'%(i))
#         elif pb<cumpb[2]:
#             off[i], = toolbox.mutate(off[i])
#             del off[i].fitness.values
#         elif pb<cumpb[3]:
#             off[i], = toolbox.mut_eph(off[i])
#             del off[i].fitness.values
#             #print('off N=%i, mutate'%(i))
#         # else:
#         #     print('off N=%i, reproduce'%(i))
    

#     # compute the fitness of the individuals with an invalid fitness
#     invalid_ind = [ind for ind in off if not ind.fitness.valid]
#     fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit

#     # Update the hall of fame with the generated individuals
#     if hof is not None:
#         hof.update(off)

#     # Replace the current population by the offspring and the elite
#     pop[:] = off + elites

#     # Append the current generation statistics to the logbook
#     record = stats.compile(pop) if stats else {}

#     logbook.record(gen=gen, nevals=len(invalid_ind), 
#                     lmean = record['size']['mean'], fmin =  record['fitness']['min'],
#                     **record)
    
#     #print
#     print(logbook.stream)
    

#%% Display end solution
bestfunct = toolbox.compile(expr=hof[0])

print(hof[0])

#%%
y_pred = bestfunct(points[:,0],points[:,1],points[:,2],points[:,3],points[:,4],points[:,5],points[:,6],points[:,7],points[:,8],points[:,9],points[:,10])
plt.plot(CL_vec,CL_vec,'--')
plt.plot(CL_vec, y_pred,'.')
