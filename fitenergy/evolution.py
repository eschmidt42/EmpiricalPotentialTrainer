from deap import base, creator, tools
import numpy as np
import random, copy

def measure(x,y):
    return (np.linalg.norm(x-y)**2/x.shape[0],)

def eval_fun(x,structures=None,measure=None,references=None):
    
    values = np.zeros(len(structures))
    num_s = len(structures)
    
    for i in range(num_s):
        structures[i].update_params(x)
    
    for i in range(num_s):
        values[i] = structures[i].get_energy()
    return measure(values,[v.energy for v in structures])

def evo_fun(structures,num_paras,ex_prob=0.25,mut_prob=0.25,mu=0,sigma=1.,ngen=100,npop=25,lb=-50,ub=50,indpb=.25):
    """
    
    Parameters
    ----------
    
    structures : list of structures_template instances
    references : ndarray
        reference energies from DFT
        
    Returns
    -------
    
    pops :
    fits :
    records :
    
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_flt", random.uniform, lb, ub)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_flt, num_paras)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", eval_fun, structures=structures, measure=measure)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print('Evolving ...')
    records = []
    
    pop = toolbox.population(n=npop)  
    pops = [copy.deepcopy(pop)]
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for g in range(ngen):
        print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < ex_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # mutate
        for m in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(m)
                del m.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind) #parallelization?
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        record = stats.compile(pop)
        records += [record]
        pops += [copy.deepcopy(offspring)]
    return pops, records, fits