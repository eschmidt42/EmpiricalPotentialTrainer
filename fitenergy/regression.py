import numpy as np
from scoop import shared, futures
from deap import base, creator, tools, cma
from collections import deque
import warnings

from fitenergy.misc import setup_weights, setup_weights_pair, setup_weights_pair_embmod
from fitenergy.potential import calculate_all_energies, calculate_all_forces, calculate_all_forces_from_splines

def var(k,key=None):
    return shared.getConst(k) if key is None else shared.getConst(k)[key]

def Potty_energies_and_forces(x,force_analytic=True): # for pure systems
	tmp_x = np.array(x,dtype=float)
		
	W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x) #fe.nonlin.setup_weights(tmp_x,params,mapper)
	W_pair_neigh = W_pair

	energies = calculate_all_energies(W_emb,W_pair)

	if force_analytic:
		forces = calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh)
	else:
		forces = calculate_all_forces_from_splines(tmp_x)
	return energies, forces
	
def Potty_energies_and_forces_pair(x,force_analytic=True,embmod=False): # for beyond pure systems
	tmp_x = np.array(x,dtype=float)
		
	if embmod:
		W_emb,W_pair,W_emb_neigh = setup_weights_pair_embmod(tmp_x)
	else:
		W_pair = setup_weights_pair(tmp_x)
		W_emb, W_emb_neigh = var("W_emb_c"), var("W_emb_neigh_c")
	W_pair_neigh = W_pair

	energies = calculate_all_energies(W_emb,W_pair)

	if force_analytic:
		forces = calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh)
	else:
		forces = calculate_all_forces_from_splines(tmp_x)
	return energies, forces

def log_joint(x,alphas,betas,negative=True,force_analytic=True,t=None):
    """Calculates the log joint p(X,Z|theta) for new weight parameters Z as provided with x and 
    hyperparameters theta (alphas & beta).

    Parameters
    ----------
    
    x : np.ndarray of float
        contains Z
    
    t : dict
        > t["energy"] target energy values = np.ndarray of shape (N,) with N atomic models
        > t["forces"] target force values = list of N np.ndarrays of shape (Na,3) with Na the
        number of atoms
    
    alphas : np.ndarray of float of shape (M,)
        M is the total number of parameters
    
    betas : np.ndarray of float of shape (2,)
        the first is the beta for the energy values and the second is the beta for the force values
    
    negative : bool
        if True (default) this function returns the negative log joint, useful for minimization
    
    force_analytic : bool
        True by default. If False the forces will be calculated using scipy's InterpolatedUnivariateSpline 
        to obtain the forces.

    Returns
    -------
    
    logp : float
        log p(X,Z|theta)
    """
    tmp_x = np.array(x,dtype=float)
        
    W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x) #fe.nonlin.setup_weights(tmp_x,params,mapper)
    W_pair_neigh = W_pair
    
    energies = calculate_all_energies(W_emb,W_pair)
    
    if force_analytic:
        forces = calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = calculate_all_forces_from_splines(tmp_x)
    if t is None:
        t = var("data",key="t")
    N = len(t["forces"])
    M = len(x)
    
    logp_reg = -.5 * np.dot(alphas,tmp_x**2)
    logp_evi_energy = -.5 * betas[0] * np.sum((energies-t["energy"])**2)
    logp_evi_force = -.5 * betas[1] * np.sum([np.sum((np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))**2)/float(len(forces[v_s])) for v_s in range(N)])
    logp_hyp = .5 * (N+M) * np.log(2*np.pi) + .5 * np.sum(np.log(alphas)) + .5 * len(betas) * N * np.sum(np.log(betas))
    if negative:
        return - (logp_reg + logp_evi_energy + logp_evi_force + logp_hyp)
    else:
        return logp_reg + logp_evi_energy + logp_evi_force + logp_hyp
        
def log_joint_pair(x,alphas,betas,negative=True,force_analytic=True,t=None,embmod=False):
    """Calculates the log joint p(X,Z|theta) for new weight parameters Z as provided with x and 
    hyperparameters theta (alphas & beta).

    The difference between this function and log_joint is that this function is specialised for
    the regression of pair energy terms for unlike elements.

    Parameters
    ----------
    
    x : np.ndarray of float
        contains Z
    
    t : dict
        > t["energy"] target energy values = np.ndarray of shape (N,) with N atomic models
        > t["forces"] target force values = list of N np.ndarrays of shape (Na,3) with Na the
        number of atoms
    
    alphas : np.ndarray of float of shape (M,)
        M is the total number of parameters
    
    betas : np.ndarray of float of shape (2,)
        the first is the beta for the energy values and the second is the beta for the force values
    
    negative : bool
        if True (default) this function returns the negative log joint, useful for minimization
    
    force_analytic : bool
        True by default. If False the forces will be calculated using scipy's InterpolatedUnivariateSpline 
        to obtain the forces.

    Returns
    -------
    
    logp : float
        log p(X,Z|theta)
    """
    tmp_x = np.array(x,dtype=float)
        
    if embmod:
    	W_emb,W_pair,W_emb_neigh = setup_weights_pair_embmod(tmp_x)
    	
    else:
    	W_pair = setup_weights_pair(tmp_x)
    	W_emb, W_emb_neigh = var("W_emb_c"), var("W_emb_neigh_c")
    
    W_pair_neigh = W_pair
    
    energies = calculate_all_energies(W_emb,W_pair)
    
    if force_analytic:
        forces = calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = calculate_all_forces_from_splines(tmp_x)
    if t is None:
        t = var("data",key="t")
    N = len(t["forces"])
    M = len(x)
    
    logp_reg = -.5 * np.dot(alphas,tmp_x**2)
    logp_evi_energy = -.5 * betas[0] * np.sum((energies-t["energy"])**2)
    logp_evi_force = -.5 * betas[1] * np.sum([np.sum((np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))**2)/float(len(forces[v_s])) for v_s in range(N)])
    logp_hyp = .5 * (N+M) * np.log(2*np.pi) + .5 * np.sum(np.log(alphas)) + .5 * len(betas) * N * np.sum(np.log(betas))
    if negative:
        return - (logp_reg + logp_evi_energy + logp_evi_force + logp_hyp)
    else:
        return logp_reg + logp_evi_energy + logp_evi_force + logp_hyp

def least_square_measure(x,t=None):
    """
    Least square measure provided the references, the functions and the weights.
    Energies and forces are being calculated.
    """
    tmp_x = np.array(x,dtype=float)
    W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x)#fe.nonlin.setup_weights(tmp_x,params,mapper)
    W_pair_neigh = W_pair
    energies = calculate_all_energies(W_emb,W_pair)
    if force_analytic:
        forces = calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = calculate_all_forces_from_splines(tmp_x)
    N = len(t["forces"])
    M = len(x)
    Natoms = np.array([len(v) for v in t["forces"]])
    e_error = np.linalg.norm( (energies - t["energy"]) )
    f_error = sum([np.sum(np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))/Natoms[v_s] for v_s in range(N)]) / float(3)
    
    return e_error + f_error

def least_square_measure_energy(x,t=None,e_fun=None):
    energies = e_fun(x)    
    N = len(t["energy"])
    M = len(x)
    
    e_error = np.linalg.norm( (energies - t["energy"]) )
    
    return e_error

class calculator:
    """Class to do regressions with.
    
    The idea is that this class is handed the necessary information required for regression
    as well as the regression function (reg_fun) and, if necessary, an evaluation funciton (evl_fun).
    Two use cases are envisioned:
    
    1) running a regression such as with RVM
    calc = calculator(params,targets,reg_fun=reg_fun,evl_fun=None)
    reg_results = calc.run()
    
    2) using a generic optimizer
    calc = calculator(params,targets,reg_fun=None,evl_fun=evl_fun)
    reg_results = optimizer(calc,x0,...)
    
    The cruical difference is the specification in reg_fun and evl_fun. For generic optimizers
    an evl_fun is a necessary condition and can be anything, such as a Bayesian measure.
    
    Parameters
    ----------
    
    reg_fun : function instance or string
        regression function
    
    evl_fun : function instance or string
        evaluation function
    
    params : dict
        parameters required for regression / evaluation, i.e. params = {"Phi": np.ndarray}
    
    targets : dict
        target values for the optimization targets = {"density": np.ndarray}
    
    opts : dict
        optional parameters for reg_fun or evl_fun
    
    Notes
    -----
    
    The keys in params and targets should be the names of the parameters the respective
    functions expect.
    """
    def __init__(self,params,targets=None,opts=dict(),reg_fun=None,evl_fun=None,fitness_tuple=False,embmod=False):
        
        # initializing
        self.reg_fun = reg_fun
        self.evl_fun = evl_fun
        self.t = targets
        self.params = params
        self.opts = opts
        self.fitness_tuple = fitness_tuple
        self.embmod = embmod

        if reg_fun is not None: print("Regressing with {}".format(reg_fun.__name__))
        if evl_fun is not None: print("Measuring fit quality with {}".format(evl_fun.__name__)) 
    
    def __getstate__(self):
        return self.reg_fun, self.evl_fun, self.t, self.params, self.opts, self.fitness_tuple

    def __setstate__(self,state):
        self.reg_fun, self.evl_fun, self.t, self.params, self.opts, self.fitness_tuple = state

    def __call__(self,x):
        """Used to return a fitness value given model parameters and previously calculated
        and stored target and other processing values.
        
        Parameters
        ----------
        
        x : np.ndarray of float
            array of model parameters
        """
        assert self.evl_fun is not None, "Assertion failed - evl_fun is not yet set!"
        
        if self.evl_fun.__name__ == "log_joint":
            alphas = self.params["alphas"]
            betas = self.params["betas"]
            val = self.evl_fun(x,alphas,betas,t=self.t)
        
        elif self.evl_fun.__name__ == "log_joint_pair":
            alphas = self.params["alphas"]
            betas = self.params["betas"]
            embmod = True
            warnings.warn("'embmod' is hard coded to '{}' since the evaluation with deap with toolbox.map(toolbox.evaluate,population) seemed to loose 'self.embmod' of this class!".format(embmod))
            val = self.evl_fun(x,alphas,betas,t=self.t,embmod=embmod) # this will need to be fixed, if this works ADD WARNING FOR EMBMOD!

        elif self.evl_fun.__name__ == "least_square_measure_energy":
            e_fun = self.params["e_fun"]
            val = self.evl_fun(x,self.t,e_fun)

        elif self.evl_fun.__name__ == "least_square_measure":
            val = self.evl_fun(x,self.t)
            
        else:
            raise NotImplementedError
        
        if self.fitness_tuple:
            return (val,)
        else:
            return val
            
    def run(self): # Currently not maintained!
        """Calls the given reg_fun.
        
        """
        assert self.reg_fun is not None, "Assertion failed - reg_fun is not yet set!"
        
        if self.reg_fun.__module__ == "fitelectrondensity.rvm":
            Phi = self.params["Phi"]
            t = self.t["density"]
            logbook = self.reg_fun(Phi,t,**self.opts)
            return logbook
        else:
            raise NotImplementedError

def eam_cma(fun,N,MAXITER=100,verbose=True,init_lb=-4,init_ub=4,sigma=10.,lambda_=None,weights=None,std_lb=1e-4,Nhof=1):
    """
    Parameters
    ----------
    
    fun : function or class instance 
        returns the fitness when provided with a parameter array  
    
    N : int
        number of potential parameters
    
    MAXITER : int
        maximum number of iterations
    
    verbose : bool
        verbosity setting
    
    init_lb : float
        initial lower bound for centroid generation
    
    init_ub : float
        initial upper bound for centroid generation
    
    sigma : float
        1/5th of the domain
    
    lambda_ : int
    """

    toolbox = base.Toolbox()
    toolbox.register("map",futures.map)
    print("Calc ",id(fun)," embmod ",fun.embmod)
    toolbox.register("evaluate", fun)
    print("toolbox ",toolbox.evaluate.embmod,id(toolbox.evaluate))

    halloffame = tools.HallOfFame(Nhof)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    t = 0
    #global lambda_
    if lambda_ is None:
        #lambda0 = 4 + int(3 * np.log(N))
        #lambda_ = int(lambda0 * (0.5**(np.random.rand()**2)))
        lambda_ = 20*int(N)
    else:
        lambda_ = int(lambda_)
    print("lambda_ {}".format(lambda_))
    #sigma = 2 * 10**(-2 * np.random.rand())
    
    if weights is not None:
        strategy = cma.Strategy(centroid=weights, sigma=sigma, lambda_=lambda_)
    else:
        strategy = cma.Strategy(centroid=np.random.uniform(init_lb, init_ub, N), sigma=sigma, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    print("\nDoing CMA...")       
    
    TOLHISTFUN = 10**-12
    TOLHISTFUN_ITER = 10 + int(np.ceil(30. * N / lambda_))
    EQUALFUNVALS = 1. / 3.
    EQUALFUNVALS_K = int(np.ceil(0.1 + lambda_ / 4.))
    TOLX = 10**-12
    TOLUPSIGMA = 10**20
    CONDITIONCOV = 10**14
    STAGNATION_ITER = int(np.ceil(0.2 * t + 120 + 30. * N / lambda_))
    NOEFFECTAXIS_INDEX = t % N

    sstring = "\nN = {}:\nlambda_ {}\nsigma {}\nMAXITER {}\nTOLHISTFUN {}\n TOLHISTFUN_ITER {}\nSTAGNATION_ITER {}".format(N,lambda_,sigma,MAXITER,TOLHISTFUN,TOLHISTFUN_ITER,STAGNATION_ITER)
    print(sstring)
    
    equalfunvalues = list()
    bestvalues = list()
    medianvalues = list()
    mins = deque(maxlen=TOLHISTFUN_ITER)

    # We start with a centroid in [-lb, ub]**D

    conditions = {"MaxIter" : False, "TolHistFun" : False, "EqualFunVals" : False,
                "TolX" : False, "TolUpSigma" : False, "Stagnation" : False,
                "ConditionCov" : False, "NoEffectAxis" : False, "NoEffectCoor" : False,
                "small_std":False}
                
    # Run the current regime until one of the following is true:
    while not any(conditions.values()):
        # Generate a new population
        population = toolbox.generate()

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
                
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=t, evals=lambda_, **record)
        if verbose:
            print(logbook.stream)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        # Count the number of times the k'th best solution is equal to the best solution
        # At this point the population is sorted (method update)
        if population[-1].fitness == population[-EQUALFUNVALS_K].fitness:
            equalfunvalues.append(1)

        # Log the best and median value of this population
        bestvalues.append(population[-1].fitness.values)
        medianvalues.append(population[int(round(len(population)/2.))].fitness.values)

        t += 1
        STAGNATION_ITER = int(np.ceil(0.2 * t + 120 + 30. * N / lambda_))
        NOEFFECTAXIS_INDEX = t % N

        if t >= MAXITER:
            # The maximum number of iteration per CMA-ES ran
            conditions["MaxIter"] = True

        mins.append(record["min"])
        if (len(mins) == mins.maxlen) and max(mins) - min(mins) < TOLHISTFUN:
            # The range of the best values is smaller than the threshold
            conditions["TolHistFun"] = True

        if t > N and sum(equalfunvalues[-N:]) / float(N) > EQUALFUNVALS:
            # In 1/3rd of the last N iterations the best and k'th best solutions are equal
            conditions["EqualFunVals"] = True

        if all(strategy.pc < TOLX) and all(np.sqrt(np.diag(strategy.C)) < TOLX):
            # All components of pc and sqrt(diag(C)) are smaller than the threshold
            conditions["TolX"] = True

        if strategy.sigma / sigma > strategy.diagD[-1]**2 * TOLUPSIGMA:
            # The sigma ratio is bigger than a threshold
            conditions["TolUpSigma"] = True

        if len(bestvalues) > STAGNATION_ITER and len(medianvalues) > STAGNATION_ITER and \
        np.median(bestvalues[-20:]) >= np.median(bestvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]) and \
            np.median(medianvalues[-20:]) >= np.median(medianvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]):
            # Stagnation occured
            conditions["Stagnation"] = True

        if strategy.cond > 1e15:
            # The condition number is bigger than a threshold
            conditions["ConditionCov"] = True

        if all(strategy.centroid == strategy.centroid + 0.1 * strategy.sigma * strategy.diagD[-NOEFFECTAXIS_INDEX] * strategy.B[-NOEFFECTAXIS_INDEX]):
            # The coordinate axis std is too low
            conditions["NoEffectAxis"] = True

        if any(strategy.centroid == strategy.centroid + 0.2 * strategy.sigma * np.diag(strategy.C)):
            # The main axis std has no effect
            conditions["NoEffectCoor"] = True
            
        if logbook[-1]["std"] < std_lb:
            conditions["small_std"] = True

    stop_causes = [k for k, v in conditions.items() if v]
    print("Stopped because of condition%s %s" % ((":" if len(stop_causes) == 1 else "s:"), ",".join(stop_causes)))
    
    return halloffame, logbook

def wrapper_bayes_noise(x,alphas,force_analytic=True,pair_only=False,embmod=False):
    t = var("data",key="t")
    alphas = np.absolute(alphas)
    tmp_x = np.array(x,dtype=float)

    if pair_only:
        W_pair = setup_weights_pair(tmp_x)
        W_emb, W_emb_neigh = var("W_emb_c"), var("W_emb_neigh_c")
    elif embmod:
    	W_emb,W_pair,W_emb_neigh = setup_weights_pair_embmod(tmp_x)
    else:
        W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x)
    W_pair_neigh = W_pair
    energies = calculate_all_energies(W_emb,W_pair)
    
    if force_analytic:
        forces = calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = get_forces_from_splines(tmp_x)
    
    N = len(t["forces"])
    M = len(x)
    
    logp_reg = -.5 * np.dot(alphas,tmp_x**2)
    
    de = np.sum((energies-t["energy"])**2)
    df = np.sum([np.sum((np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))**2)/float(len(forces[v_s])) for v_s in range(N)])
    
    def bayes_noise(betas):
        """Calculates the log likelihood for new weight parameters as provided with x.
        """
        betas = np.absolute(betas)
        logp_evi_energy = -.5 * betas[0] * de
        logp_evi_force = -.5 * betas[1] * df
        logp_hyp = .5 * (len(betas)*N+M) * np.log(2*np.pi) + .5 * np.sum(np.log(alphas)) + .5 * len(betas) * N * np.sum(np.log(betas))
        return - (logp_reg + logp_evi_energy + logp_evi_force + logp_hyp)
    return bayes_noise
        
def wrapper_bayes_regularizer(x,betas):
    t = var("data",key="t")
    betas = np.absolute(betas)
    tmp_x = np.array(x,dtype=float)
    
    N = len(t["forces"])
    M = len(x)
        
    def bayes_regularizer(alphas):
        """Calculates the log likelihood for new weight parameters as provided with x.
        """
        alphas = np.absolute(alphas)
        logp_reg = -.5 * np.dot(alphas,tmp_x**2)
        logp_hyp = .5 * (len(betas)*N+M) * np.log(2*np.pi) + .5 * np.sum(np.log(alphas)) + .5 * len(betas) * N * np.sum(np.log(betas))
        return - (logp_reg + logp_hyp)
    return bayes_regularizer
