################ DEPRECATE???? ##############


def evo(eval_fun,N=100,MAXITER=100,verbose=True,sigma=10,lambda_=None,init_lb=0,init_ub=100):
    """
    Parameters
    ----------
    N : int
        number of potential parameters
    init_lb : float
        initial lower bound for centroid generation
    init_ub : float
        initial upper bound for centroid generation
    sigma : float
        1/5th of the domain
    """
    
    print("\nDoing CMA...")  
    creator.create("FitnessMin", base.Fitness, weights=(-1.,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    #toolbox.register("map",futures.map)
    toolbox.register("evaluate", eval_fun)

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #lambda_=None,sigma=None,init_lb=-4,init_ub=4
    centroid = np.random.uniform(init_lb, init_ub, N)
    print("centroid {} N {}".format(len(centroid),N))
    strategy = cma.Strategy(centroid=centroid, sigma=sigma, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    #logbooks.append(tools.Logbook())
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    #toolbox = base.Toolbox()
    
    t = 0
    #global lambda_
    if lambda_ is None:
        lambda0 = 4 + int(3 * np.log(N))
        lambda_ = int(lambda0 * (0.5**(np.random.rand()**2)))
    print("lambda_ {}".format(lambda_))
    #sigma = 2 * 10**(-2 * np.random.rand())
    
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
                "ConditionCov" : False, "NoEffectAxis" : False, "NoEffectCoor" : False}
                
                
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

    stop_causes = [k for k, v in conditions.items() if v]
    print("Stopped because of condition%s %s" % ((":" if len(stop_causes) == 1 else "s:"), ",".join(stop_causes)))
    
    return halloffame, logbook

def wrapper_neg_log_likelihood_nonlinear(X,t,fun,alpha,beta,basis_type,Phi=None):
    # for normal parameter (weight) optimization
    N = len(t)
    def neg_log_likelihood(weights):
        weights = np.array(weights)
        t_pred = fun(X,weights,basis_type=basis_type,Phi=Phi)
        regularizer = N * np.dot(alpha,weights**2)
        main_pt = beta * np.sum( (t-t_pred)**2)
        norm_pt = N * ( np.sum(np.log(1./alpha)) + np.sum(np.log(1./beta)) )
        return (main_pt + regularizer + norm_pt,)
    return neg_log_likelihood

def wrapper_log_likelihood_nonlinear(fun,alpha,beta,basis_type,weights,Phi=None,verbose=False):
    # for log likelihood calculation
    norm_pt =  np.sum(np.log(1./alpha)) + np.log(1./beta)
    weights = np.reshape(weights,(-1,))
    regularizer = np.dot(alpha,weights**2)
    def log_likelihood(X,t):
        try:
            N = len(t)
        except:
            N = 1
        
        t_pred = fun(X,weights,basis_type=basis_type,Phi=Phi)
        tmp_reg = N * regularizer
        evidence = beta * np.sum((t-t_pred)**2)
        
        if verbose: 
            print("\nalpha {} beta {}".format(alpha,beta))
            print("regularizer {} main {} norm {}".format(regularizer,evidence,norm_pt))
        return -.5 * (evidence + tmp_reg + N * norm_pt)
    return log_likelihood

def wrapper_neg_log_likelihood_nonlinear_hyper_alpha(X,t,fun,basis_type,weights,beta,Phi=None):
    # for hyper parameter optimization (alpha)
    N = len(t)
    beta = abs(beta)
    def log_likelihood(alpha):
        alpha = np.absolute(alpha)
        alpha[np.where(alpha<1e-6)] = 1e-6
        t_pred = fun(X,weights,basis_type=basis_type,Phi=Phi)
        regularizer = N * np.dot(alpha,weights**2)
        main_pt = beta * np.sum( (t-t_pred)**2)
        norm_pt = N * ( np.sum(np.log(1./alpha)) + np.sum(np.log(1./beta)) )
        #print("\nalpha {}\nbeta {}".format(alpha,beta))
        #print("main {} regularize {} norm {} alpha {}".format(main_pt,regularizer,norm_pt,alpha))
        return main_pt + regularizer + norm_pt
    return log_likelihood

def get_updated_hyperparameters_alpha_nonlinear(X,fun,alpha0=None,method="Nelder-Mead"):
    M = len(basis)
    
    if alpha0 is not None: 
        assert len(alpha0)==len(basis), "Assertion failed - length of basis and alpha0 of unequal length..."
    else:
        alpha0 = np.ones(M)
    
    #print("alpha0 {}".format(alpha0))
    
    res = optimize.minimize(fun,alpha0,method=method)
    #print("res {}".format(res))
    res["x"] = np.absolute(res["x"])
    res["x"][np.where(res["x"]<1e-6)] = 1e-6
    alpha = res["x"]
    return alpha
    
def save_data_for_fit(dump_path,data):
    suffix = "pckl"
    tmp_path = dump_path.split('.')
    if tmp_path[-1] != suffix:
        tmp_path.append(suffix)
        path = '.'.join(tmp_path)
    else:
        path = dump_path
    print("Dumping data for the fitting at {}...".format(path))
    with open(path,"wb") as f:
        pickle.dump(data,f)

def get_mappers(params,mapper):
    pair_map = [[[] for n in range(params["N_atoms"][s])] for s in range(params["N_bonds"])]
    emb_neigh_map = [[[] for n in range(params["N_atoms"][s])] for s in range(params["N_bonds"])]
    emb_map = [[] for s in range(params["N_bonds"])]
    for s in range(params["N_bonds"]):
        emb_map[s] = np.array([mapper["emb"][params["species"][s][v_n]] for v_n in range(params["N_atoms"][s])])
        for n in range(params["N_atoms"][s]):
            pair_map[s][n] = np.array([mapper["pair"][params["pair_species"][s][n][v_i]] \
                            for v_i in range(params["coskr"][s][n].shape[1])])
            emb_neigh_map[s][n] = np.array([mapper["emb"][params["emb_species"][s][n][i]] \
                                for i in  range(params["fprho_i"][s][n].shape[0])])
    return pair_map, emb_neigh_map, emb_map

def setup_weights(tmp_x,params,mapper,emb_map,pair_map,emb_neigh_map):
    _W_emb = [[] for s in range(params["N_bonds"])]
    _W_pair = [[[]
             for n in range(params["N_atoms"][s])] \
             for s in range(params["N_bonds"])]
    _W_emb_neigh = [[[]
             for n in range(params["N_atoms"][s])] \
             for s in range(params["N_bonds"])]
    for s in range(params["N_bonds"]):
        _W_emb[s][:] = tmp_x[emb_map[s]]
        for n in range(params["N_atoms"][s]):
            _W_pair[s][n][:] = tmp_x[pair_map[s][n]]
            _W_emb_neigh[s][n][:] = tmp_x[emb_neigh_map[s][n]]
    
    return _W_emb,_W_pair,_W_emb_neigh,_W_pair


#@profile
def setup_weights_old(tmp_x,params,mapper):
    W_emb = [np.array([tmp_x[mapper["emb"][params["species"][s][v_n]]] for v_n in range(params["N_atoms"][s])]) \
            for s in range(params["N_bonds"])]
    W_pair = [[np.array([tmp_x[mapper["pair"][params["pair_species"][s][v_n][v_i]]] \
             for v_i in range(params["coskr"][s][v_n].shape[1])]) \
             for v_n in range(params["N_atoms"][s])] \
             for s in range(params["N_bonds"])]
    W_emb_neigh = [[np.array([tmp_x[mapper["emb"][params["emb_species"][s][n][i]]] \
                    for i in  range(params["fprho_i"][s][n].shape[0])]) \
                    for n in range(params["N_atoms"][s])] \
                    for s in range(params["N_bonds"])]
    """
    W_pair_neigh = [[np.array([tmp_x[mapper["pair"][params["pair_species"][s][n][i]]] \
                    for i in  range(params["r_vec"][s][n].shape[0])]) \
                    for n in range(params["N_atoms"][s])] \
                    for s in range(params["N_bonds"])]
    """
    return W_emb,W_pair,W_emb_neigh,W_pair

def calculate_all_energies_fast_new(x,W_emb,W_pair,params):
    # also requires params and mapper, but assumes thay are global
    tmp_x = np.array(x,dtype=float)
        
    all_energies = np.zeros(params["N_bonds"])
    all_pair_energies = np.zeros(params["N_bonds"])
    
    # embedding
    tmp_emb_energy = np.zeros(params["N_bonds"])
    tmp_pair_energy = np.zeros(params["N_bonds"])

    tmp_emb_energy += np.array([np.dot(params["psirho"][s],np.einsum("ij,ji->i",W_emb[s],params["coskrho"][s]))/float(params["N_atoms"][s]) \
                        for s in range(params["N_bonds"])])
    
    # pair
    tmp_pair_energy += np.array([np.sum([np.dot(params["psir"][s][v_n],np.einsum("ij,ji->i",W_pair[s][v_n],params["coskr"][s][v_n])) \
                         for v_n in range(params["N_atoms"][s])])/float(params["N_atoms"][s]) \
                         for s in range(params["N_bonds"])])
    
    #return tmp_emb_energy, tmp_pair_energy
    return tmp_emb_energy + .5* tmp_pair_energy

def calculate_all_forces_fast_new(x,W_emb,W_pair_neigh,W_emb_neigh,params,emb_n,emb_i,pair_n):
    """
    params = {"coskrho":coskrho,"psirho":psirho,"coskr":coskr,"psir":psir, # energy related
              "ksinkr":ksinkr,"psipr":psipr,"ksinkrho":ksinkrho,"psiprho":psiprho, # force related
              "N_bonds":N_super,"N_atoms":N_atoms,
              "emb_species":X["emb_species"],"pair_species":X["pair_species"]}
    """
    tmp_x = np.array(x,dtype=float)
    #print("tmp_x {}".format(tmp_x))
    all_forces = [np.zeros((params["N_atoms"][v_s],3)) for v_s in range(params["N_bonds"])]
    
    f_emb_n = np.array([np.einsum("ij,ikj->ik",W_emb[s],emb_n[s]) for s in range(params["N_bonds"])])    
    f_emb_i = [np.array([np.sum(np.einsum("ij,ikj->ik",W_emb_neigh[s][n],emb_i[s][n]),axis=0) \
              for n in range(params["N_atoms"][s])]) \
              for s in range(params["N_bonds"])] 
    
    f_emb = [f_emb_n[s]+f_emb_i[s] for s in range(params["N_bonds"])]
    f_pair = [np.array([np.sum(np.einsum("ij,ikj->ik",W_pair_neigh[s][n],pair_n[s][n]),axis=0) \
              for n in range(params["N_atoms"][s])]) \
              for s in range(params["N_bonds"])]
    for s in range(params["N_bonds"]):
        all_forces[s] = f_pair[s] + f_emb[s]
    
    return all_forces

def get_splined_energy_functions(weights,smooth_emb,smooth_pair,f_smooth_emb,f_smooth,r_smooth,\
                                 rho_lb,rho_ub,N_steps,rho_scaling,mapper,rho_dict,r_lb,r_ub):
    """Generates splined energy functions for a 1d weights array.
    
    This function can be used in conjunction with write_EAM_setfl_file to generate
    setfl format files for an EAM potential to be read by LAMMPS for example.
    
    """
    
    def _simple(x):
        if isinstance(x,np.ndarray):
            return np.ones(x.shape)
        else:
            return 1.

    if smooth_emb:
        smooth_fun_rho = wrapped_smooth(f_smooth_emb,0,kind="<") # the contribution of the embedding energy is zero if the background density is zero
    else:
        smooth_fun_rho = _simple
        
    if smooth_pair:
        smooth_fun_r = wrapped_smooth(f_smooth,r_smooth,kind=">")
    else:
        smooth_fun_r = _simple
        
    Es = {"rho":None,
          "r":None,
          "emb":dict(),
          "pair":dict(),
          "rhos":dict()}

    tmp_rho_lb, tmp_rho_ub = min(rho_lb.values()), max(rho_ub.values())
    drho = np.round((tmp_rho_ub - tmp_rho_lb)/float(N_steps),decimals=6)
    dr = np.round((r_ub - r_lb)/float(N_steps),decimals=6)

    # rho and embedding energy
    rho = np.array([tmp_rho_lb+v*drho for v in range(N_steps)])
    Es["rho"] = rho
    f_rho = np.pi/float(rho_scaling)
    E_emb = lambda w,x: smooth_fun_rho(x) * np.dot(w,np.array([np.cos(f_rho*v*x) for v in range(len(w))]))
    
    Es["emb"] = {species: E_emb(weights[mapper["emb"][species]],rho) for species in mapper["emb"]}
    
    # r and pair energy
    r = np.array([r_lb+v*dr for v in range(N_steps)])
    Es["r"] = r
    f_r = np.pi/float(r_smooth)
    E_pair = lambda w,x: smooth_fun_r(x) * np.dot(w,np.array([np.cos(f_r*v*x) for v in range(len(w))]))
    
    Es["pair"] = {pair: E_pair(weights[mapper["pair"][pair]],r) for pair in mapper["pair"]}

    # density functions
    Es["rhos"] = {species: spline(rho_dict["r"],rho_dict["rhos"][species],ext=0)(r) for species in mapper["emb"]}

    return Es

def get_forces_from_splines_fast(x,X,): # NEEDS TO BE CHECKED!
    # assumes that the following are global: params,mapper,X,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,
    #                     r_lb=0,r_ub=5.,rho_lb=-5.,rho_ub=42.,N_steps=100,rho_scaling=1.
    
    N_super = X["N_bonds"]
    N_atoms = X["N_atoms"]
    #t0 = time.time()
    Es = get_splined_functions_fast(x)
    #print("splining energies {}s...".format(time.time()-t0))

    #t0 = time.time()
    funs = {"pair":{p:spline(Es["r"],v) for p,v in Es["pair"].items()},
                   "emb":{s:spline(Es["rho"],v) for s,v in Es["emb"].items()},
                   "rhos":{r:spline(Es["r"],v) for r,v in Es["rhos"].items()},}
    derivatives = {"pair":{p:v.derivative(n=1) for p,v in funs["pair"].items()},
                   "emb":{s:v.derivative(n=1) for s,v in funs["emb"].items()},
                   "rhos":{r:v.derivative(n=1) for r,v in funs["rhos"].items()},}
    #print("generating splines for forces {}s...".format(time.time()-t0))
    
    #all_forces = [np.zeros((N_atoms[s],3),dtype=float) for s in range(N_super)]
    #t0 = time.time()
    N_neighs = [[len(X["r"][s][n]) for n in range(N_atoms[s])] for s in range(N_super)]
    fun_fprho = lambda s,n: np.sum([X["r_vec"][s][n][i] * derivatives["rhos"][X["emb_species"][s][n][i]](X["r"][s][n][i]) for i in range(N_neighs[s][n])],axis=0)
    fun_emb_neigh = lambda s,n: np.sum([X["r_vec"][s][n][i] * derivatives["rhos"][X["species"][s][n]](X["r"][s][n][i]) * derivatives["emb"][X["emb_species"][s][n][i]](X["density"][s][X["neigh_idx_super"][s][n][i]]) for i in range(N_neighs[s][n])],axis=0)
    fun_forces_pair = lambda s,n: np.sum([X["r_vec"][s][n][i] * derivatives["pair"][tuple(sorted([X["species"][s][n],X["emb_species"][s][n][i]]))](X["r"][s][n][i]) for i in range(N_neighs[s][n])],axis=0)
    
    fast_all_forces = [np.array([fun_forces_pair(s,n) + fun_fprho(s,n) * derivatives["emb"][X["species"][s][n]](X["density"][s][n]) + fun_emb_neigh(s,n) for n in range(N_atoms[s])],dtype=float) for s in range(N_super)]
    
    return fast_all_forces

def load_data_for_fit(load_path):
    print("Loading data for the EAM fitting from {}...".format(load_path))
    with open(load_path,"rb") as f:
        data = pickle.load(f)
    return data
    
def generate_EAM_calculator(Es,N_steps,r_cut,show=False,
                        atom_info={'Ni':{"mass":58.6934,"number":28,"lattice":"fcc","a0":3.52},
                                   'Al':{"mass":26.9815,"number":13,"lattice":"fcc","a0":4.02},
                                   'Ti':{"mass":47.88,"number":22,"lattice":"fcc","a0":2.95},
                                   'Nb':{"mass":92.90637,"number":41,"lattice":"bcc","a0":3.3}}):
    """
    This function takes the regressed empirical potential in 'Es' and creates an ase EAM calculator object.
    """
    
    elements = list(sorted(Es["emb"].keys()))
    N_ele = len(elements)
    pair_map = {tuple([el1,el2]):tuple(sorted([el1,el2])) for el1,el2 in itertools.product(elements,elements)}
    
    dr = Es["r"][1] - Es["r"][0]
    drho = Es["rho"][1] - Es["rho"][0]
    
    Nrho = N_steps
    Nr = N_steps
    
    embedded_energy = np.array([spline(Es["rho"],Es["emb"][el]) for el in elements])
    electron_density = np.array([spline(Es["r"],Es["rhos"][el]) for el in elements])
    phi = np.array([[spline(Es["r"],Es["pair"][(pair_map[(el2,el1)])]) for el2 in elements] for el1 in elements])

    EAM_obj = EAM(elements=elements, 
                  embedded_energy=embedded_energy,
                  d_embedded_energy=np.array([v.derivative() for v in embedded_energy]),
                  electron_density=electron_density,
                  d_electron_density=np.array([v.derivative() for v in electron_density]),
                  phi=phi,
                  d_phi=np.array([[phi[v,v2].derivative() for v2 in range(N_ele)] for v in range(N_ele)]),  
                  cutoff=r_cut, form='alloy',
                  Z=[atom_info[el]["number"] for el in elements], nr=Nr, nrho=Nrho, dr=dr, drho=drho,
                  lattice=[atom_info[el]["lattice"] for el in elements], 
                  mass=[atom_info[el]["mass"] for el in elements], 
                  a=[atom_info[el]["a0"] for el in elements])
    setattr(EAM_obj,"Nelements",N_ele)
    return EAM_obj
