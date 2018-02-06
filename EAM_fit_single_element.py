"""
Script for the non-linear regression of EAM potentials to DFT structures 
for single element systems.

Example:
>>> python -m scoop EAM_fit_single_element.py

Note:
This script requires some preprocessing of the DFT structures stored as some 
*.pckl file. An example of this kind of preprocessing can be found in the 
'EAM regression tutorial.ipynb' notebook under '2. Preparing an EAM regression'.

Eric Schmidt
2017-11-08
"""

from __future__ import print_function
import numpy as np
from scipy import stats, optimize

import fitenergy as fe
from fitenergy.misc import setup_emb_and_pair_contribution, get_mappers,\
    setup_weights
from fitenergy.potential import calculate_all_energies, calculate_all_forces,\
    get_splined_functions
from fitenergy.regression import log_joint, calculator, eam_cma,\
    wrapper_bayes_noise, wrapper_bayes_regularizer

from scoop import shared
from deap import base, creator

creator.create("FitnessMin", base.Fitness, weights=(-1.,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def fit_single_element(fit_data_load_path, template_save_Es_path, maxiter=1000,\
                       lambda_=None, maxiter_all=7, maxiter_hyper=2000, seed=None,\
                       NRPT=list(range(7)), fix_beta=False,\
                       minimize_method="deap-cma", minimize_method_noise="BFGS",\
                       minimize_method_regularizer="BFGS", alphas_init=None, betas_init=None):
    """Fits pre-processed  DFT energy and force data for a single element.

    Parameters
    ----------

    fit_data_load_path : str
        Path to pre-processed data.

    template_save_Es_path : callable
        Produces a string to use as path to write *.Es files to for each job.

    maxiter : int, optional, default 1000
        Number of iterations for each hyperparameter setting.

    lambda_ : int, optional, default None
        CMA parameter to control the number of individuals.

    maxiter_all : int, optional, default 7
        Number of rounds to optimize hyperparameters and model weights.

    maxiter_hyper : int, optional, default 2000
        Maximum number of iterations for the optimization of the hyperparameters.

    seed : int, optional, default None

    NRPT : list of ints, optional, default [0, 1, 2, 3, 4, 5, 6, 7]
        Number of RePeTitions (NRPT)/number and names of jobs for the entire optimization.
        There will be as many jobs as NRPT is long. Note that 'template_save_Es_path'
        will make use of the entries in NRPT. Thus repeating entries will lead to 
        an overwriting of files.

    fix_beta : boolean, optional, default False
        Whether to fix betas during the optimization or not.

    minimize_method : str, optional, default 'deap-cma'
        Currently only 'deap-cma' is a method implemented and tested for the weights
        optimization.

    minimize_method_noise : str, optional, default 'BFGS'
        Method to minimize the noise hyperparmaeters with.

    minimize_method_regularizer : str, optional, default 'BFGS'
        Method to minimize the weight hyperparmaeters with.

    alphas_init : None or instance of stats.rv_continuous, optional, default None
        If None is provided the alpha parameters will be initialized using a 
        uniform distribution.

    betas_init : None or instance of stats.rv_continuous, optional, default None
        If None is provided the beta parameters will be initialized using a 
        uniform distribution.

    Returns
    -------
    
    Nothing, writes results to one or multiple *.Es files to disk.
    """
    print("Loading file from ",fit_data_load_path," ...")
    data = fe.load_data(fit_data_load_path)
    M = len(data["x0"])

    for nrpt in NRPT:
        print("\n\n")
        print("###################################")
        print("############### nrpt = {} ###############".format(nrpt))
        print("###################################")
        print("\n\n")

        # output details
        save_Es_path = template_save_Es_path(nrpt)

        # hyperparameter initialization
        if not alphas_init is None and isinstance(alphas_init,stats.rv_continuous):
            alphas_init = alphas_init.rvs(size=M)
        else:
            alphas_init = np.random.uniform(0.1,1000,size=M)
        
        if not betas_init is None and isinstance(betas_init,stats.rv_continuous):
            betas_init = betas_init.rvs(size=2)
        else:
            betas_init = np.random.uniform(0.1,1000,size=2) # betas[0]: energy, betas[1]: force		

        #### regression
        all_weights = None
        all_fitnesses = []
        weights = None
        all_potty_energies = []
        all_potty_forces = []

        for i_iter in range(maxiter_all):
            print("\n ============== global iteration {}/{} ==============\n".format(i_iter+1, maxiter_all))

            if i_iter == 0:
                alphas_curr = np.array(alphas_init)
                betas_curr = np.array(betas_init)

            print("alphas {}\nbetas {}\ninitial weights {}".format(alphas_curr, betas_curr, weights))

            # calculator class
            fitness_tuple = True
            calc_params = {"alphas":alphas_curr,"betas":betas_curr}
            calc = calculator(calc_params, opts=dict(), evl_fun=log_joint,\
                    fitness_tuple=fitness_tuple)

            # running deap-cma
            hof, logbook = eam_cma(calc, M, MAXITER=maxiter, verbose=True,\
                lambda_=lambda_, weights=weights, Nhof=Nhof)

            _weights = np.array([v for v in hof],dtype=float)
            weights = _weights[0,:]
            fitnesses = [calc(v) for v in _weights]
            all_fitnesses.extend(fitnesses)
            print("weight optimization results:\n    weights: {}\n    fitness: {}".format(weights,min([v["min"] for v in logbook])))
            print("hof fitnesses {}".format(fitnesses))

            if i_iter == 0:
                all_weights = np.array(_weights)
            else:
                all_weights = np.vstack((all_weights,_weights))

            # computing energies and forces for the hof
            for _w in _weights:
                _e,_f = fe.regression.Potty_energies_and_forces(_w, force_analytic=True)
                all_potty_energies.append(_e)
                all_potty_forces.append(_f)

            if i_iter+1 == maxiter_all:
                continue

            # optimization of the noise levels
            if not fix_betas:
                noise_fun = wrapper_bayes_noise(weights, alphas_curr)
                res_noise = optimize.minimize(noise_fun, betas_curr,\
                    method=minimize_method_noise,\
                    options={"maxiter":maxiter_hyper,"disp":True})
                betas_curr = np.absolute(res_noise["x"])
                print("noise optimization results:\n    betas: {}\n    fitness: {}".format(betas_curr,res_noise["fun"]))

            else:
                betas_curr = betas_init

            # optimization of the weight priors
            regularizer_fun = wrapper_bayes_regularizer(weights, betas_curr)            
            res_regularizer = optimize.minimize(regularizer_fun, alphas_curr,\
                method=minimize_method_regularizer, \
                options={"maxiter":maxiter_hyper, "disp":True})    
            alphas_curr = np.absolute(res_regularizer["x"])
            print("weight prior optimization results:\n    alphas: {}\n    fitness: {}".format(alphas_curr,res_regularizer["fun"]))

        #calculating regressed energies and forces for later comparison
        W_emb, W_pair, W_emb_neigh = setup_weights(weights)
        W_pair_neigh = W_pair
        reg_e = calculate_all_energies(W_emb, W_pair)        
        reg_f = calculate_all_forces(W_emb, W_pair_neigh, W_emb_neigh)

        Es = get_splined_functions(weights)

        fe.save_Es(save_Es_path,Es,data["X"],data["t"],calc,regressed_e=reg_e,\
            regressed_f=reg_f,dft_path=data["dft_path"],load_path_rhos=data["load_path_rhos"],\
            rho_dict=data["rho_dict"],num_neigh=data["num_neigh"],r_cut=data["r_cut"],aniso=data["aniso"],\
            ultra_num=data["ultra_num"],selection=data["selection"],seed=seed,k_pair=data["k_pair"],
            k_emb=data["k_emb"],smooth_emb=data["smooth_emb"],smooth_pair=data["smooth_pair"],\
            type_pair=data["type_pair"],type_emb=data["type_emb"],r_smooth=data["r_smooth"],f_smooth=data["f_smooth"],\
            minimize_method=minimize_method,maxiter=maxiter,lambda_=lambda_,\
            eam_path=None,reg_time=None, res=None,logbook=logbook,\
            fitness_tuple=fitness_tuple,r_lb=data["r_lb"],r_ub=data["r_ub"],rho_lb=data["rho_lb"],rho_ub=data["rho_ub"],\
            N_steps=data["N_steps"],rho_scaling=data["rho_scaling"],weights=weights,\
            alphas=alphas_curr,betas=betas_curr,force_analytic=True,
            all_fitnesses=all_fitnesses,all_weights=all_weights,\
            fit_data_load_path=fit_data_load_path,f_smooth_emb=data["f_smooth_emb"],
            mapper_complete=data["mapper_complete"],x0_complete=data["x0_complete"],
            mapper=data["mapper"],x0=data["x0"],all_potty_energies=all_potty_energies,all_potty_forces=all_potty_forces)

if __name__ == "__main__":

    # regression settings
    minimize_method = "deap-cma" # "Nelder-Mead" and a bunch of others will lead to the usage of scopy.optimize.minimize, whereas "deap-cma" will lead to the usage of a CMA algorithm
    minimize_method_noise = "BFGS"
    minimize_method_regularizer = "BFGS"
    maxiter = 100 #1000 int, maximum number of iterations for both scipy.optimize.minimize and deap 
    lambda_ = 50 #None int/None, number of individuals per generation (lambda_ = 20*M recommended, M = number of weights). if None then lambda_ will be guessed
    maxiter_all = 2 # int, max number of hyperparameter optimization with optimized weights
    maxiter_hyper = 2000 # int, max number of iteration steps for individual hyperparameters during a loop where the weights are also updated
    seed = None
    np.random.seed(seed)

    NRPT = list(range(2)) # the number of values is how often the entire optimization of weights and hyperparameters is repeated. the values themselves are the name modifiers for the .Es file

    Nhof = 25 # number of individuals for the hall of fame to be stored for each iteration with fixed hyperparameters
    fix_betas = False

    # input details - change both files here to fit energy functions for the respective element and write results to disk
    fit_data_load_path = "./tests/unittest_files/EAM_test_Al-Ni/setup_plain_normed_Ni.pckl"
    template_save_Es_path = lambda x:"./tests/unittest_files/EAM_test_Al-Ni/Ni_plain_normed_{}.Es".format(x)

    # hyperparameter priors
    alphas_init = stats.halfnorm(scale=1.)
    betas_init = stats.halfnorm(scale=1.)

    ###########
    data = fe.load_data(fit_data_load_path)
    shared.setConst(data=data)
    emb_n, emb_i, pair_n = setup_emb_and_pair_contribution()
    shared.setConst(emb_n=emb_n, emb_i=emb_i, pair_n=pair_n)
    pair_map, emb_neigh_map, emb_map = get_mappers()
    shared.setConst(pair_map=pair_map, emb_neigh_map=emb_neigh_map, emb_map=emb_map)

    fit_single_element(fit_data_load_path, template_save_Es_path,\
        maxiter=maxiter, lambda_=lambda_, maxiter_all=maxiter_all,\
        maxiter_hyper=maxiter_hyper, seed=seed, NRPT=NRPT, fix_beta=fix_betas,\
        minimize_method=minimize_method, minimize_method_noise=minimize_method_noise,\
        minimize_method_regularizer=minimize_method_regularizer, alphas_init=alphas_init,\
        betas_init=betas_init)