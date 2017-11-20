"""
This script is intended to fit an EAM potential using data prepared using the 
EAM_setup.py script.
"""

import pickle, time, timeit, copy
from scipy import optimize
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
from scoop import futures
from collections import deque
import sys

# Potty specific
sys.path.insert(0,"../")
import fitenergy as fe

# optimization
minimize_method = "deap-cma" # "Nelder-Mead" and a bunch of others will lead to the usage of scopy.optimize.minimize, whereas "deap-cma" will lead to the usage of a CMA algorithm
minimize_method_noise = "BFGS"
minimize_method_regularizer = "BFGS"
maxiter = 10 # int, maximum number of iterations for both scipy.optimize.minimize and deap 
lambda_ = 10 #None # int/None, number of individuals per generation. if None then lambda_ will be guessed  
maxiter_all = 2 # int, max number of hyperparameter optimization with optimized weights
maxiter_hyper = 2000 # int, max number of iteration steps for individual hyperparameters during a loop where the weights are also updated
#np.random.seed(42)

# output details
fit_data_load_path = "./unittest_files/test_energyforce.pckl"
save_Es_path = "./unittest_files/test_energyforce.Es"

eam_path = None #"./unittest_files/test_Ni3Al_ideal_dis_bayes.eam.alloy"
force_analytic = True

#betas_init = np.array([326.02193541,  1563.57444054],dtype = float)
betas_init = None
fix_betas = False
alphas_init = None

data = fe.nonlin.load_data_for_fit(fit_data_load_path)
rho_scaling = data["rho_scaling"]
seed = data["seed"]
t = data["t"]
X = data["X"]
mapper = data["mapper"]
params = data["params"]
x0 = data["x0"]
smooth_emb=data["smooth_emb"]
smooth_pair=data["smooth_pair"]
f_smooth=data["f_smooth"]
f_smooth_emb = data["f_smooth_emb"]
r_smooth=data["r_smooth"]
rho_dict=data["rho_dict"]
r_lb=data["r_lb"]
r_ub=data["r_ub"]
r_cut=data["r_cut"]
num_neigh=data["num_neigh"]
rho_lb=data["rho_lb"]
rho_ub=data["rho_ub"]
N_steps=data["N_steps"]
dft_path=data["dft_path"]
load_path_rhos=data["load_path_rhos"]
aniso=data["aniso"]
ultra_num=data["ultra_num"]
selection=data["selection"]
k_pair=data["k_pair"]
k_emb=data["k_emb"]
type_pair=data["type_pair"]
type_emb=data["type_emb"]

if smooth_emb:
    emb_n = [np.array([[params["fprho_n"][s][n][j] * (params["psiprho"][s][n]*params["coskrho"][s][:,n] - params["psirho"][s][n] * params["ksinkrho"][s][:,n]) \
            for j in range(3)] for n in range(params["N_atoms"][s])])\
            for s in range(params["N_bonds"])] #(Natoms,Nk,dim)
    emb_i = [[np.array([[params["fprho_i"][s][n][i][j] * (params["psiprho"][s][params["neigh_idx_super"][s][n][i]]*params["coskrho"][s][:,params["neigh_idx_super"][s][n][i]] - params["psirho"][s][params["neigh_idx_super"][s][n][i]]*params["ksinkrho"][s][:,params["neigh_idx_super"][s][n][i]])\
            for j in range(3)] for i in range(params["fprho_i"][s][n].shape[0])]) \
            for n in range(params["N_atoms"][s])] \
            for s in range(params["N_bonds"])]
else:
    emb_n = [np.array([[params["fprho_n"][s][n][j] * ( - params["ksinkrho"][s][:,n]) \
                for j in range(3)] for n in range(params["N_atoms"][s])])\
                for s in range(params["N_bonds"])] #(Natoms,Nk,dim)
    emb_i = [[np.array([[params["fprho_i"][s][n][i][j] * (- params["ksinkrho"][s][:,params["neigh_idx_super"][s][n][i]])\
                for j in range(3)] for i in range(params["fprho_i"][s][n].shape[0])]) \
                for n in range(params["N_atoms"][s])] \
                for s in range(params["N_bonds"])]
pair_n = [[np.array([[params["r_vec"][s][n][i][j] * (params["psipr"][s][n][i]*params["coskr"][s][n][:,i] - params["psir"][s][n][i]*params["ksinkr"][s][n][:,i])\
                for j in range(3)] for i in range(params["r_vec"][s][n].shape[0])]) \
                for n in range(params["N_atoms"][s])] \
                for s in range(params["N_bonds"])]


pair_map, emb_neigh_map, emb_map = fe.nonlin.get_mappers(params,mapper)

global rho_scaling, seed, t, X, mapper, params, x0, smooth_emb, smooth_pair, \
        f_smooth, f_smooth_emb, r_smooth, rho_dict, r_lb, r_ub, r_cut, rho_lb,\
        rho_ub, N_steps, emb_n, emb_i, pair_n, pair_map, emb_neigh_map, emb_map, \
        _W_emb, _W_pair, _W_emb_neigh

#@profile
def setup_weights(tmp_x):
    _W_emb = [np.take(tmp_x,emb_map[s],axis=0) for s in range(params["N_bonds"])]
    _W_pair = [[np.take(tmp_x,pair_map[s][n],axis=0)
             for n in range(params["N_atoms"][s])] \
             for s in range(params["N_bonds"])]
    _W_emb_neigh = [[np.take(tmp_x,emb_neigh_map[s][n],axis=0)
             for n in range(params["N_atoms"][s])] \
             for s in range(params["N_bonds"])]
    
    return _W_emb,_W_pair,_W_emb_neigh

def get_splined_functions_fast(weights,verbose=False):
    # assumes that the following are global:
    # mapper,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,r_lb=0,
    #                      r_ub=5.,rho_lb=-5,rho_ub=42.,N_steps=100,rho_scaling=1.
    
    def _simple(x):
        if isinstance(x,np.ndarray):
            return np.ones(x.shape)
        else:
            return 1.

    if verbose: print("Splining energy functions...")
    if smooth_emb:
        smooth_fun_rho = fe.nonlin.wrapped_smooth(f_smooth_emb,0,kind="<") # the contribution of the embedding energy is zero if the background density is zero
    else:
        smooth_fun_rho = _simple
        
    if smooth_pair:
        smooth_fun_r = fe.nonlin.wrapped_smooth(f_smooth,r_smooth,kind=">")
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
    if verbose:
        print("drho {}".format(drho))
        print("dr {}".format(dr))   

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

def calculate_all_forces_fast2(x):
    """
    new improved version of the function calculate_all_forces_fast
    """
    N_bonds = params["N_bonds"]
    N_atoms = params["N_atoms"]

    tmp_x = np.array(x,dtype=float)
    
    all_forces = [np.zeros((params["N_atoms"][v_s],3)) for v_s in range(params["N_bonds"])]
    
    #tmp_emb_force = np.zeros(3,dtype=np.float64)
    #tmp_pair_force = np.zeros(3,dtype=np.float64)
    
    #tmp_emb_n = np.zeros(3,dtype=np.float64)
    #tmp_emb_i = np.zeros(3,dtype=np.float64)
    #tmp_pair_n = np.zeros(3,dtype=np.float64)
    #tmp_pair_i = np.zeros(3,dtype=np.float64)

    #all_emb_forces = [np.zeros((N_atoms[s],3),dtype=float) for s in range(N_bonds)]
    #all_pair_forces = [np.zeros((N_atoms[s],3),dtype=float) for s in range(N_bonds)]

    fun_emb_i = lambda s,n: np.array([params["fprho_i"][s][n][i] * \
        np.dot(tmp_x[mapper["emb"][params["emb_species"][s][n][i]]], params["psiprho"][s][params["neigh_idx_super"][s][n][i]] * \
        params["coskrho"][s][:,params["neigh_idx_super"][s][n][i]] - params["psiprho"][s][params["neigh_idx_super"][s][n][i]]*params["ksinkrho"][s][:,params["neigh_idx_super"][s][n][i]])\
        for i in range(params["fprho_i"][s][n].shape[0])],\
        dtype=float).sum(axis=0)
    
    fun_emb_n = lambda s,n: np.array(params["fprho_n"][s][n] * np.dot(tmp_x[mapper["emb"][params["species"][s][n]]], \
        params["psiprho"][s][n]*params["coskrho"][s][:,n] - params["psiprho"][s][n] * params["ksinkrho"][s][:,n]),\
        dtype=float)

    fun_pair_n = lambda s,n: np.array([params["r_vec"][s][n][i] * (params["psipr"][s][n][i] *\
        np.dot(tmp_x[mapper["pair"][params["pair_species"][s][n][i]]],params["coskr"][s][n][:,i]) \
        - params["psir"][s][n][i] * np.dot(tmp_x[mapper["pair"][params["pair_species"][s][n][i]]],params["ksinkr"][s][n][:,i]))\
        for i in range(params["r_vec"][s][n].shape[0])],\
        dtype=float).sum(axis=0)

    
    for s in range(params["N_bonds"]):
        
        for n in range(params["N_atoms"][s]):
            
            all_forces[s][n,:] = fun_emb_n(s,n) + fun_emb_i(s,n) + fun_pair_n(s,n)
    return all_forces

def get_forces_from_splines_fast(x):
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

def force_spline_wrapper_fast(): # temporary solution to a bug in the analytic force calculation...
    def wrapped_force_fun_fast(x):
        return get_forces_from_splines_fast(x)
    return wrapped_force_fun_fast

def calculate_all_energies_fast(x):
    # also requires params and mapper, but assumes thay are global
    tmp_x = np.array(x,dtype=float)
        
    all_energies = np.zeros(params["N_bonds"])
    all_pair_energies = np.zeros(params["N_bonds"])
    
    # embedding
    tmp_emb_energy = np.zeros(params["N_bonds"])
    tmp_pair_energy = np.zeros(params["N_bonds"])
    
    for s in range(params["N_bonds"]):  
        tmp = np.dot(params["psirho"][s], np.array([np.dot(tmp_x[mapper["emb"][params["species"][s][v_n]]], params["coskrho"][s][:,v_n]) \
                      for v_n in range(params["N_atoms"][s])])) / float(params["N_atoms"][s])
        tmp_emb_energy[s] += tmp
        
    # pair
    for s in range(params["N_bonds"]):
        tmp = np.sum([np.dot( params["psir"][s][v_n], np.array([np.dot(tmp_x[mapper["pair"][params["pair_species"][s][v_n][v_i]]], params["coskr"][s][v_n][:,v_i]) \
               for v_i in range(params["coskr"][s][v_n].shape[1])])) \
               for v_n in range(params["N_atoms"][s])]) / float(params["N_atoms"][s])
        tmp_pair_energy[s] += tmp
    
    #return tmp_emb_energy, tmp_pair_energy
    return tmp_emb_energy + .5* tmp_pair_energy

#@profile
def calculate_all_energies_fast_new(x,W_emb,W_pair):
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

def calculate_all_forces_fast(x):
    """
    params = {"coskrho":coskrho,"psirho":psirho,"coskr":coskr,"psir":psir, # energy related
              "ksinkr":ksinkr,"psipr":psipr,"ksinkrho":ksinkrho,"psiprho":psiprho, # force related
              "N_bonds":N_super,"N_atoms":N_atoms,
              "emb_species":X["emb_species"],"pair_species":X["pair_species"]}
    """
    tmp_x = np.array(x,dtype=float)
    #print("tmp_x {}".format(tmp_x))
    all_forces = [np.zeros((params["N_atoms"][v_s],3)) for v_s in range(params["N_bonds"])]
    
    tmp_emb_force = np.zeros(3,dtype=np.float64)
    tmp_pair_force = np.zeros(3,dtype=np.float64)
    
    tmp_emb_n = np.zeros(3,dtype=np.float64)
    tmp_emb_i = np.zeros(3,dtype=np.float64)
    tmp_pair_n = np.zeros(3,dtype=np.float64)
    tmp_pair_i = np.zeros(3,dtype=np.float64)
    for s in range(params["N_bonds"]):
        for n in range(params["N_atoms"][s]):
            
            curr_species = params["species"][s][n]
                                                
            ### embedding
            # dE(rho_n)/drho_n
            w_emb_n = tmp_x[mapper["emb"][curr_species]]

            fprho_n = params["fprho_n"][s][n]
            psiprho_n = params["psiprho"][s][n]
            coskrho_n = params["coskrho"][s][:,n]
            psirho_n = params["psirho"][s][n]
            ksinkrho_n = params["ksinkrho"][s][:,n]

            tmp_emb_n[:] = 0.
            
            tmp_emb_n[:] = fprho_n * np.dot(w_emb_n, psiprho_n*coskrho_n - psirho_n * ksinkrho_n)
                        
            # dE(rho_i)/drho_n
            tmp_emb_i[:] = 0
            
            for i in range(params["fprho_i"][s][n].shape[0]):
                idx = params["neigh_idx_super"][s][n][i] # index in supercell corresponding to neighbor found in ultracell
                neigh_species = params["emb_species"][s][n][i]
                w_emb_i = tmp_x[mapper["emb"][neigh_species]]
                fprho_i = params["fprho_i"][s][n][i]

                psiprho_i = params["psiprho"][s][idx]
                coskrho_i = params["coskrho"][s][:,idx]
                psirho_i = params["psirho"][s][idx]
                ksinkrho_i = params["ksinkrho"][s][:,idx]

                tmp_emb_i += fprho_i * np.dot(w_emb_i, psiprho_i*coskrho_i - psirho_i*ksinkrho_i)
            
            # total embedding force on atom n of supercell s
            tmp_emb_force[:] = 0.
            tmp_emb_force = tmp_emb_n + tmp_emb_i

            ### pair
    
            # dE(r_n)/dr_n
            tmp_pair_n[:] = 0.
            for i in range(params["r_vec"][s][n].shape[0]):
                r_vec_i = params["r_vec"][s][n][i] # (r_n - r_l)/r_nl
                psipr_i = params["psipr"][s][n][i]
                pair_i = params["pair_species"][s][n][i]
                w_pair_i = tmp_x[mapper["pair"][pair_i]]
                coskr_i = params["coskr"][s][n][:,i]
                psir_i = params["psir"][s][n][i]
                ksinkr_i = params["ksinkr"][s][n][:,i]
                tmp_pair_n += r_vec_i * (psipr_i * np.dot(w_pair_i,coskr_i) - psir_i * np.dot(w_pair_i,ksinkr_i))
            
            # dE(r_i)/dr_n
            
            # total pair force on atom n of supercell s
            tmp_pair_force[:] = tmp_pair_n #+ tmp_pair_i

            ### total
            all_forces[s][n] = tmp_emb_force + tmp_pair_force
    
    return all_forces

#@profile
def calculate_all_forces_fast_new(x,W_emb,W_pair_neigh,W_emb_neigh):
    
    tmp_x = np.array(x,dtype=float)
    
    all_forces = np.array([np.einsum("ij,ikj->ik",W_emb[s],emb_n[s]) for s in range(params["N_bonds"])])
        
    for s in range(params["N_bonds"]):
        f_emb_i = np.zeros((params["N_atoms"][s],3))
        f_pair = copy.deepcopy(f_emb_i)
        
        for n in range(params["N_atoms"][s]):            
            all_forces[s][n,:] += np.einsum("ij,ikj->ik",W_pair_neigh[s][n],pair_n[s][n]).sum(axis=0) \
                                  + np.einsum("ij,ikj->ik",W_emb_neigh[s][n],emb_i[s][n]).sum(axis=0) # f_pair + f_emb_i
        #all_forces[s] *= -1
    return all_forces

def calculate_all_forces_fast_new_old(x,W_emb,W_pair_neigh,W_emb_neigh):
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

#@profile
def bayes_evidence(x,t,alphas,betas):
    """Calculates the log likelihood for new weight parameters as provided with x.
    """
    tmp_x = np.array(x,dtype=float)
        
    #t0 = time.time()
    W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x) #fe.nonlin.setup_weights(tmp_x,params,mapper)
    W_pair_neigh = W_pair
    #print("setting up weights {} s...".format(time.time()-t0))
    #t0 = time.time()
    energies = calculate_all_energies_fast_new(tmp_x,W_emb,W_pair)
    #print("energies {} s...".format(time.time()-t0))

    #t0 = time.time()
    if force_analytic:
        forces = calculate_all_forces_fast_new(tmp_x,W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = get_forces_from_splines_fast(tmp_x)
    #print("forces {} s...".format(time.time()-t0))
    #t0 = time.time()
    N = len(t["forces"])
    M = len(x)
    
    logp_reg = -.5 * np.dot(alphas,tmp_x**2)
    logp_evi_energy = -.5 * betas[0] * np.sum((energies-t["energy"])**2)
    logp_evi_force = -.5 * betas[1] * np.sum([np.sum((np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))**2)/float(len(forces[v_s])) for v_s in range(N)])
    logp_hyp = .5 * (N+M) * np.log(2*np.pi) + .5 * np.sum(np.log(alphas)) + .5 * len(betas) * N * np.sum(np.log(betas))
    #print("rest {} s...".format(time.time()-t0))
    
    return - (logp_reg + logp_evi_energy + logp_evi_force + logp_hyp)
    #return - (logp_reg + logp_evi_energy + logp_hyp)

def energy_wrapper_fast():
    def wrapped_energy_fun_fast(x): # x is an np.ndarray containing the weights
        E = calculate_all_energies_fast(x)
        return E[0] + .5*E[1]
    return wrapped_energy_fun_fast

def least_square_measure_mod(x,t=None):
    """
    Least square measure provided the references, the functions and the weights.
    Energies and forces are being calculated.
    """
    tmp_x = np.array(x,dtype=float)
    W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x)#fe.nonlin.setup_weights(tmp_x,params,mapper)
    W_pair_neigh = W_pair
    energies = calculate_all_energies_fast_new(tmp_x,W_emb,W_pair)
    if force_analytic:
        forces = calculate_all_forces_fast_new(tmp_x,W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = get_forces_from_splines_fast(tmp_x)
    N = len(t["forces"])
    M = len(x)
    Natoms = np.array([len(v) for v in t["forces"]])
    e_error = np.linalg.norm( (energies - t["energy"]) )
    f_error = sum([np.sum(np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))/Natoms[v_s] for v_s in range(N)]) / float(3)
    #print("e_error {} f_error {}".format(e_error,f_error))
    return e_error + f_error

def least_square_measure_energy(x,t=None,e_fun=None):
    energies = e_fun(x)    
    N = len(t["energy"])
    M = len(x)
    
    e_error = np.linalg.norm( (energies - t["energy"]) )
    
    return e_error

class calculator(dict):
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
    """
    def __init__(self,params,targets,opts=dict(),reg_fun=None,evl_fun=None,fitness_tuple=False):
        """
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
        super().__init__()
        
        # initializing
        self.reg_fun = reg_fun
        self.evl_fun = evl_fun
        self.t = targets
        self.params = params
        self.opts = opts
        self.fitness_tuple = fitness_tuple

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
        if self.evl_fun.__name__ == "bayesian_measure":
            alphas = self.params["alphas"]
            betas = self.params["betas"]
            e_fun = self.params["e_fun"]
            f_fun = self.params["f_fun"]
            val = self.evl_fun(x,alphas,betas,self.t,e_fun,f_fun)
            
        elif self.evl_fun.__name__ == "least_square_measure":
            e_fun = self.params["e_fun"]
            f_fun = self.params["f_fun"]
            val = self.evl_fun(x,self.t,e_fun,f_fun)
        elif self.evl_fun.__name__ == "least_square_measure_energy":
            e_fun = self.params["e_fun"]
            val = self.evl_fun(x,self.t,e_fun)
        elif self.evl_fun.__name__ == "least_square_measure_mod":
            val = self.evl_fun(x,self.t)
        elif self.evl_fun.__name__ == "bayes_evidence":
            alphas = self.params["alphas"]
            betas = self.params["betas"]
            val = self.evl_fun(x,self.t,alphas,betas)
            #timer = timeit.Timer(lambda: self.evl_fun(x,self.t,alphas,betas))
            #print("bayes {}".format(timer.timeit(number=400)))
            
        else:
            raise NotImplementedError
        
        if self.fitness_tuple:
            return (val,)
        else:
            return val
            
    def run(self):
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


# necessary for optimizer
if minimize_method == "deap-cma":
    fitness_tuple = True # True if deap is being used, False otherwise
else:
    fitness_tuple = False # True if deap is being used, False otherwise

# setting up energy and force functions for the optimization
#e_fun = fe.nonlin.energy_wrapper(params,mapper)
#e_fun = calculate_all_energies_fast
e_fun = calculate_all_energies_fast_new
#f_fun = fe.nonlin.force_wrapper(params,mapper)
#f_fun = calculate_all_forces_fast
#f_fun = fe.nonlin.force_spline_wrapper(params,mapper,X,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,
#                     r_lb=r_lb,r_ub=r_cut,rho_lb=rho_lb,rho_ub=rho_ub,N_steps=N_steps,rho_scaling=rho_scaling)
if force_analytic:
    f_fun = calculate_all_forces_fast_new
else:
    f_fun = get_forces_from_splines_fast

# least squares
#calc_params = {"e_fun":e_fun,"f_fun":f_fun}
#calc_params = {}
#calc = calculator(calc_params,t,opts=dict(),evl_fun=fe.nonlin.least_square_measure,fitness_tuple=fitness_tuple)

from deap import base, creator, tools, cma
creator.create("FitnessMin", base.Fitness, weights=(-1.,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#@profile
def main(fun,N,MAXITER=100,verbose=True,init_lb=-4,init_ub=4,sigma=10.,lambda_=None,weights=None,std_lb=1e-4):
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
    toolbox.register("evaluate", fun)

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #lambda_=None,sigma=None,init_lb=-4,init_ub=4
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

    #logbooks.append(tools.Logbook())
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    print("\nDoing CMA...")    

    #toolbox = base.Toolbox()    
    
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
    
def wrapper_bayes_noise(x,t,alphas):
    alphas = np.absolute(alphas)
    tmp_x = np.array(x,dtype=float)
    W_emb,W_pair,W_emb_neigh = setup_weights(tmp_x)#fe.nonlin.setup_weights(tmp_x,params,mapper)
    W_pair_neigh = W_pair
    energies = calculate_all_energies_fast_new(tmp_x,W_emb,W_pair)
    
    if force_analytic:
        #forces = calculate_all_forces_fast(tmp_x)
        forces = calculate_all_forces_fast_new(tmp_x,W_emb,W_pair_neigh,W_emb_neigh)
    else:
        forces = get_forces_from_splines_fast(tmp_x)
    
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
        #return - (logp_reg + logp_evi_energy + logp_hyp)
    return bayes_noise
        
def wrapper_bayes_regularizer(x,betas):
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

def least_square_measure_energy(x,t=None,e_fun=None):
    energies = e_fun(x)    
    N = len(t["energy"])
    M = len(x)
    
    e_error = np.linalg.norm( (energies - t["energy"]) )
    
    return e_error

if __name__=="__main__":
    """
    graphviz = GraphvizOutput()
    graphviz.output_file = 'callgraph_eam_fit_CMA_fast_opt_parallel.png'

    with PyCallGraph(output=graphviz):
        main()
    """
    # optimization itself
    t0 = time.time()
    print("minimize_method {}".format(minimize_method))
    N = len(x0)
    if alphas_init is None or not "alphas_init" in locals():
        alphas_init = np.ones(N,dtype=float)
    if betas_init is None:
        betas_init = np.array([75.,42.],dtype=float) # betas[0]: energy, betas[1]: force
    weights = None

    for i_iter in range(maxiter_all):
        print("\n ============== global iteration {}/{} ==============\n".format(i_iter+1,maxiter_all))
        if i_iter == 0:
            alphas_curr = np.array(alphas_init)
            betas_curr = np.array(betas_init)
                
        print("alphas {}\nbetas {}\ninitial weights {}".format(alphas_curr,betas_curr,weights))
        
        # optimization of the weights
        calc_params = {"alphas":alphas_curr,"betas":betas_curr}
        calc = calculator(calc_params,t,opts=dict(),evl_fun=bayes_evidence,fitness_tuple=fitness_tuple)
        
        if minimize_method == "deap-cma":
            hof, logbook = main(calc,len(alphas_curr),MAXITER=int(maxiter),verbose=True,lambda_=lambda_,weights=weights)
            weights = np.array(hof[0],dtype=float)
            print("weight optimization results:\n    weights: {}\n    fitness: {}".format(weights,min([v["min"] for v in logbook])))
        else:
            raise NotImplementedError
        
        if i_iter+1 == maxiter_all:
            continue
        
        # optimization of the noise levels
        if not fix_betas:
            noise_fun = wrapper_bayes_noise(weights,t,alphas_curr)
            res_noise = optimize.minimize(noise_fun,betas_curr,method=minimize_method_noise,options={"maxiter":maxiter_hyper,"disp":True})
            betas_curr = np.absolute(res_noise["x"])
            print("noise optimization results:\n    betas: {}\n    fitness: {}".format(betas_curr,res_noise["fun"]))
            
        else:
            betas_curr = betas_init
        
        # optimization of the weight priors
        
        regularizer_fun = wrapper_bayes_regularizer(weights,betas_curr)            
        res_regularizer = optimize.minimize(regularizer_fun,alphas_curr,method=minimize_method_regularizer,options={"maxiter":maxiter_hyper,"disp":True})    
        alphas_curr = np.absolute(res_regularizer["x"])
        print("weight prior optimization results:\n    alphas: {}\n    fitness: {}".format(alphas_curr,res_regularizer["fun"]))
            
    res = None

    reg_time = time.time() - t0
    print("regression finished {}s...".format(reg_time))

    # comparing regressed and original values
    W_emb,W_pair,W_emb_neigh = setup_weights(weights)#fe.nonlin.setup_weights(weights,params,mapper)    
    W_pair_neigh = W_pair
    reg_e = e_fun(weights,W_emb,W_pair)        
    reg_f = f_fun(weights,W_emb,W_pair_neigh,W_emb_neigh)

    #Es = fe.nonlin.get_splined_functions(weights,mapper,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,r_lb=r_lb,
    #                           r_ub=r_cut,rho_lb=rho_lb,rho_ub=rho_ub,N_steps=N_steps,rho_scaling=rho_scaling)
    Es = get_splined_functions_fast(weights)

    fe.nonlin.save_Es(save_Es_path,Es,X,t,calc,regressed_e=reg_e,regressed_f=reg_f,dft_path=dft_path,load_path_rhos=load_path_rhos,rho_dict=rho_dict,num_neigh=num_neigh,r_cut=r_cut,aniso=aniso,
                ultra_num=ultra_num,selection=selection,seed=seed,k_pair=k_pair,k_emb=k_emb,smooth_emb=smooth_emb,
                smooth_pair=smooth_pair,type_pair=type_pair,type_emb=type_emb,r_smooth=r_smooth,f_smooth=f_smooth,
                minimize_method=minimize_method,maxiter=maxiter,lambda_=lambda_,eam_path=eam_path,reg_time=reg_time,
                res=res,logbook=logbook,fitness_tuple=fitness_tuple,r_lb=r_lb,r_ub=r_ub,rho_lb=rho_lb,rho_ub=rho_ub,
                N_steps=N_steps,rho_scaling=rho_scaling,weights=weights,alphas=alphas_curr,betas=betas_curr,force_analytic=force_analytic)
