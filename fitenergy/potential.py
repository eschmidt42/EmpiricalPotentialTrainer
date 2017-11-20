import numpy as np
from scoop import shared
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def var(k,key=None):
    return shared.getConst(k) if key is None else shared.getConst(k)[key]

def calculate_all_energies(W_emb,W_pair):
        # also requires params and mapper, but assumes thay are global
        params = var("data",key="params")
            
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

def calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh):
        params, emb_n, emb_i, pair_n = var("data",key="params"), var("emb_n"), var("emb_i"), var("pair_n")
        
        all_forces = np.array([np.einsum("ij,ikj->ik",W_emb[s],emb_n[s]) for s in range(params["N_bonds"])])
            
        for s in range(params["N_bonds"]):
            
            for n in range(params["N_atoms"][s]):            
                all_forces[s][n,:] += np.einsum("ij,ikj->ik",W_pair_neigh[s][n],pair_n[s][n]).sum(axis=0) \
                                    + np.einsum("ij,ikj->ik",W_emb_neigh[s][n],emb_i[s][n]).sum(axis=0) # f_pair + f_emb_i
            
        return all_forces

def wrapped_smooth(f_s,r_s,kind=None):
    """
    Parameters
    ----------
    
    kind : str or None
        ">" zero for r > r_s
        "<" zero for r < r_s
    """
    def smooth_fun(r):
        x = (r-r_s)/float(f_s)
        if kind is not None:
            if isinstance(x,np.ndarray):
                lo = np.where(r<r_s)[0]
                hi = np.where(r>r_s)[0]
                if (kind == ">" and len(hi)>0):
                    x[hi] = 0.
                elif (kind == "<" and len(lo)>0):
                    x[lo] = 0.
            else:
                if (kind == ">" and r>r_s):
                    x = 0.
                elif (kind == "<" and r<r_s):
                    x = 0.
                        
        x4 = x**4
        return x4/(1.+x4)
    return smooth_fun

def get_splined_functions(weights,verbose=False):
        
        def _simple(x):
            if isinstance(x,np.ndarray):
                return np.ones(x.shape)
            else:
                return 1.
        smooth_emb, smooth_pair = var("data",key="smooth_emb"), var("data",key="smooth_pair")
        f_smooth_emb, f_smooth = var("data",key="f_smooth_emb"), var("data",key="f_smooth")
        r_smooth = var("data",key="r_smooth")
        rho_lb, rho_ub = var("data",key="rho_lb"), var("data",key="rho_ub")
        r_lb, r_ub = var("data",key="r_lb"), var("data",key="r_ub")
        N_steps = var("data",key="N_steps")
        rho_scaling = var("data",key="rho_scaling")
        mapper = var("data",key="mapper")
        rho_dict = var("data",key="rho_dict")
        
        if verbose: print("Splining energy functions...")
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

def calculate_all_forces_from_splines(x):
    X = var("data",key="X")
    N_super = X["N_bonds"]
    N_atoms = X["N_atoms"]
    
    Es = get_splined_functions(x)
    
    funs = {"pair":{p:spline(Es["r"],v) for p,v in Es["pair"].items()},
                "emb":{s:spline(Es["rho"],v) for s,v in Es["emb"].items()},
                "rhos":{r:spline(Es["r"],v) for r,v in Es["rhos"].items()},}
    derivatives = {"pair":{p:v.derivative(n=1) for p,v in funs["pair"].items()},
                "emb":{s:v.derivative(n=1) for s,v in funs["emb"].items()},
                "rhos":{r:v.derivative(n=1) for r,v in funs["rhos"].items()},}
    
    N_neighs = [[len(X["r"][s][n]) for n in range(N_atoms[s])] for s in range(N_super)]
    fun_fprho = lambda s,n: np.sum([X["r_vec"][s][n][i] * derivatives["rhos"][X["emb_species"][s][n][i]](X["r"][s][n][i]) for i in range(N_neighs[s][n])],axis=0)
    fun_emb_neigh = lambda s,n: np.sum([X["r_vec"][s][n][i] * derivatives["rhos"][X["species"][s][n]](X["r"][s][n][i]) * derivatives["emb"][X["emb_species"][s][n][i]](X["density"][s][X["neigh_idx_super"][s][n][i]]) for i in range(N_neighs[s][n])],axis=0)
    fun_forces_pair = lambda s,n: np.sum([X["r_vec"][s][n][i] * derivatives["pair"][tuple(sorted([X["species"][s][n],X["emb_species"][s][n][i]]))](X["r"][s][n][i]) for i in range(N_neighs[s][n])],axis=0)
    
    fast_all_forces = [np.array([fun_forces_pair(s,n) + fun_fprho(s,n) * derivatives["emb"][X["species"][s][n]](X["density"][s][n]) + fun_emb_neigh(s,n) for n in range(N_atoms[s])],dtype=float) for s in range(N_super)]
    
    return fast_all_forces