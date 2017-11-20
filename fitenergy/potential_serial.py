import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import warnings

def calculate_all_energies(W_emb,W_pair,params):
        # also requires params and mapper, but assumes thay are global
                    
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

def calculate_all_forces(W_emb,W_pair_neigh,W_emb_neigh,params,emb_n,emb_i,pair_n):
                
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

def wrapped_smooth_prime(f_s,r_s,kind=None):
    def smooth_fun_prime(r):
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
        a = 4.*x**3
        b = (1.+x**4)**2
        return a/b/float(f_s)
    return smooth_fun_prime

def get_splined_functions(weights,data,verbose=False):
        
        def _simple(x):
            if isinstance(x,np.ndarray):
                return np.ones(x.shape)
            else:
                return 1.
        smooth_emb, smooth_pair = data["smooth_emb"], data["smooth_pair"]
        f_smooth_emb, f_smooth = data["f_smooth_emb"], data["f_smooth"]
        r_smooth = data["r_smooth"]
        rho_lb, rho_ub = data["rho_lb"], data["rho_ub"]
        r_lb, r_ub = data["r_lb"], data["r_ub"]
        N_steps = data["N_steps"]
        rho_scaling = data["rho_scaling"]
        mapper = data["mapper"]
        rho_dict = data["rho_dict"]
        
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

def calculate_all_forces_from_splines(x,data):
    X = data["X"]
    N_super = X["N_bonds"]
    N_atoms = X["N_atoms"]
    
    Es = get_splined_functions(x,data)
    
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

def generate_derivative_of_iso_rho_splines(r,ys,kind="linear",fill_value="extrapolate",shift=1e-4,lb=0,ub=None):
    """Function to generate derivatives of isotropic rho functions in form
    of splines.
    
    The function uses the input r-y pairs, generates splines of these and uses 
    these splines to generate derivatives, also in form of splines.
    
    Parameters
    ----------
    
    r : np.ndarray
        radial distance
    
    ys : dict
        keys are elements and values are np.ndarrays of float mapping in the same order to r
    
    kind : str
        "linear", "quadratic", ... all those of scipy.interpolate.interp1d
    
    shift : float
        two axes will be generated which will be used to calculate the symmetric
        finite difference derivative of f. shift specifies by how much these
        axes are shifted relatively.
    
    fill_value : str or other
        value which will be passed to scipy.interpolate.interp1d's fill_value parameter
        
    Returns
    -------
    
    r_new : np.ndarray
        radial distance values used to create the derivative functions
    
    derivatives : dict
        keys are species and values are scipy.interpolate.interp1d splines
    """
    print("Generating iso rho derivatives...")
    if fill_value == "extrapolate": warnings.warn("Splining is done using the 'extrapolate' value for the fill_value parameter! This may cause problems, check for adequate cutoff and smoothing settings.")
    
    yps = {species: spline(r,ys[species],ext=0) for species in ys}
    derivatives = {}
        
    for species in ys:
        derivatives[species] = yps[species].derivative(n=1)
    
    return derivatives

def get_precomputed_energy_force_values(X,f_smooth,r_smooth,rho_dict,type_pair="Fourier",type_emb="Fourier",
                                        smooth_emb=True,smooth_pair=True,rho_scaling=1.,k_emb=5,k_pair=5,rho_shift=0,f_smooth_emb=None):
    
    print("Precomputing values for energies and forces...")
    print("smooth_emb {} smooth_pair {} rho_scaling {} r_smooth {} f_smooth {}".format(smooth_emb,smooth_pair,rho_scaling,
                                                                                       r_smooth,f_smooth))
    if type_pair != "Fourier" or type_emb != "Fourier":
        raise ValueError
    
    def _simple(x):
        if isinstance(x,np.ndarray):
            return np.ones(x.shape)
        else:
            return 1.
    def _simple_prime_wrap(f_s):
        def _simple_prime(x):
            if isinstance(x,np.ndarray):
                return np.ones(x.shape)
            else:
                return 1.
        return _simple_prime
    
    # set smoothers - 0th derivative
    if f_smooth_emb is None:
        f_smooth_emb = f_smooth
        
    if smooth_emb:
        #smooth_fun_rho = wrapped_smooth(f_smooth,0,kind="<") # the contribution of the embedding energy is zero if the background density is zero
        smooth_fun_rho = wrapped_smooth(f_smooth_emb,rho_shift) # the contribution of the embedding energy is zero if the background density is zero
    else:
        smooth_fun_rho = _simple
        
    if smooth_pair:
        smooth_fun_r = wrapped_smooth(f_smooth,r_smooth,kind=">")
    else:
        smooth_fun_r = _simple
        
    # set smoothers - 1st derivative
    if smooth_emb:
        smooth_fun_rhop = wrapped_smooth_prime(f_smooth_emb,0,kind="<") # the contribution of the embedding energy is zero if the background density is zero

    else:
        smooth_fun_rhop = _simple_prime_wrap(f_smooth_emb)
        
    if smooth_pair:
        smooth_fun_rp = wrapped_smooth_prime(f_smooth,r_smooth,kind=">")
    else:
        smooth_fun_rp = _simple_prime_wrap(f_smooth)
    
    N_super = X["N_bonds"]
    N_atoms = X["N_atoms"]
    
    # calculate - energy related : all_cosr, all_damping
    f_rho = np.pi/rho_scaling
    f_r = np.pi/r_smooth
    print("cos(rho) arguments = {}".format([f_rho*np.array(f_rho*rho) for rho in X["density"]]))
    coskrho = [np.array([np.cos(f_rho*k*np.array(rho)) for k in range(k_emb)]) for rho in X["density"]]
    psirho = [smooth_fun_rho(np.array(rho)) for rho in X["density"]]
    
    coskr = [[np.array([np.cos(f_r*k*np.array(r_atom)) for k in range(k_pair)]) for r_atom in supercell] for supercell in X["r"]]
    psir = [[np.array(smooth_fun_r(np.array(r_atom))) for r_atom in supercell] for supercell in X["r"]]
    
    # calculate - force related : normalized distance vector, psipr, psir, ksinkr, coskr
    rho_derivatives = generate_derivative_of_iso_rho_splines(rho_dict["r"],rho_dict["rhos"],shift=1e-5,lb=0,ub=r_smooth)

    ksinkrho = [np.array([f_rho*float(k)*np.sin(f_rho*k*np.array(rho)) for k in range(k_emb)]) for rho in X["density"]]
    psiprho = [smooth_fun_rhop(np.array(rho)) for rho in X["density"]]
    
    ksinkr = [[np.array([f_r*float(k)*np.sin(f_r*k*np.array(r_atom)) for k in range(k_pair)]) for r_atom in supercell] for supercell in X["r"]]
    psipr = [[np.array(smooth_fun_rp(np.array(r_atom))) for r_atom in supercell] for supercell in X["r"]]
     
    fprho_n = [np.zeros((N_atoms[s],3),dtype=np.float64) for s in range(N_super)]
    fprho_i = [[np.zeros((len(X["r"][s][i]),3),dtype=np.float64) for i in range(N_atoms[s])] for s in range(N_super)]

    for s in range(N_super): # iterate supercells
        for n in range(N_atoms[s]): # iterate atoms
                        
            N_neigh = len(X["r"][s][n])
            curr_species = X["species"][s][n]
            
            for i in range(N_neigh): # iterate neighbors of atom
                
                species = X["emb_species"][s][n][i]
                r = X["r"][s][n][i]
                
                tmp_fprho_n = rho_derivatives[species](r) # contribution to n from neighbor i
                tmp_fprho_i = rho_derivatives[curr_species](r) # contribution of n to neighbor i
                
                vec = X["r_vec"][s][n][i]
                
                fprho_n[s][n,:] += vec * tmp_fprho_n
                fprho_i[s][n][i] = vec * tmp_fprho_i
                          
    # storing all arguments for the energy and force calculation functions
    params = {"coskrho":coskrho,"psirho":psirho,"coskr":coskr,"psir":psir, # energy related
              "ksinkr":ksinkr,"psipr":psipr,"ksinkrho":ksinkrho,"psiprho":psiprho,"fprho_n":fprho_n,"fprho_i":fprho_i,"r_vec":X["r_vec"], # force related
              "N_bonds":N_super,"N_atoms":N_atoms,
              "emb_species":X["emb_species"],"pair_species":X["pair_species"],"species":X["species"],"neigh_idx_super":X["neigh_idx_super"]}
    
    return params