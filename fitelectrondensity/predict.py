import numpy as np
import itertools, warnings
from scipy import spatial
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from fitelectrondensity.misc import local_bond_info, get_ultracell, get_quality_measures_fun

def predict_rho_iso(mapper, weights, basis, r_smooth, Nsteps):
    """Computes the 2-body electron density functions.

    Parameters
    ----------

    mapper : list of tuples
        Indicates, by index, which functions in basis["functions"] relates
        to which element and body approximation. 
        Example : [('Al', 0, 49), ('Ni', 49, 98)]
    
    weights : np.ndarray of shape (M,)
        Model parameters.
    
    basis : dict
        "functions" : list of basis functions
        "info" : information on the basis setup

    r_smooth : float
        Distance to taper basis functions to zero at.

    Nsteps : int
        Number of points to evaluate the electron densities at.
    
    Returns
    -------
    r : np.ndarray
        0 to r_smooth in Nsteps

    rhos : dict of np.ndarrays
    """
    r = np.linspace(0,r_smooth,Nsteps)
    dr = np.round(r_smooth/float(Nsteps),decimals=6)
    r = np.array([dr*v for v in range(Nsteps)],dtype=float)
    print("Calculating isotropic rhos: N_steps = {}, r_smooth = {}, incrementing by {} up to {}...".format(Nsteps,r_smooth,dr,r[-1]))
    rhos = {}
    for s,ix_s,ix_e in mapper:
        if isinstance(s,str):
            tmp_Phi = np.array([f(r) for f in basis["functions"][ix_s:ix_e]] ,dtype=float).T
            t_pred = np.dot(tmp_Phi,weights[ix_s:ix_e])
            rhos[s] = t_pred
    return r, rhos

def predict_rho_ani_q(mapper,weights,basis,Nsteps,lb={0:0},ub={0:1}):
    """
    Returns
    -------
    q : np.ndarray
        0 to 1 in Nsteps (if standard BOP)
    rhos : dict of np.ndarrays
    """
    
    q = {}
    for l in lb.keys():
        
        dq = np.round((ub[l]-lb[l])/float(Nsteps),decimals=6)
        q[l] = np.array([lb[l]+dq*v for v in range(Nsteps)],dtype=float)
    
    print("Calculating anisotropic (q) rhos: N_steps = {} ...".format(Nsteps))
    rhos = {}
    for s,ix_s,ix_e in mapper:
        print("s {}".format(s))
        if isinstance(s,tuple) and  s[1] in {"q","q-r","q-1/r"}:
            tmp_Phi = np.array([[f(v) for f in basis[ix_s:ix_e]] for v in q[s[2]]],dtype=float)
            t_pred = np.dot(tmp_Phi,weights[ix_s:ix_e])
            rhos[s] = t_pred
    return q, rhos

def _select_density_points(s,selection):
    xyz, t = s["edensity"]["xyz"], s["edensity"]["density"]

    N = len(t)
    if selection[0] == "random":
        if len(selection)>2:
            if selection[2] == "absolute":
                Nsamples = int(selection[1])
            else:
                Nsamples = int(selection[1]*1e-2*N)
        else:
            Nsamples = int(selection[1]*1e-2*N)
        if Nsamples>N:
            Nsamples = N
        idx = np.random.choice(range(N),size=Nsamples)
        
    elif selection[0] == "importance_mag":
        if len(selection)>2:
            if selection[2] == "absolute":
                Nsamples = int(selection[1])
            else:
                Nsamples = int(selection[1]*1e-2*N)
        else:
            Nsamples = int(selection[1]*1e-2*N)
        if Nsamples>N:
            Nsamples = N
        rho_min, rho_max = np.amin(t), np.amax(t)
        drho = rho_max - rho_min
        prob = np.array([(v-rho_min)/(drho) for v in t])
        prob /= prob.sum()
        
        if Nsamples < N:
            idx = np.random.choice(np.arange(N),size=Nsamples,p=prob,replace=False)
        else:
            idx = np.arange(N)    
    elif selection[0] == "atom":
        skd_edensity = spatial.KDTree(xyz)
        if selection[2] == "r":
            idx = skd_edensity.query_ball_point(np.dot(s["positions"],s["cell"]),selection[1])
        elif selection[2] == "N":
            _, idx = skd_edensity.query(np.dot(s["positions"],s["cell"]),k=selection[1])
        else:
            raise NotImplementedError
        idx = [list(v) for v in idx]
        idx = np.array(list(set([v for v2 in idx for v in v2])),dtype=int)
        
    elif selection[0] == "all":
        idx = np.arange(N)
    else:
        raise NotImplementedError("selection type {} not implemented!".format(selection))

    xyz, t = xyz[idx], t[idx]
    return xyz,t

# predict density via a spline version of rho
def predict_n_of_r(r_smooth,r,rhos,s,ultra_num=2,r_cut=None,num_neigh=None,aniso=False,selection=("all",),
                   q=None,rhos_ani_q=None,ani_specification=None,ani_type=None): # add aniso option
    xyz, pred_density = [], []
    
    xyz,t = _select_density_points(s,selection)

    # generating ultracell
    cell = s["cell"]
    ultracell_pos, ultracell_spec = get_ultracell(s["positions"],s["cell"],s["species"],ultra_num=ultra_num,r_cut=r_cut,search_pos=None)
    
    # searching atom neighbors of electron density points
    skd = spatial.KDTree(ultracell_pos)
    
    if r_cut is not None:
        idx = skd.query_ball_point(xyz,r_cut)
        
    elif num_neigh is not None:
        _,idx = skd.query(xyz,k=num_neigh)

    # predicting the electron density values 
    iso_rhos = {k: spline(r,rho,ext=0) for k,rho in rhos.items() if isinstance(k,str)}
    if aniso: 
        warnings.warn("Predictions is chosen to be done using anisotropy. Be aware that there is a bug related to the splining which leads to a divergence from expected values.")
        print("rhos_ani_q {}".format(rhos_ani_q.keys()))
        print("q {}".format(q.keys()))
        ani_rhos = {(el,qtag,l): spline(q[l],rho,ext=3) for (el,qtag,l),rho in rhos_ani_q.items()}
        q_tag = list(ani_rhos.keys())[0][1]
        print("q_tag {}".format(q_tag))
        

    zero = 0
    not_zero = 0
    
    for i in range(len(xyz)):
        bond = local_bond_info(aniso=aniso,ani_type=ani_type,ani_specification=ani_specification)
        neigh_idx = idx[i]
        target = {"density":np.array([t[i]],dtype=float)}
        bond.set_info(xyz[i],neigh_idx,ultracell_pos,ultracell_spec,cell,target)
        
        # iso contribution
        tmp_rho = sum([sum([iso_rhos[k](v) for v in bond.x["r"][k]]) if k in bond.x["r"] else 0 for k in iso_rhos.keys()])
        
        # ani contribution
        if aniso:
            
            for el in bond.x["ani"]:
                
                for j,qval in enumerate(bond.x["ani"][el]):
                    l = ani_specification["l_range"][j]
                    tmp_rho += ani_rhos[(el,q_tag,l)](qval)
            
        pred_density.append(tmp_rho)
        
        if tmp_rho > 1e-6:
            not_zero += 1
        else:
            zero += 1
            
    return xyz, pred_density, t
