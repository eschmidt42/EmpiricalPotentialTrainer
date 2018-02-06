import matplotlib.pylab as plt
import numpy as np
from scipy import interpolate, spatial, special
import itertools
from mpl_toolkits.mplot3d import Axes3D
import pickle, time, warnings
from fortran import interface as f90

def get_quality_measures(t,weights,Phi,alpha,beta,N,M):
    """
    Returns just a bunch of quality/difference measures, such as mse, loglikelihood,...

    """
    t_pred = np.dot(Phi,weights)
    t_pred = np.reshape(t_pred,(-1,1))
    
    hyper = np.sum(np.log(alpha)) + N * np.log(beta)
    dt = t-t_pred
    dt2 = dt**2
    
    evidence = beta * np.sum(dt**2)
    ln2pi = np.log(2*np.pi)
    regularizer = np.dot(alpha,weights**2)
    
    log_joint = - .5*(M+N)*ln2pi + .5*N*np.log(beta) + .5*np.sum(np.log(alpha) - alpha*(weights**2)) - .5*evidence
    total_square_error = (dt2).sum()
    mean_square_error = (dt2).mean()
    median_square_error = np.median(dt2)
    min_delta = np.amin(dt)
    max_delta = np.amax(dt)
    dev_est = dt.mean()
    dev_std = dt.std(ddof=1)
    return {"L":log_joint,"tse":total_square_error,"mse":mean_square_error,
        "min":min_delta,"max":max_delta,"median_se":median_square_error,
        "dev_est":dev_est,"dev_std":dev_std}

def get_quality_measures_fun(t,t_pred,weights,alpha,beta):
    """
    Same as get_quality_measures returns just a bunch of quality/difference measures, such as mse, loglikelihood,...
    BUT this takes already regressed density values

    """
    M,N = len(weights), len(t)
    t_pred = np.reshape(np.array(t_pred),(-1,))
    t = np.reshape(np.array(t),(-1,))
    dt = t-t_pred
        
    hyper = np.sum(np.log(alpha)) + N * np.log(beta)
    evidence = beta * np.sum(dt**2)
    ln2pi = np.log(2*np.pi)
    regularizer = np.dot(alpha,weights**2)
    
    log_likelihood = - .5*(M+N)*ln2pi + .5*N*np.log(beta) + .5*np.sum(np.log(alpha) - alpha*(weights**2)) - .5*evidence
    total_square_error = np.sqrt(np.sum(dt**2))
    mean_square_error = np.sqrt(np.mean(dt**2))
    median_square_error = np.sqrt(np.median(dt**2))
    min_delta = np.amin(dt)
    max_delta = np.amax(dt)
    return {"L":log_likelihood,"tse":total_square_error,"mse":mean_square_error,"min":min_delta,"max":max_delta,"median_se":median_square_error}

def show_rho_iso(r,rhos,title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for s, rho in rhos.items():
        ax.plot(r,rho,'-',label=s,linewidth=2,alpha=0.6)
    
    ax.set_xlabel("r (Distance) [A]")
    ax.set_ylabel(r"$\rho$ (Density) [1/A^3]")
    plt.legend(loc=0)
    plt.title(title)
    plt.grid()
    plt.show()


def show_rho_ani_q(q,rhos,title=""):
    
    keys = sorted(rhos.keys(),key= lambda x: (x[2],x[0]))
    key_types = set([v[0] for v in keys])
    for key_type in key_types:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        keys = sorted([v for v in rhos.keys() if v[0]==key_type],key= lambda x: x[2])
        
        N = float(len(keys))
        for i,s in enumerate(keys):
            rho = rhos[s]
            ax.plot(q[s[2]],rho,'-',label="l={}".format(s[2]),linewidth=2,alpha=0.6,c=plt.cm.coolwarm(i/N))
        
        ax.set_xlabel("q (BOP)")
        ax.set_ylabel(r"$\rho$ (Density) [?/A^3]")
        plt.legend(loc=0)
        plt.title(title+" {} {}".format(s[1],key_type))
        plt.grid()
    plt.show()


def get_angles(r_vec):
    # theta
    theta = np.arctan2(r_vec[:,1],r_vec[:,0]) + np.pi
    theta = np.absolute(theta)
    # phi
    phi = np.arccos(r_vec[:,2])
    phi = np.absolute(phi)
    return theta, phi

def get_spherical_harmonics(m,l,theta,phi):
    Y = special.sph_harm(m,l,theta,phi)
    return Y

def get_q(theta,phi,l): #weighting by number of neighbors
    q = 0
    
    Nneighs = float(len(theta))
    m_range = np.arange(-l,l+1)
    
    for i,m in enumerate(m_range):
        Y = get_spherical_harmonics(m,l,theta,phi)
        sum_Y = np.absolute(np.sum(Y/Nneighs))**2
        q += sum_Y
        
    q *= 4*np.pi/(2.*l+1)
    q = np.sqrt(q)
    return q

def get_q_r(theta,phi,l,rs): #weighting by distances
    q = 0
    m_range = np.arange(-l,l+1)    
    Nneighs = len(theta)
    
    if Nneighs>1:    
        inv_rs = 1./rs
        inv_rs_sum = np.sum(inv_rs)
        
        fun = lambda v0: inv_rs[v0]/inv_rs_sum # smoothing function
        tmp_w = fun(np.arange(Nneighs))
        tmp_w /= tmp_w.sum()
        for i,m in enumerate(m_range):
            Y = get_spherical_harmonics(m,l,theta,phi)
            sum_Y = np.absolute(np.dot(tmp_w,Y))**2
            q += sum_Y
    else:
        for i,m in enumerate(m_range):
            Y = get_spherical_harmonics(m,l,theta,phi)[0]
            sum_Y = np.absolute(Y)**2
            q += sum_Y
        
    q *= 4*np.pi/(2.*l+1)
    q = np.sqrt(q)
    return q

def get_q_linr(theta,phi,l,rs,decimals=6): #linear weighting by distances
    q = 0
    m_range = np.arange(-l,l+1)    
    Nneighs = len(theta)

    if Nneighs > 1:
        rs_sum = round(np.sum(rs),decimals)
        
        fun = lambda v0: 1-rs[v0]/rs_sum # smoothing function
        tmp_w = fun(np.arange(Nneighs))
        tmp_w /= tmp_w.sum()
        
        
        for i,m in enumerate(m_range):
            Y = get_spherical_harmonics(m,l,theta,phi)
            sum_Y = np.absolute(np.sum(np.dot(tmp_w,Y)))**2
            q += sum_Y
    else:
        for i,m in enumerate(m_range):
            Y = get_spherical_harmonics(m,l,theta,phi)[0]
            sum_Y = np.absolute(np.sum(Y))**2
            q += sum_Y
      
    q *= 4*np.pi/(2.*l+1)
    q = np.sqrt(q)
    q = round(q,decimals)
    return q

def get_q_taper1(theta,phi,l,rs,rcut_aniso,decimals=6):
    """
    taper contribution of atom to q value, to 0 at r=0 and rcut_aniso

    weight(r) = (rcut-r)**2 * taper(r) * taper(rcut-r)

    with taper(x) = x**4 / (1+x**4) for x >=0 , 0 otherwise
    """
    def _taper(x,damping_factor):
        t = lambda x,scl : 0.0 if x <= 0.0 else (x*scl)**4/(1.0+(x*scl)**4)

        scl = 1.0/damping_factor

        return np.array([t(_x,scl) for _x in x])


    def _weight(x,rcut):
        tmp1 = (rcut-x)**2
        tmp2 = _taper(x=x,damping_factor=0.5)
        tmp3 = _taper(x=rcut-x,damping_factor=0.5)

        return np.array([tmp1[i]*tmp2[i]*tmp3[i] for i in range(len(x))])

    q = 0

    Nneighs = len(theta)
    m_range = np.arange(-l,l+1)

    # weights
    weights = _weight(x=rs,rcut=rcut_aniso)

    # normalise sum of weights to be 1
    inv_weight_total = 1.0/np.sum(weights)

    # normalised weights
    weights = np.multiply(weights,inv_weight_total)

    for i,m in enumerate(m_range):
        Y = get_spherical_harmonics(m,l,theta,phi)
        q += np.absolute(np.dot(Y,weights))
        #q += np.absolute(np.sum([Y[v0]*weights[v0] for v0 in range(Nneighs)]))

    q *= 4.0*np.pi/(2.0*l+1.0)
    q = np.sqrt(q)
    return round(q,decimals) 

class local_bond_info:
    """Class for the IO of neighboring info for a (ghost) atom.
    
    This class is intended for the use in electron density regression as well as
    empirical potential optimization.
    
    A ghost atom is a point in space used to do electron density regression
    in places between atoms. An atom is an atom.
    
    Methods
    -------

    __getitem__ :
        to return the attributes when called with "r", "phi", "r_idx" or "phi_idx".
    
    set_info : 
        takes ultra cell, list of elements, current (ghost) index, 
        and indices pointing to neighborhood of the current (ghost) atom
    
    get_x :
        returns input for regression machinery
    
    get_t : 
        returns reference to match for regression machinery
        
    Attributes
    ----------
    
    observables : dict of list of str
        {"x":["r","phi",...], "t":["E","F","density",..]
    
    cell : np.ndarray of float
        simulation cell
    
    r_idx :
        dict of list of ultracell indices of atoms for distance calculation,
        example: {"all":[1,2,3],"Al":[1],"Ni":[2,3]}
    
    phi_idx : np.ndarray of int
        dict of list of ultracell index pairs of atoms for angle calculation
        example: {"all":[(1,2),(1,3),(2,3)],("Al","Ni"):[(1,2),(1,3)],("Ni,"Ni"):[(2,3)],("Al,"Al"):[]}
    
    x : dict of np.arrays
        input values, i.e. {"r":[...],"r_vec":[...],"phi":[...]}
        "r_vec" = np.ndarray of float shape (N,3)
        "r" = np.ndarray of float shape (N,),distances
        "phi" = np.ndarray of float shape (.5*N*(N-1),), angles in [0,2pi[
    
    t : dict of np.ndarary of float
        target value(s), i.e. {"E":[42.],"F":[1.,2.,3.],"density":[21.],...}
    
    atom_pos : np.ndarray of float (3,)
        position of (ghost) atom in space
    
    atom_species : string or None
        if ghost atom this value is None, a string otherwise
    
    atom_index : integer or None
        if ghost atom this value is None, an integer otherwise
    """
    
    def __init__(self,observables={"x":["r","ani","r_vec"],"t":["density"]},aniso=False,ani_type=None,ani_specification={"l_range":np.arange(0,7)}):
        
        # setting input x and output t
        self.observables = observables
        for obs_type in observables:
            setattr(self,obs_type, {val: None for val in observables[obs_type]})
        
        # anisotropy configuration
        self.ani = aniso
        self.impl_ani_types = {"MEAM","BOP","BOP-r","BOP-invr","R","BOP4atoms","BOP4atoms-ref","BOP4atoms-ref2","BOP-taper1",\
                "polynomial"}
        self.ani_selected_keys = None
        self.ani_specification = ani_specification
        self.ani_type = ani_type

        if aniso:
            if  not ani_type in self.impl_ani_types:
                raise NotImplementedError
            if ani_type in {"BOP","BOP-invr","BOP-r","R","BOP4atoms","BOP4atoms-ref","BOP4atoms-ref2","MEAM","BOP-taper1","polynomial"}:
                if ani_type not in ["MEAM","polynomial"]:
                    assert ani_specification is not None and "l_range" in ani_specification, "Assertion failed - incompatible ani_specification for ani_type 'BOP'!"
                else:
                    assert "r_ani" in ani_specification,"Assertion failed - {} type must have r_ani specified".format(ani_type)
                    assert isinstance(ani_specification["r_ani"],(float,int)),"Assertion failed - r_ani not a float"

                    if "f_smooth" not in ani_specification:
                        # default smoothing scaling factor
                        self.ani_specification.update({"f_smooth":0.1})
                if "usage" in ani_specification:
                    if isinstance(ani_specification["usage"],str):
                        if ani_specification["usage"] == "everything":
                            ani_selected_keys = "everything"
                        else:
                            ani_selected_keys = [ani_specification["usage"]]
                    elif isinstance(ani_specification["usage"],(list,tuple)):
                        if "all" in ani_specification["usage"]:
                            ani_selected_keys = ["all"] + sorted([v for v in ani_specification["usage"] if v!="all"])
                        else:    
                            ani_selected_keys = sorted([v for v in ani_specification["usage"] if v!="all"])
                    else:
                        raise NotImplementedError
                else:
                    ani_selected_keys = "everything"
            else:
                ani_selected_keys = None
        else: 
            ani_selected_keys = None
        self.ani_selected_keys = ani_selected_keys
        
        # (ghost) atom info
        self.atom_pos = None
        self.atom_element = None
        self.atom_index = None
        
        # neighboring info
        self.r_idx = None
        self.ani_idx = None
        
        # safety check
        self.idx_attrs_to_check = ["r_idx","ani_idx"] #both should not be None after __call__
        
    def set_info(self,atom_pos,r_idx,ultracell_pos,ultracell_species,ultracell_cell,t,
                 atom_species=None,atom_index=None,usefortran=False):
        """
        Calculates properties such as distance vectors, distances and bond angles. these
        properties are then stored according to what has been defined with "observables" 
        during initialization. Further it stores r_idx and phi_idx which may be useful
        later on when referring to the original ultracell.
        
        Parameters
        ----------
        atom_pos : np.ndarray (3,)
            position in r space
        r_idx : list or np.ndarray of int
            ultracell indices found by neighboring search
        ultracell_pos : np.ndarray of float (N,3)
            atom positions in rspace
        ultracell_species : list of string 
            specifies chemical symbols of atoms in configurations, i.e. 
            chemical symbol of atom i is in ultracell_elements[i]
        ultracell_cell : np.ndarray (3,3) of float
            simulation box for ultracell
        t : dict
            contains target values, i.e. {"density":[21.],"E":[42.],"F":[1.,2.,3.],...}
        atom_species : str or None
            optional
        atom_index : int or None
            optional        
        """
        
        self.atom_pos = atom_pos
        self.atom_species = atom_species
        self.atom_indx = atom_index
        self.box = ultracell_cell
        r_idx = np.array(r_idx,dtype=int)
        self.r_idx = r_idx
        self.t = t
        
        if len(r_idx) == 0:
            r = {}
            ani_data = {}
            r_vec = {}
            r_idx = {}
            ani_idx = {}
        else:
            # r related
            r_vec = np.array(ultracell_pos[r_idx]) - atom_pos
            Nneigh = len(r_idx)
            
            # calculate all distances and normalize vectors
            r = {"all":np.linalg.norm(r_vec,axis=1)}
            r_vec /= np.reshape(r["all"],(-1,1))
            r_vec = {"all":r_vec}
            
            # find neighboring species and sort distances in ascending order
            neigh_species = np.array(ultracell_species)[r_idx]
            sorted_idx = np.argsort(r["all"])

            # store things in dictionaries
            r_idx = {"all":r_idx[sorted_idx]}            
            r["all"] = r["all"][sorted_idx]
            r_vec["all"] = r_vec["all"][sorted_idx]

            neigh_species = neigh_species[sorted_idx]

            # add species dependent entries to r_idx, r and r_vec
            for i in range(Nneigh):
                species = neigh_species[i] # this should be correct
                if species in r:
                    r[species].append(r["all"][i])
                    r_idx[species].append(r_idx["all"][i])
                    r_vec[species].append(r_vec["all"][i])
                else:
                    r[species] = [r["all"][i]] # distances for given species value
                    r_idx[species] = [r_idx["all"][i]] # indices for ultracell atoms
                    r_vec[species] = [r_vec["all"][i]] # normalized distance vectors for given species value
            
            for el in r:
                r_vec[el] = np.array(r_vec[el],dtype=float)
                r[el] = np.array(r[el],dtype=float)
                r_idx[el] = np.array(r_idx[el],dtype=int)
            
            # ani related section
                        
            ani_idx = {}
            ani_data = {}
            if self.ani:
                idx = list(range(Nneigh))
                if self.ani_selected_keys == "everything":
                    self.ani_selected_keys = ["all"] + [v for v in sorted(r.keys()) if v!="all"]
                
                # finding distances which are larger than zero amd storing their index to r
                idx_greater_zero = {el:np.where(np.logical_not(np.isclose(r[el],np.zeros(r[el].shape))))[0] for el in self.ani_selected_keys}
                                
                # storing the r, idx and r_vec values associated with non-zero distances
                tmp_r = {el:r[el][idx_greater_zero[el]] for el in self.ani_selected_keys}
                tmp_r_vec = {el:r_vec[el][idx_greater_zero[el]] for el in self.ani_selected_keys}

                tmp_idx = {el:r_idx[el][idx_greater_zero[el]] for el in self.ani_selected_keys}
                                
                # select distances for anisotropic calculation
                tmp_idx_ani = {}
                for el in tmp_r: 
                    if "r_ani" in self.ani_specification:
                        tmp_idx_ani[el] = np.where(tmp_r[el]<self.ani_specification["r_ani"])[0]
                    else:
                        tmp_idx_ani[el] = np.arange(len(tmp_r[el]))
                    
                if self.ani_type == "MEAM" or self.ani_type == "polynomial": 
                    
                    elements = sorted([v for v in tmp_r.keys() if v!="all"])
                    pairs = [(_el1,_el2) for v1,_el1 in enumerate(elements) for v2,_el2 in enumerate(elements) if v1<=v2]
                    ani_data = {}
                    for _el1,_el2 in pairs:
                        
                        if usefortran:
                            _tmp_before = [tuple(v) for v in f90.MEAM_aniso_bonds(_el1,_el2,tmp_r,tmp_r_vec,tmp_idx_ani)]
                        else:
                            # assume tmp_r_vec are unit vectors here !!
                           if _el1 == _el2:
                               idx_mix = [v for v in itertools.product(range(len(tmp_idx_ani[_el1])),range(len(tmp_idx_ani[_el2]))) if v[0]<v[1]] # v[0]<v[1] to avoid selecting a neighboring atom twice
                           else:
                               idx_mix = [v for v in itertools.product(range(len(tmp_idx_ani[_el1])),range(len(tmp_idx_ani[_el2])))]
                           
                           idx_mix = [(tmp_idx_ani[_el1][v],tmp_idx_ani[_el2][v2]) for v,v2 in idx_mix]
                           _tmp_before = [tuple([tmp_r[_el1][v1],tmp_r[_el2][v2],np.arccos(np.around(np.dot(tmp_r_vec[_el1][v1],tmp_r_vec[_el2][v2]),decimals=10))]) for v1,v2 in idx_mix]
                           
                        _tmp = np.array(_tmp_before,dtype=[("r0",np.float64),("r1",np.float64),("theta",np.float64)])
                        _tmp.sort(order=("r0","r1","theta"))
                        try:
                            _tmp = _tmp.view(np.float64).reshape(_tmp.shape+(-1,))
                        except ValueError:
                            raise ValueError("No neighbors within ani cutoff.")
                       
                        ani_data[(_el1,_el2)] = _tmp
                            
                        if np.isnan(ani_data[(_el1,_el2)]).any():
                            print(ani_data[(_el1,_el2)])
                            raise
                        if (ani_data[(_el1,_el2)][:,2]<0).any() or (ani_data[(_el1,_el2)][:,2]>np.pi).any():
                            print("b {} a {}".format(_tmp_before.shape,_tmp.shape))
                            print("<0 {}".format((ani_data[(_el1,_el2)][:,2]<0).any()))
                            print(">pi {}".format((ani_data[(_el1,_el2)][:,2]>np.pi).any()))
                            raise
                        
                    soert = sorted(ani_data.keys())
                    for _i,k in enumerate(soert):
                        if _i == 0:
                            ani_data["all"] = ani_data[k]
                        else:
                            ani_data["all"] = np.vstack((ani_data["all"],ani_data[k]))
                        
                    to_pop = []
                    for _pair in ani_data:
                        if len(ani_data[_pair])>0:
                            ani_data[_pair] = np.array(ani_data[_pair])
                        else:
                            to_pop.append(_pair)
                    for _pair in to_pop:
                        ani_data.pop(_pair,None)
                    
                
                elif self.ani_type == "R":
                    for el in tmp_r:
                        
                        if len(tmp_r[el]) == 0:
                            R = 0
                        else:
                            R = np.linalg.norm(np.sum(tmp_r_vec[el]*np.reshape(tmp_r[el],(-1,1)),axis=0))
                            eps = 0.1
                        
                        #rbf = np.exp(-(eps*R**2)) # bit spiky - Gaussian
                        rbf = np.sqrt(1.+(eps*R)**2) # not bad - multiquadric
                        #rbf = 1./(1.+(eps*R)**2) # bit spiky - inverse quadratic
                        #rbf = 1./np.sqrt(1.+(eps*R)**2) # not bad - inverse multiquadric
                        #rbf = R**2 * np.log(R) # not bad - thin plate
                        #rbf = R * np.log(R) # not bad
                        
                        ani_data[el] = rbf

                elif self.ani_type == "BOP4atoms":
                    Nl = len(self.ani_specification["l_range"])
                    ani_data = {el:np.zeros(Nl,dtype=float) for el in self.ani_selected_keys}
                    
                    for el in tmp_r:

                        if len(tmp_idx_ani[el])==0: # if no elements of that type are neighboring skip the following
                            continue
                        
                        # selecting the original supercell indices from the neighboring knowledge in the ultracell                            
                        _idx = [self.ani_specification["map_idx_ultra2super"][v] for v in tmp_idx[el][tmp_idx_ani[el]]]
                    
                        for i,l in enumerate(self.ani_specification["l_range"]):
                            if len(_idx) > 1:            
                                # retrieving the "l" and element dependent q value for each neighboring atom by its supercell index
                                tmp_qs = np.array([self.ani_specification["atom_qtable"][v][el][i] for v in _idx])
                                
                                # using the distances to calculate weights for the q values
                                tmp_rs = np.around(tmp_r[el][tmp_idx_ani[el]],decimals=6)
                                Z = round(np.sum(tmp_rs),6)
                                w = 1. - tmp_rs/Z
                                w /= w.sum()
                                ani_data[el][i] = np.dot(tmp_qs,w)      
                            else:
                                # only one neighbor, so no weighting
                                ani_data[el][i] = self.ani_specification["atom_qtable"][_idx[0]][el][i]

                    if np.isnan(ani_data[el]).any():
                        print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                        print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                        print("ani_data[el] {}".format(ani_data[el])) 
                        raise ValueError("Error! Found NaN value!")                    

                elif self.ani_type == "BOP4atoms-ref":
                    Nl = len(self.ani_specification["l_range"])
                    ani_data = {el:np.zeros(Nl,dtype=float) for el in self.ani_selected_keys}
                    
                    for el in tmp_r:
                        if len(tmp_idx_ani[el])==0: # if no elements of that type are neighboring skip the following
                            continue
                        
                        # selecting the original supercell indices from the neighboring knowledge in the ultracell                            
                        _idx = [self.ani_specification["map_idx_ultra2super"][v] for v in tmp_idx[el][tmp_idx_ani[el]]]
                        
                        theta, phi = get_angles(tmp_r_vec[el][tmp_idx_ani[el]])
                        for i,l in enumerate(self.ani_specification["l_range"]):
                            q_ref = get_q_linr(theta,phi,l,tmp_r[el][tmp_idx_ani[el]])
                            if len(_idx) > 1:            
                                # retrieving the "l" and element dependent q value for each neighboring atom by its supercell index
                                tmp_qs = np.array([self.ani_specification["atom_qtable"][v][el][i] for v in _idx])
                                
                                # using the distances to calculate weights for the q values
                                tmp_rs = np.around(tmp_r[el][tmp_idx_ani[el]],decimals=6)
                                Z = round(np.sum(tmp_rs),6)
                                w = 1. - tmp_rs/Z
                                w /= w.sum()
                                ani_data[el][i] = np.dot(tmp_qs,w) - q_ref
                            else:
                                # only one neighbor, so no weighting
                                ani_data[el][i] = self.ani_specification["atom_qtable"][_idx[0]][el][i] - q_ref
                            """
                            tmp_qs = np.array([self.ani_specification["atom_qtable"][v][el][i] for v in tmp_idx[el]])
                            tmp_rs = tmp_r[el]
                            Z = np.sum(tmp_rs)
                            w = 1 - tmp_rs/Z
                            w /= w.sum()
                            q_ref = get_q_linr(theta,phi,l,tmp_r[el][tmp_idx_ani[el]])
                            #print("tmp_qs {} tmp_rs {} w {}".format(tmp_qs[:5],tmp_rs[:5],w[:5]))
                            #print("q_ref {}".format(q_ref))
                            ani_data[el][i] = np.dot(tmp_qs,w)-q_ref   
                            """
                    if np.isnan(ani_data[el]).any():
                        print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                        print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                        print("ani_data[el] {}".format(ani_data[el])) 
                        raise ValueError("Error! Found NaN value!")       
                        

                elif self.ani_type == "BOP4atoms-ref2":
                    elements = [v for v in tmp_r.keys() if v!="all"]
                    assert len(elements)>0, "The list of elements to loop over is empty! Please change your ani_specification['usage']!"
                    
                    ani_data = {}

                    for j,el in enumerate(elements): 
                        theta, phi = get_angles(tmp_r_vec[el][tmp_idx_ani[el]])
                        ani_data[el] = np.zeros((len(tmp_r[el]),len(self.ani_specification["l_range"]),2)) #(Nl,Nneigh,2)
                        
                        for i,l in enumerate(self.ani_specification["l_range"]):
                            
                            tmp_qs = np.array([self.ani_specification["atom_qtable"][v][el][i] for v in tmp_idx[el]])
                            tmp_rs = tmp_r[el]
                            Z = np.sum(tmp_rs)
                            w = 1 - tmp_rs/Z
                            w /= w.sum()
                            q_ref = get_q_linr(theta,phi,l,tmp_r[el][tmp_idx_ani[el]])
                            
                            ani_data[el][:,i,0] = tmp_r[el]
                            ani_data[el][:,i,1] = tmp_qs-q_ref   
                        
                        if j == 0:
                            ani_data["all"] = ani_data[el]
                        else:
                            ani_data["all"] = np.concatenate((ani_data["all"],ani_data[el]),axis=0)
                        if np.isnan(ani_data[el]).any():
                            print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                            print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                            print("ani_data[el] {}".format(ani_data[el])) 
                            raise ValueError("Error! Found NaN value!")  
                    
                    to_pop = []
                    for el in ani_data:
                        if ani_data[el].shape[0] == 0: # empty entry
                            to_pop.append(el)
                    for _pair in to_pop:
                        ani_data.pop(_pair,None)           
                    
                elif self.ani_type == "BOP": # kind of MEAM-like using additional terms but as a function of the Bond Order Parameters (q_l)                    
                    Nl = len(self.ani_specification["l_range"])
                    ani_data = {el:np.zeros(Nl,dtype=float) for el in self.ani_selected_keys}
                    
                    for el in tmp_r:
                        if len(tmp_idx_ani[el])==0: # if no elements of that type are neighboring skip the following
                            continue
                        theta, phi = get_angles(tmp_r_vec[el][tmp_idx_ani[el]])
                        
                        for i,l in enumerate(self.ani_specification["l_range"]):
                            q = get_q(theta,phi,l)
                            ani_data[el][i] = q
                        
                        if np.isnan(ani_data[el]).any():
                            print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                            print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                            print("ani_data[el] {}".format(ani_data[el])) 
                            raise ValueError("Error! Found NaN value!")
                     
                elif self.ani_type == "BOP-r": # kind of MEAM-like using additional terms but as a function of the Bond Order Parameters weighted by inverse distances(q_l)
                    #print("processing BOP")
                    
                    Nl = len(self.ani_specification["l_range"])
                    ani_data = {el:np.zeros(Nl,dtype=float) for el in self.ani_selected_keys}
                    
                    for el in tmp_r:
                        
                        if len(tmp_idx_ani[el])==0: # if no elements of that type are neighboring skip the following
                            continue
                        theta, phi = get_angles(tmp_r_vec[el][tmp_idx_ani[el]])
                        
                        for i,l in enumerate(self.ani_specification["l_range"]):
                            ani_data[el][i] = get_q_linr(theta,phi,l,tmp_r[el][tmp_idx_ani[el]])

                        if np.isnan(ani_data[el]).any():
                            print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                            print("r<r_ani {}".format(tmp_r[el][tmp_idx_ani[el]]))
                            print("r {}".format(tmp_r[el]))
                            print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                            print("ani_data[el] {}".format(ani_data[el])) 
                            raise ValueError("Error! Found NaN value!")
                    
                elif self.ani_type == "BOP-invr": # kind of MEAM-like using additional terms but as a function of the Bond Order Parameters weighted by inverse distances(q_l)
                    #print("processing BOP")
                    
                    Nl = len(self.ani_specification["l_range"])
                    ani_data = {el:np.zeros(Nl,dtype=float) for el in self.ani_selected_keys}
                    
                    for el in tmp_r:
                        if len(tmp_idx_ani[el])==0: # if no elements of that type are neighboring skip the following
                            continue
                        theta, phi = get_angles(tmp_r_vec[el][tmp_idx_ani[el]])
                        
                        for i,l in enumerate(self.ani_specification["l_range"]):
                            ani_data[el][i] = get_q_r(theta,phi,l,tmp_r[el][tmp_idx_ani[el]])
                        
                        if np.isnan(ani_data[el]).any():
                            print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                            print("r<r_ani {}".format(tmp_r[el][tmp_idx_ani[el]]))
                            print("r {}".format(tmp_r[el]))
                            print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                            print("ani_data[el] {}".format(ani_data[el])) 
                            raise ValueError("Error! Found NaN value!")

                elif self.ani_type == "BOP-taper1":

                    Nl = len(self.ani_specification["l_range"])
                    ani_data = {el:np.zeros(Nl,dtype=float) for el in self.ani_selected_keys}

                    for el in tmp_r:
                        if len(tmp_idx_ani[el])==0: # if no elements of that type are neighboring skip the following
                            continue
                        theta, phi = get_angles(tmp_r_vec[el][tmp_idx_ani[el]])

                        for i,l in enumerate(self.ani_specification["l_range"]):
                            q_r = get_q_taper1(theta,phi,l,tmp_r[el][tmp_idx_ani[el]],self.ani_specification["r_ani"])
                            ani_data[el][i] = q_r  
                         
                        if np.isnan(ani_data[el]).any():
                            print("r_vec {}".format(tmp_r_vec[el][tmp_idx_ani[el]]))
                            print("\nel {}:theta {}\nphi {}".format(el,theta,phi))
                            print("ani_data[el] {}".format(ani_data[el])) 
                            raise ValueError("Error! Found NaN value!")
                                                                      
        self.x["r"] = r
        self.x["r_vec"] = r_vec
        self.x["ani"] = ani_data
        
        self.r_idx = r_idx
        self.ani_idx = ani_idx
        
        for val in self.idx_attrs_to_check:
            assert getattr(self,val) is not None, "Assertion failed - after processing the bond information {} still was not set!".format(val)
        for obs_type in self.observables:
            for val in self.observables[obs_type]:
                assert getattr(self,obs_type)[val] is not None, "Assertion failed - after processing self.{}['{}'] was not set!".format(obs_type,val)
    
    def __getitem__(self,key):
        return getattr(self,key)
    
    def get_x(self):
        """
        Returns
        -------
        observations : dict of np.ndarrays of float
            {"r":{"all":[...],"Al":[...],"phi":{"all":[...],...}}
        indices : dict of dict of np.ndarrays of int
            {"r":{"all":[...],"Al":[...],...},"phi":{"all":[...],"Al":[...],"Ni":...}}
        """
        return self.x, {"r":self.r_idx,"phi":self.phi_idx}
    
    def get_t(self):
        """
        Returns
        -------
        targets : t
            dict of properties as specified during initialization (see observations parameter of __init__)
            {"density":[4.],"E":[42.],...}
        """
        return self.t
    
def get_ultracell_old(fpos,cell,species,ultra_num=None,r_cut=6,search_pos=None):
    #structure = Atoms(species,positions=np.dot(fpos,cell),pbc=[1,1,1],cell=cell)
    
    if ultra_num is not None:
        i_range = np.arange(-ultra_num,ultra_num+1)
        ijks = list(itertools.product(i_range,i_range,i_range))
        idx_atoms = np.arange(len(fpos))
        
        e0, e1, e2 = np.array([1,0,0],dtype=float), np.array([0,1,0],dtype=float), np.array([0,0,1],dtype=float)
        for h,(i,j,k) in enumerate(ijks):
            tmp_fpos = np.array(fpos) + i * e0 + j*e1 + k*e2
            tmp_species = np.array(species)

            if h == 0:
                print("also store idx info...")
                ultracell_fpos = tmp_fpos
                ultracell_species = tmp_species
                ultracell_idx = idx_atoms
            else:
                ultracell_fpos = np.vstack((ultracell_fpos,tmp_fpos))
                ultracell_species = np.hstack((ultracell_species,tmp_species))
                ultracell_idx = np.hstack((ultracell_idx,idx_atoms))
        
    elif ultra_num is None and isinstance(r_cut,(int,float)):
        r_cut = float(abs(r_cut))
        # calculate i_ranges for different axes given the cutoff
        if search_pos is None:
            search_pos = np.array([[0,0,0],
                                   [1,1,1],
                                   [0,0,1],
                                   [1,1,0],
                                   [1,0,1],
                                   [0,1,0],
                                   [0,1,1],
                                   [1,0,0]],dtype=float)
        factors = [None for v in range(3)]
        ranges = [None for v in range(3)]
        tmp_species = np.array(species)
        
        #delta_comp = np.amin(np.diag(cell))
        tmp = np.absolute(cell)
        tmp = tmp[tmp>0]
        delta_comp = np.amin(tmp)
        #delta_comp_max = np.amax(np.diag(cell))
        delta_comp_max = np.amax(cell)
        basis = np.eye(3)
        delta_int = int((delta_comp+r_cut)/delta_comp+.5) # THIS NEEDS MORE THOUGHT!

        ijk = np.arange(-delta_int-1,delta_int+2)
        ijks = np.array(list(itertools.product(ijk,ijk,ijk)),dtype=int)

        idx_atoms = np.arange(len(fpos))
        
        for h,(i,j,k) in enumerate(ijks):
            if h == 0:
                grid_fpos = fpos + i*basis[0,:] + j*basis[1,:] + k*basis[2,:]
                ultracell_species = tmp_species
                ultracell_idx = idx_atoms
            else:
                grid_fpos = np.vstack((grid_fpos,fpos + i*basis[0,:] + j*basis[1,:] + k*basis[2,:]))
                ultracell_species = np.hstack((ultracell_species,tmp_species))
                ultracell_idx = np.hstack((ultracell_idx,idx_atoms))
        
        grid_pos = np.dot(grid_fpos,cell)
        
        skd = spatial.KDTree(grid_pos)
        idx = skd.query_ball_point(np.dot(search_pos,cell),2*r_cut)
        all_idx = np.array(list(set([v for v2 in idx for v in v2])),dtype=int)
        
        ultracell_fpos = grid_fpos[all_idx]  
        ultracell_species = ultracell_species[all_idx]
        ultracell_idx = ultracell_idx[all_idx] 
    else:
        raise ValueError("In order for dynamic generation of the ultracell when ultra_num == None r_cut is required to be int or float! Got r_cut = {} ...".format(r_cut))     

    return np.dot(ultracell_fpos,cell), ultracell_species, ultracell_idx
    
def get_ultracell(fpos,cell,species,r_cut,show=False,verbose=False,max_iter=20):
    if verbose:
        print("Generating ultracell with r_cut = {:.2f}".format(r_cut))
    Vcell = np.absolute(np.linalg.det(cell))
    
    # find center and corners of the cell
    center = .5 * cell.sum(axis=0)
    
    fcorners = np.array([[0,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,1,0],
                         [1,0,1],
                         [0,1,1],
                         [1,1,1]])
    corners = fcorners.dot(cell)
    
    # plot to make sure
    if show:
        fig = plt.figure()
        ax = Axes3D(fig)
        xco, yco, zco = corners.T
        ax.plot(xco,yco,zco,'dw',alpha=0.4,markeredgecolor="blue",label="corners")
        ax.plot([center[0]],[center[1]],[center[2]],"ro")
        plt.show()
    
    r_center2corner = np.linalg.norm(corners - center,axis=1).max()
    if verbose:
        print ('rcut = {} rcenter21corner = {}'.format(r_cut,r_center2corner))
    r_search = (r_cut + r_center2corner) * 1.5
    
    Vsphere = 4./3.*np.pi*r_search**3
    if verbose:
        print("approx. num of required cells = {}".format(Vsphere/float(Vcell)))
    
    start = list(itertools.product(*[[-1,0,1] for v in range(3)]))
    ijks_accepted = set(start) # contains all ijks ever accepted
    ijks_to_test = set(start) # contains all ijks which should be tested
    ijks_saturated = set() # contains all ijks which have max number of neighbors
    
    allowed_moves = [v for v in itertools.product(*[[-1,0,1] for v in range(3)]) if not (v[0]==0 and v[1]==0 and v[2]==0)]
    if verbose: 
        print("allowed moves {}".format(allowed_moves))
    
    i = 0
    while i<max_iter:
        if verbose:
            print("\n{}/{}".format(i+1,max_iter))
            print("cells: current = {} estimate for final = {}".format(len(ijks_accepted),Vsphere/float(Vcell)))
        
        # generate possible ijks by going through ijks_to_test comparing to ijks_saturated
        ijks_possible = [(i0+m0,i1+m1,i2+m2) for (i0,i1,i2) in ijks_to_test \
            for (m0,m1,m2) in allowed_moves if (i0+m0,i1+m1,i2+m2) not in ijks_saturated]
        if verbose: 
            print("possible new cells: {}".format(len(ijks_possible)))
        
        # check which ijks are within the specified search radius and add those to ijks_accpeted
        ijks_possible = [(i0,i1,i2) for (i0,i1,i2) in ijks_possible if np.linalg.norm(i0*cell[0,:]+i1*cell[1,:]+i2*cell[2,:])<=r_search]
        if verbose: print("cells after r filter {}".format(len(ijks_possible)))
        if len(ijks_possible) == 0:
            print("Found all cells for r_cut {} => r_search = {:.2f} Ang, terminating after {} iterations".format(r_cut,r_search,i+1))
            break

        # add all ijks_possible points to ijks_accepted
        ijks_accepted.update(ijks_possible)
        if verbose:
            print("accepted new cells: {}".format(len(ijks_accepted)))
        
        # all ijks_to_test points now are saturated, hence add to ijks_saturated
        ijks_saturated.update(ijks_to_test)
        if verbose:
            print("stored cells so far: {}".format(len(ijks_saturated)))
        
        # remove all previously tested points
        ijks_to_test.clear()
        
        # add all points which were not already known to ijks_to_test
        ijks_to_test.update(ijks_possible)
        if verbose:
            print("cell to test next round: {}".format(len(ijks_to_test)))
        
        i += 1
    if i == max_iter:
        warnings.warn("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
        raise Error("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
    
    # calculating the fractional atom positions
    fbasis = np.eye(3)
    idx_atoms = np.arange(len(fpos))
    
    for h,(i,j,k) in enumerate(ijks_accepted):
        new_fpos = fpos + i*fbasis[0,:] + j*fbasis[1,:] + k*fbasis[2,:]
        if h == 0:
            ultra_fpos = new_fpos
            ultra_species = np.array(species)
            ultracell_idx = idx_atoms
        else:
            ultra_fpos = np.vstack((ultra_fpos,new_fpos))
            ultra_species = np.hstack((ultra_species,species))
            ultracell_idx = np.hstack((ultracell_idx,idx_atoms))
                
    # converting atom positions into umm... non-fractional ...
    ultra_pos = np.dot(ultra_fpos,cell)
    
    return ultra_pos, ultra_species, ultracell_idx

def get_qs_for_atoms(ani_specification,skd,pos,ultracell,ultracell_species,ultracell_idx,
                    r_cut=None,num_neigh=None,decimals=6):
    """Generates a dict whose keys are the atom neighbor indices the ghost atoms will receive
    during their neighborsearch with values of np.ndarrays of BOPs.
    
    Example: # q_range for a specific atom number 5 in the original supercell selecting "all" elements

    l_range = [3,4,5]
    super_idx = 5
    el = "all"    
    q_range = [ultra_qs_atom_dict[super_idx][el][v] for v in range(len(l_range))] # q_range and l_range in same order
    """
    
    ultra_qs_atom_dict = {}

    # search for neighbors to pos
    if r_cut is not None:
        idx = skd.query_ball_point(pos,r_cut)
        
    elif num_neigh is not None:
        r,idx = skd.query(pos,k=num_neigh)
        
    qs_atom_dict = {} # will be returned after being filled
    elements = sorted(list(set(ultracell_species)))
    tmp_q = np.zeros(len(ani_specification["l_range"]))
    
    # looping over atoms and their neighbors
    for i,r_idx in enumerate(idx):
        
        qs_atom_dict[i] = {}
        
        # neighboring info
        r_idx = np.array(r_idx)
        
        # distance vectors
        r_vec = np.array(ultracell[r_idx]) - pos[i]
        r = np.linalg.norm(r_vec,axis=1)

        # finding non zero distances and limiting data to those
        tmp_idx = np.where(np.logical_not(np.isclose(r,np.zeros(r.shape))))

        r_idx = r_idx[tmp_idx]
        r_vec = r_vec[tmp_idx]
        r = r[tmp_idx]

        r_vec /= np.reshape(r,(-1,1))
        
        # generation of element groupings
        ele_idx = {el: np.where(ultracell_species[r_idx]==el)[0] for el in elements}
        ele_idx["all"] = np.arange(len(ultracell_species[r_idx]))
        for el in ele_idx:
            tmp_q[:] = 0
            
            if len(ele_idx[el])>0: # in case more than one neighbor with that element is found
                _r = r[ele_idx[el]]
                _r_vec = r_vec[ele_idx[el]]

                theta, phi = get_angles(_r_vec)
                
                for j,l in enumerate(ani_specification["l_range"]):
                    tmp_q[j] = get_q(theta,phi,l)
            
            # storing q data
            qs_atom_dict[i][el] = np.around(tmp_q,decimals=decimals)
   
    # using supercell indices as keys instead of the previous
    ultra_qs_atom_dict = {iv: qs_atom_dict[v] for iv,v in enumerate(ultracell_idx)}
    
    return ultra_qs_atom_dict

def get_observations(s, r_cut=None, num_neigh=6, selection=("random",10.),\
                     iso=True, aniso=False, ultra_num=2, verbose=False,\
                     seed=False, ani_type="BOP",\
                     ani_specification={"l_range":np.arange(0,7)}, usefortran=False):
    
    """Produces target densities and related input X, such as distances and angles.
    
    Parameters
    ----------
    
    s : instance of supercell

    r_cut : None or float
    
    num_neigh : None or int
    
    selection : tuple
        specifies how the selection of electron density points is done. Currently implemented 
        implemented are the following: 
        * uniformly at "random" selection: 
            ("random",10.) = 10% of the available points in a DFT structure
            ("random",10000,"absolute") = 10000 points out of the available points
        * "importance_mag" - biasing the probability to larger density values:
            ("importance_mag",10.) = as above
            ("importance_mag",10000,"absolute") = as above
        * "atom" centric selection of grid points
            ("atom",1.,"r") = selection of all grid points within 1 Angstrom of each atom
            ("atom",42,"N") = selection of the 42 nearest grid points to each atom 
    
    iso : boolean
    
    aniso : boolean
    
    ultra_num : int
        number of additional supercells in each dimension (plus and minus along all cell vectors)
    
    verbose : boolean
    
    seed : None or int

    ani_type : str, optional, default "BOP"

    ani_specification : dict, optional, default {"l_range":np.arange(0,7)}
    
    usefortran : boolean, optional, default False
    
    Returns
    -------

    bonds : list of local_bond_info instances
        Each local_bond_info instance represents a single electron density point
        and its neighbourhood information.

    Notes
    -----
    Note if both r_cut and num_neigh are provided num_neigh is selected. Also, the search is done
    via KD-trees which return indices for the nearest positions, which may include the present atom
    the neighbors are being searched for.
    """
    if verbose: print("Generating observations from DFT configurations and electron density...")
    assert r_cut is not None or num_neigh is not None, "Assertion failed - r_cut and num_neigh are both None please provide valid input."
    if r_cut is not None: assert isinstance(r_cut,(float,int)), "Assertion failed - r_cut is neither int nor float..."
    if num_neigh is not None: assert isinstance(num_neigh,int), "Assertion failed - num_neigh has to be of int type..."
    
    if r_cut is not None and num_neigh is not None: r_cut = None
    if seed is not False:
        np.random.seed(seed=seed)
    points,values = s["edensity"]["xyz"], s["edensity"]["density"] # points is in terms of real space
            
    N = len(points)
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
        rho_min, rho_max = np.amin(values), np.amax(values)
        drho = rho_max - rho_min
        prob = np.array([(v-rho_min)/(drho) for v in values])
        prob /= prob.sum()
        
        if Nsamples < N:
            idx = np.random.choice(np.arange(N),size=Nsamples,p=prob,replace=False)
        else:
            idx = np.arange(N)    
    elif selection[0] == "atom":
        skd_edensity = spatial.KDTree(points)
        if selection[2] == "r":
            idx = skd_edensity.query_ball_point(np.dot(s["positions"],s["cell"]),selection[1])
        elif selection[2] == "N":
            _, idx = skd_edensity.query(np.dot(s["positions"],s["cell"]),k=selection[1])
        elif selection[2] == "nearest":
            rvals,idx = skd_edensity.query(np.dot(s["positions"],s["cell"]),k=30)
            newidx = [None]*len(idx)
            for i,atomlist in enumerate(rvals):
                # min distance to grid point from atom
                mindist = np.argmin(atomlist)

                for j,_r in enumerate(atomlist):
                    if abs(mindist-_r)<1e-8:
                        newidx[i].append(idx[i][j])
            # list of grid idxs for each atom
            idx = newidx
        else:
            raise NotImplementedError
        idx = [list(v) for v in idx]
        idx = np.array(list(set([v for v2 in idx for v in v2])),dtype=int)
        
    elif selection[0] == "all" or selection == "all":
        idx = np.arange(N)
    else:
        raise NotImplementedError("selection type {} not implemented!".format(selection))
    
    eval_points = points[idx]
    N_eval = len(eval_points)
    t = values[idx]

    if verbose: 
        print("Generating {} ghost atoms...".format(len(t)))

    # generate ultra cell
    t0 = time.time()
    #ultracell, ultracell_species, ultracell_idx = get_ultracell_old(s["positions"],s["cell"],s["species"],ultra_num=ultra_num,r_cut=r_cut,search_pos=None)
    ultracell, ultracell_species, ultracell_idx = get_ultracell(s["positions"],s["cell"],s["species"],r_cut=r_cut,verbose=verbose)
    if verbose:
        print("generating ultracell {} s...".format(time.time()-t0))

    if r_cut is not None:
        if False:#usefortran:
            print ('doing fortran neighbour search')
            idx = f90.query_ball_point(ultracell,eval_points,r_cut)
        else:
            skd = spatial.cKDTree(ultracell)
            idx = skd.query_ball_point(eval_points,r_cut)
            for v,_ix in enumerate(idx):
                r = np.linalg.norm(ultracell[np.array(_ix,dtype=int)]-eval_points[v],axis=1)
                idx[v] = [_ix[v] for v in np.argsort(r)]
        
    elif num_neigh is not None:
        skd = spatial.cKDTree(ultracell)
        r,idx = skd.query(eval_points,k=num_neigh)
        
    bonds = []
    t_bond_prep, t_bond_init, t_bond_info = 0,0,0
    
    t0 = time.time()   
    if aniso:
        if "BOP4atoms" in ani_type:
            # atom_qtable contain the q value for each supercell atom in a dictionary
            atom_qtable = get_qs_for_atoms(ani_specification,skd,np.dot(s["positions"],s["cell"]),ultracell,ultracell_species,\
                                        ultracell_idx,r_cut=r_cut,num_neigh=num_neigh)
            ani_specification["atom_qtable"] = atom_qtable
            
            # this map is crucial to relate the ultracell indices in "idx" from the neighborhood search to supercell indices for the atom_qtable
            ani_specification["map_idx_ultra2super"] = {v:v2 for v,v2 in enumerate(ultracell_idx)}
        t_bond_prep += time.time()-t0

    if verbose:
        print('generating bond instances')
    for i in range(len(eval_points)):
        t0 = time.time()
        bond = local_bond_info(aniso=aniso,ani_type=ani_type,ani_specification=ani_specification)
        neigh_idx = idx[i]
        target = {"density":np.array([t[i]],dtype=float)}
        t_bond_init += time.time()-t0
        
        t0 = time.time()
        try:
            bond.set_info(eval_points[i],neigh_idx,ultracell,ultracell_species,s["cell"],target,\
                      usefortran=usefortran)
            if any([len(list(bond.x["r"].keys()))>0, len(list(bond.x["ani"].keys()))>0]):
                bonds.append(bond)
        except ValueError:
            print("Found zero neighbors within ani cutoff.")
        t_bond_info += time.time()-t0
    if verbose:
        print("prep {} s...\ninit {} s...\ninfo {} s...".format(t_bond_prep,t_bond_init,t_bond_info))
     
    return bonds

def show_rvm_performance(niter,logbook,log=True):
    fig = plt.figure()
    x_iter = np.arange(len(logbook["L"]))
    ax = fig.add_subplot(131)
    #x_iter = list(range(niter))
    ax.plot(x_iter,logbook["L"],'-',linewidth=2,alpha=0.8)
    ax.set_xlabel("#iteration")
    ax.set_ylabel("Log likelihood")
    ax.grid()
    #ax.margins(1)

    ax2 = fig.add_subplot(132)
    ax2.plot(x_iter,logbook["beta"],'-')
    ax2.set_xlabel('#iteration')
    if log: ax2.set_yscale("log")
    ax2.set_ylabel("beta")
    ax2.grid()
    #ax2.margins(1)

    ax3 = fig.add_subplot(133)
    ax3.plot(x_iter,logbook["mse"],'-') 
    ax3.set_xlabel('#iteration')
    if log: ax3.set_yscale("log")
    ax3.set_ylabel("mse")
    ax3.grid()
    #ax3.margins(1)

    plt.suptitle("RVM approach")
    plt.tight_layout()
    plt.show()

def show_densities_near_atoms(s,xyz,pred_density,dim=2,tol=1e-3,title=""):
    
    for pos in s["positions"]:
        fig = plt.figure()
        ax = Axes3D(fig)
        idx = np.array([v for v in range(len(xyz)) if abs(xyz[v][2]-pos[2])<tol],dtype=int)
        x,y,_ = xyz[idx].T
        z = pred_density[idx]
        
        ax.plot([pos[0]],[pos[1]],[pos[2]],'ro',markersize=10)
        ax.plot(x,y,z,'d')
        z = s["edensity"]["density"][idx]
        ax.plot(x,y,z,'+')
        plt.title(title)
        
    plt.show()

def save_regressed_rho(r, rhos, save_rhos_path, lb=0, ub=None, dft_path=None,\
                       logbook=None, i=None, niter=None, tol=None,\
                       fix_beta=None, beta_init=None, sequential=None, k_iso=None,\
                       k_ani=None, type_iso=None, type_ani=None, ultra_num=None,\
                       selection=None, num_neigh=None, r_cut=None, aniso=None,\
                       smooth=None, r_smooth=None, f_smooth=None, q=None, 
                       rhos_ani_q=None, Nsteps_ani=None, ani_type=None,\
                       ani_specification=None, alpha_init=None, weights=None,
                       seed=None,fit_idx=None,training_set=None):
    """Stores all relevant information on density regressions on disc.

    Parameters
    ----------

    r, rhos : obtained by fitelectrondensity.predict.predict_rho_iso

    save_rhos_path : str
        Path to save the information at.
    
    lb : float, optional, default 0
    
    ub : float, optional, default None
    
    dft_path : str, optional, default None

    logbook : dict, optional, default None
    
    i : int, optional, default None
    
    niter : int, optional, default None
    
    tol : float, optional, default None

    fix_beta : boolean, optional, default None
    
    beta_init : float, optional, default None
    
    sequential : boolean, optional, default None
    
    k_iso : int, optional, default None

    k_ani : int, optional, default None
    
    type_iso : boolean, optional, default None
    
    type_ani : boolean, optional, default None
    
    ultra_num : int, optional, default None
    
    selection : list of tuples, optional, default None
    
    num_neigh : int, optional, default None
    
    r_cut : float, optional, default None
    
    aniso : boolean, optional, default None
    
    smooth : boolean, optional, default None
    
    r_smooth : float, optional, default None
    
    f_smooth : float, optional, default None
    
    q : float, optional, default None
    
    rhos_ani_q : dict, optional, default None
    
    Nsteps_ani : int, optional, default None
    
    ani_type : str, optional, default None

    ani_specification : dict, optional, default None
    
    alpha_init : float np.ndarray, optional, default None
    
    weights : float np.ndarray, optional, default None
    
    seed : int, optional, default None
    
    fit_idx : int, optional, default None
    
    training_set : dict, optional, default None
    """
    
    print("Saving the regression data to {}...".format(save_rhos_path))
    rho_dict = {"r":r,
                "rhos":rhos,
                "lb":lb,
                "ub":ub,
                "dft_path":dft_path,
                "logbook":logbook,
                "i":i,
                "niter":niter,
                "tol":tol,
                "fix_beta":fix_beta,
                "beta_init":beta_init,
                "alpha_init":alpha_init,
                "weights":weights,
                "sequential":sequential,
                "k_iso":k_iso,
                "k_ani":k_ani,
                "type_iso":type_iso,
                "type_ani":type_ani,
                "ultra_num":ultra_num,
                "selection":selection,
                "num_neigh":num_neigh,
                "r_cut":r_cut,
                "aniso":aniso,
                "smooth":smooth,
                "r_smooth":r_smooth,
                "f_smooth":f_smooth,
                "q":q,
                "rhos_ani_q":rhos_ani_q,
                "Nsteps_ani":Nsteps_ani,
                "ani_type":ani_type,
                "ani_specification":ani_specification,
                "seed":seed,
                "fit_idx":fit_idx,
                "training_set":training_set}
    with open(save_rhos_path,"wb") as f:
        pickle.dump(rho_dict,f)

def save_Phi(save_path_Phi,Phi):
    print("Saving the design matrix Phi to {}...".format(save_path_Phi))
    with open(save_path_Phi,"wb") as f:
        pickle.dump(Phi,f)
        
def save_predicted_density(save_path_predicted,xyz,pred):
    print("Saving the predicted density and coordinates to {}...".format(save_path_predicted))
    with open(save_path_predicted,"wb") as f:
        pickle.dump({"density":pred,"xyz":xyz},f)

def load_Phi(path_Phi):
    print("Loading the design matrix Phi from {}...".format(path_Phi))
    with open(path_Phi,"rb") as f:
        Phi = pickle.load(f)
    return Phi

def load_regressed_rho(path_rhos,operations=[],conv=None,params=None,
        return_bounds=False,show=False):
    """Loads a file containing information about previously regressed rho(r) functions.

    Parameters
    ----------
    path_rhos : str, list of str or tuple of str
        path(s) to ".rhos" file(s)
    operations : list of str
        can contain "normalize", "absolute", "shift". does the respective operations to
        the rho(r) functions in the order they are put in the list.
    return_bounds : boolean
        return min and max density value observed after applying the operations
    """
    print("Loading regression data from {}...".format(path_rhos))

    if isinstance(path_rhos,str):
        with open(path_rhos,"rb") as f:
            rho_dict = pickle.load(f)
    elif isinstance(path_rhos,(list,tuple)):
        for i,p in enumerate(path_rhos):
            with open(p,"rb") as f:
                if i==0:
                    rho_dict = pickle.load(f)
                else:
                    rho_dict.update(pickle.load(f))

    def _normalize(rho_dict):
        rho_min = min([min(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        rho_max = max([max(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        drho = rho_max - rho_min
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] /= drho
        return rho_dict

    def _absolute(rho_dict):
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] = np.absolute(rho_dict["rhos"][s])
        return rho_dict

    def _shift(rho_dict):
        rho_min = min([min(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] -= rho_min
        return rho_dict
            
    def _convolute(params,rho,fun_type="exp"):
        if fun_type == "exp":
            def fun(r):
                return (rho-params[0])*np.exp(-r)
        elif fun_type == "psi":
            def fun(r):
                x = (r-params[1])/float(params[2])
                x4 = x**4
                return (rho-params[0])* x4/(1.+x4)
        elif fun_type == "psi2":
            def fun(r):
                x = (r-params[1])/float(params[2])
                x4 = x**4
                x = (r-params[3])/float(params[4])
                xp4 = x**4
                return (rho-params[0])* x4/(1+x4) * xp4/(1+xp4)
        else:
            raise NotImplementedError
        return fun

    implemented_ops = {"normalize":_normalize,"absolute":_absolute,"shift":_shift}
    implemented_convs = {"exp","psi","psi2"}

    for op in operations:
        if op in implemented_ops:
            rho_dict = implemented_ops[op](rho_dict)
        else:
            raise NotImplementedError

    if conv is not None:
        assert conv in implemented_convs, "Assertion failed - conv '{}' is not one of the implemented convolutions: {}".format(conv,implemented_convs)
        
        lb = min([min(rho_dict["rhos"][v]) for v in rho_dict["rhos"]])
        for s in rho_dict["rhos"]:
            
            if conv == "exp":
                _params = [0]#[lb]
            elif conv == "psi":
                _params = [0]+list(params)#[lb]+list(params)
            else:
                raise NotImplementedError

            fun = _convolute(_params,rho_dict["rhos"][s],fun_type=conv)
            rho_dict["rhos"][s] = fun(rho_dict["r"])
            
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ks = sorted(rho_dict["rhos"].keys())
        for k in ks:
            ax.plot(rho_dict["r"],rho_dict["rhos"][k],label=k)
        ax.set_xlabel("r [AA]",fontsize=14)
        ax.set_ylabel("rho (r)",fontsize=14)
        ax.grid()
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    if return_bounds:
        rho_lb = min([min(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        rho_ub = max([max(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        return rho_dict, rho_lb, rho_ub
    else:
        return rho_dict

def load_predicted_density(path_predicted):
    print("Loading predicted density and coordinates from {}...".format(path_predicted))
    with open(path_predicted,"rb") as f:
        d = pickle.load(f)
    return d["xyz"], d["density"]

def shift_densities(bonds,verbose=False):
    """Searches for negative densities in bonds and shifts them accordingly.
    
    This function is used for the regression of EAM potentials, which
    can not have negative embedding densities.

    Parameters
    ----------
    bonds : list of supercell instances
    
    Returns
    -------
    bonds : list of supercell instances
        with modified electron densities
    min_dens : float or None
        smallest negative density value found or None if no densities were negative
    """
    
    neg_dens = np.array([bond.t["density"] for bond in bonds if bond.t["density"]<0.])
    if neg_dens.shape[0]>0:
        if verbose:
            print("found {} negative densities...: min density = {}".format(neg_dens.shape[0],neg_dens.amin()))
        min_dens = np.amin(neg_dens) - 1e-6

        Nbonds = len(bonds)
        for i in range(Nbonds):
            bonds[i].t["density"] = bonds[i].t["density"] - 1*min_dens
        for i in range(Nbonds):
            assert bonds[i].t["density"].shape == (1,)
            if bonds[i].t["density"][0] < 0:
                print("Bugger, still negative density values left...")
                print("density {}".format(bonds[i].t["density"]))
                raise
        if verbose:
            print("Modified density values by shifting using min_dens = {}...".format(min_dens))
        time.sleep(15)
        return bonds, min_dens
    return bonds, None