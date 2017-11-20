from __future__ import print_function

import ase
from ase.calculators.eam import EAM
from fitenergy.potential_serial import get_precomputed_energy_force_values
import warnings, itertools, pickle, os, sys, copy
sys.path.append("..")
import parsers
import fitelectrondensity as fed
import numpy as np
from scipy import spatial
import scipy as sp
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pylab as plt

from . import nonlin

# new stuff - parallel
from fitenergy import misc, potential, regression

# new stuff - serial
from fitenergy import misc_serial, potential_serial, regression_serial
from fitenergy.potential_serial import wrapped_smooth

# single atom vacuum energy ground state energies
my_globals = {"E_ref":{"Al":-107.344221-0.172105115032,
                       "Nb":-1647.496121,
                       "Ni":-1358.048079374}}

def DistanceCosTapering_basis(kappa,taper_fun):
    """Wrapper function for cosines.

    Parameters
    ----------

    kappa : float
    
    taper_fun : callable
        Tapering function.

    Returns
    -------

    cos_fun : callable
        Tapered cosine function.
    """
    def cos_fun(x):
        return np.cos(kappa*x)*taper_fun(x)
    return cos_fun

def DistanceCosTapering_basis_1stDerivative(kappa,taper_fun):
    """Wrapper function for first derivatives of cosines.

    Parameters
    ----------

    kappa : float
    
    taper_fun : callable
        Tapering function.

    Returns
    -------

    cos_fun : callable
        Tapered cosine function derivative.
    """
    def cos_fun(x):
        return -kappa*np.sin(kappa*x)*taper_fun(x) + np.cos(kappa*x)*sp.misc.derivative(taper_fun,x,dx=1,n=1)
    return cos_fun

def taper_fun_wrapper(_type="x4ge",**kwargs):
    if _type=="x4ge":
        def x4_fun(x):
            x4 = ((x-kwargs["a"])/float(kwargs["b"]))**4
            if isinstance(x,(int,float)):
                x4 = 0 if x>=kwargs["a"] else x4
            else:
                x4[x>=kwargs["a"]] = 0
            return x4/(1.+x4)
        return x4_fun
    
    elif _type=="x4le":
        def x4_fun(x):
            x4 = ((x-kwargs["a"])/float(kwargs["b"]))**4
            if isinstance(x,(int,float)):
                x4 = 0 if x<=kwargs["a"] else x4
            else:
                x4[x<=kwargs["a"]] = 0
            return x4/(1.+x4)
        return x4_fun
    
    elif _type=="Behlerge":
        def Behler_fun(x):
            y = .5 * (np.cos(np.pi*x/float(kwargs["a"])) + 1.)
            if isinstance(x,(int,float)):
                y = 0 if x>=kwargs["a"] else y
            else:
                y[x>=kwargs["a"]] = 0
            return y
        return Behler_fun
    
    elif _type=="Behlerle":
        def Behler_fun(x):
            y = .5 * (np.cos(np.pi*x/float(kwargs["a"])) + 1.)
            if isinstance(x,(int,float)):
                y = 0 if x<=kwargs["a"] else y
            else:
                y[x<=kwargs["a"]] = 0
            
            return y
        return Behler_fun
    
    elif _type=="Ones":
        def ones_fun(x):
            if isinstance(x,np.ndarray):
                return np.ones(x.shape)
            elif isinstance(x,(float,int)):
                return 1
            else:
                raise
        return ones_fun
    
    else:
        raise NotImplementedError("_type '{}' unknown.".format(_type))
        
def get_splined_energy_functions(mapper, weights, rho_lb, rho_ub, rho_dict, \
                                 basis_r, basis_rho, N_steps=int(1e4),\
                                 r_lb=0., r_ub=6., \
                                 basis_r_1stder=None, basis_rho_1stder=None, show=False, \
                                 figsize=(5,5), **kwargs):
    """Generates splined energy functions for a 1d weights array.
    
    This function can be used in conjunction with the generate_EAM_calculator function 
    to generate ase EAM calculator instances and setfl format files to be read 
    by LAMMPS for example.
    
    Parameters
    ----------
    
    mapper : dict
        Maps the elements (for embedding energy functions) and element pairs
        (for pair energy functions) to the values in 'weights'. 
        Example: {"emb":{"Al":np.array([0,1,2,3,4])}, "pair":{("Al","Al"):np.array([5,6,7,8,9])}}
    
    weights : float np.ndarray of shape (M,)
        M parameters for energy functions. 
        
    rho_lb : dict
        Lower bounds for embedding densities, e.g. {"Al":0, "Ni":0}
    
    rho_ub : dict
        Upper bounds for embedding densities, e.g. {"Al":2, "Ni":np.pi}
    
    rho_dict : dict
        Stored in *.pckl files generated setting up the EAM regression. 
        Contains the following keys: ['logbook', 'rhos', 'aniso', 'ani_specification', 
            'ub', 'lb', 'training_set', 'num_neigh', 'tol', 'smooth', 'beta_init', 
            'sequential', 'weights', 'k_ani', 'r_smooth', 'rhos_ani_q', 'fix_beta', 
            'ultra_num', 'niter', 'alpha_init', 'k_iso', 'q', 'i', 'ani_type', 'fit_idx', 
            'dft_path', 'seed', 'type_iso', 'type_ani', 'f_smooth', 'r', 'r_cut', 
            'Nsteps_ani', 'selection']) dict_keys(['mse', 'median_se', 'dev_mvs_95pct', 
            'weights_full', 'L', 'max', 'beta', 'dev_std', 'Sigma', 'tse', 'weights',
            'dev_est', 'min', 'alphas'])

    basis_r : list of callables
        Basis functions for the pair energy functions.

    basis_rho : list of callables
        Basis functions for the embedding energy functions.

    N_steps : int
        Number of steps to use for the generation of data points for splining.
        
    r_lb : float
        Lower pair distance bound.
    
    r_ub : float
        Upper pair distance bound.
        
    basis_r_1stder : list of callables, optional, default None
        Basis functions for the 1st pair energy function derivatives.
        When given the returned 'energy_funs' will contain corresponding
        'dpair' key and a dict of pair energy derivatives.

    basis_rho_1stder : list of callables, optional, default None
        Basis functions for the 1st embedding energy function derivatives.
        When given the returned 'energy_funs' will contain corresponding
        'demb' key and a dict of embedding energy derivatives.
    
    figsize : int tuple of length 2, optional, default (7,5)
        parameter for plt.figure()

    show : boolean, optional, default False
        To plot or not to plot.
    
    Returns
    -------

    energy_funs : dict
        Contains 
            * "rho": float np.ndarray of shape (Nrho,)
            * "r": float np.ndarray of shape (Nr,)
            * "dr": float
            * "drho": float
            * "emb": dict of float np.ndarrays of shapes (Nrho,), e.g. {"Al":np.ndarray(...), ...}
            * "pair": dict of float np.ndarrays of shapes (Nr,), e.g. {("Al","Al"):np.ndarray(...), ...}
            * "rhos": dict of embedding density function in form of np.ndarrays of shape (Nrho,) 
            * "drhos": dict of embedding density function in form of np.ndarrays of shape (Nrho,) 
            * "demb": dict of float np.ndarrays of shapes (Nrho,), e.g. {"Al":np.ndarray(...), ...}
            * "dpair": dict of float np.ndarrays of shapes (Nr,), e.g. {("Al","Al"):np.ndarray(...), ...}
    """
            
    energy_funs = {"rho":None,
                   "r":None,
                   "emb":dict(),
                   "pair":dict(),
                   "rhos":dict(),
                   "drhos":dict(),
                   "demb":dict(),
                   "dpair":dict(),
                   "dr":None,
                   "drho":None}
    
    # density bounds
    tmp_rho_lb, tmp_rho_ub = min(rho_lb.values()), max(rho_ub.values())
    
    # increments
    drho = np.round((tmp_rho_ub - tmp_rho_lb)/float(N_steps),decimals=6)
    dr = np.round((r_ub - r_lb)/float(N_steps),decimals=6)
    energy_funs["dr"] = dr
    energy_funs["drho"] = drho
    
    # rho
    rho = np.array([tmp_rho_lb+v*drho for v in range(N_steps)])
    energy_funs["rho"] = rho
    
    # r
    r = np.array([r_lb+v*dr for v in range(N_steps)])
    energy_funs["r"] = r
    
    # Phi_r
    Phi_r = np.array([b(r) for b in basis_r]).T
    if not basis_r_1stder is None:
        dPhi_r = np.array([b(r) for b in basis_r_1stder]).T
    
    # Phi_emb
    Phi_emb = np.array([b(rho) for b in basis_rho]).T
    if not basis_rho_1stder is None:
        dPhi_emb = np.array([b(rho) for b in basis_rho_1stder]).T
    
    # embedding energy and density functions
    for species in sorted(mapper["emb"]):
        energy_funs["emb"][species] = np.dot(Phi_emb,weights[mapper["emb"][species]])
        _f = spline(rho_dict["r"],rho_dict["rhos"][species],ext=0)
        energy_funs["rhos"][species] = _f(r)
        
        # derivatives
        energy_funs["drhos"][species] = sp.misc.derivative(_f,r,dx=1,n=1)
        if not basis_rho_1stder is None:
            energy_funs["demb"][species] = np.dot(dPhi_emb,weights[mapper["emb"][species]])
            
    # pair energy
    for pair in sorted(mapper["pair"]):
        energy_funs["pair"][pair] = np.dot(Phi_r,weights[mapper["pair"][pair]])    
        
        # derivatives
        if not basis_r_1stder is None:
            energy_funs["dpair"][pair] = np.dot(dPhi_r,weights[mapper["pair"][pair]])
            
    if show:
        # embedding densities
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(311)
        for species in sorted(energy_funs["rhos"]):
            p = ax.plot(energy_funs["r"], energy_funs["rhos"][species], '-',
                        label="f(r): {}".format(species))
            if species in energy_funs["drhos"]:
                ax.plot(energy_funs["r"], energy_funs["drhos"][species], '--',
                        label="f'(r): {}".format(species), color=p[0].get_color())
        ax.set_xlabel("r")
        ax.set_ylabel("embedding density")
        plt.legend(loc=0)
        
        # embedding energies
        ax = fig.add_subplot(312)
        for species in sorted(energy_funs["emb"]):
            p = ax.plot(energy_funs["rho"], energy_funs["emb"][species], '-',
                        label="f(rho): {}".format(species))
            if species in energy_funs["drhos"]:
                ax.plot(energy_funs["rho"], energy_funs["demb"][species], '--',
                        label="f'(rho): {}".format(species), color=p[0].get_color())
        ax.set_xlabel("rho")
        ax.set_ylabel("embedding energy")
        plt.legend(loc=0)
        
        # pair energies
        ax = fig.add_subplot(313)
        for pair in sorted(energy_funs["pair"]):
            p = ax.plot(energy_funs["r"], energy_funs["pair"][pair], '-', 
                        label="f(r): {}".format(pair))
            if pair in energy_funs["dpair"]:
                ax.plot(energy_funs["r"], energy_funs["dpair"][pair], '--',
                        label="f'(rho): {}".format(pair), color=p[0].get_color())
        ax.set_xlabel("r")
        ax.set_ylabel("pair energy")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    return energy_funs

def generate_EAM_calculator(energy_funs, r_cut, N_steps=10000, eam_path=None, show=False, return_calc=True,
                            atom_info={'Ni':{"mass":58.6934,"number":28,"lattice":"fcc","a0":3.52},
                                       'Al':{"mass":26.9815,"number":13,"lattice":"fcc","a0":4.02},
                                       'Ti':{"mass":47.88,"number":22,"lattice":"fcc","a0":2.95},
                                       'Nb':{"mass":92.90637,"number":41,"lattice":"bcc","a0":3.3}}):
    """Generates an EAM calculator instance.
    
    This function takes the regressed empirical potential in 'Es' creates an 
    ase EAM calculator instance and possibly writes it to disk in form of a 
    setfl file with .eam.alloy ending.
    
    Parameters
    ----------
    
    energy_funs : dict
        Obtained from get_splined_energy_functions.

    r_cut : float
        Cutoff radius.

    N_steps : int, optional, default 10000
        Number of places functions are evaluted for splines.
    
    eam_path : string, optional, default None
        Path to write the setfl file to.
    
    show: boolean, optional, default False
        Whether to show the EAM potential or not.
        
    return_calc : boolean, optional, default True
        Whether or not to return the generated EAM calculator instance.
        
    atom_info : dict, optional, default contains "Ni", "Al", "Ti" and "Nb".
    
    Returns
    -------
    
    EAM_obj : instance of an ase EAM calculator instance
        Is returned if 'return_calc' is True, otherwise this function returns 
        None.
    """
    
    save_as_setfl = not eam_path is None
    if save_as_setfl: 
        assert os.path.exists(os.path.dirname(eam_path)), "Assertion failed - can't write to specified path: {} ...!".format(eam_path)
        if os.path.exists(eam_path):
            warnings.warn("Path ({}) exists, overwriting ....".format(eam_path))
        if eam_path[-10:] != ".eam.alloy":
            warnings.warn("The specified file doesn't end on '.eam.alloy'! The file is modified accordingly...")
            eam_path += ".eam.alloy"
    
    elements = list(sorted(energy_funs["emb"].keys()))
    N_ele = len(elements)
    pair_map = {tuple([el1,el2]):tuple(sorted([el1,el2])) for el1,el2 in itertools.product(elements,elements)}
    
    dr = energy_funs["dr"]
    r = energy_funs["r"]
    drho = energy_funs["drho"]
    rho = energy_funs["rho"]
    
    Nrho = N_steps
    Nr = N_steps

    # functions
    embedded_energy = np.array([spline(rho, energy_funs["emb"][el]) for el in elements])
    electron_density = np.array([spline(r, energy_funs["rhos"][el]) for el in elements])
    phi = np.array([[spline(r, energy_funs["pair"][(pair_map[(el2,el1)])]) for el2 in elements] for el1 in elements])
    
    # derivatives of functions
    if len(energy_funs["demb"])>0:
        d_embedded_energy = np.array([spline(rho, energy_funs["demb"][el]) for el in elements])
    else:
        d_embedded_energy = np.array([v.derivative() for v in embedded_energy])
        
    if len(energy_funs["drhos"])>0:
        d_electron_density = np.array([spline(r, energy_funs["drhos"][el]) for el in elements])
    else:
        d_electron_density = np.array([v.derivative() for v in electron_density])
        
    if len(energy_funs["dpair"])>0:
        d_phi = np.array([[spline(r, energy_funs["dpair"][(pair_map[(el2,el1)])]) for el2 in elements] for el1 in elements])
    else:
        d_phi = np.array([[phi[v0,v1].derivative() for v1 in range(N_ele)] for v0 in range(N_ele)])

    EAM_obj = EAM(elements=elements, embedded_energy=embedded_energy,
                    electron_density=electron_density,
                    phi=phi, 
                    d_embedded_energy=d_embedded_energy,
                    d_electron_density=d_electron_density,
                    d_phi=d_phi,
                    cutoff=r_cut, form='alloy',
                    Z=[atom_info[el]["number"] for el in elements], nr=Nr, nrho=Nrho, dr=dr, drho=drho,
                    lattice=[atom_info[el]["lattice"] for el in elements], 
                    mass=[atom_info[el]["mass"] for el in elements], 
                    a=[atom_info[el]["a0"] for el in elements])
    setattr(EAM_obj,"Nelements",N_ele)
    
    if show:
        EAM_obj.plot()
        plt.show()
        
    if save_as_setfl:
        print("Writing setfl file to {}...".format(eam_path))
        EAM_obj.write_potential(eam_path)
        
    if return_calc:
        return EAM_obj
    else:
        return None

def _select(selection,sample_range):
        
    N = len(sample_range)
    if selection[0] == "all":
        idx = sample_range
    elif selection[0] == "uniform":
        if selection[2] == "absolute":
            if selection[1]>=N:
                idx = sample_range
            else:
                idx = np.random.choice(sample_range,size=int(selection[1]),replace=False)
        elif selection[2] == "relative":
            if selection[1]>=1.:
                idx = sample_range
            else:
                idx = np.random.choice(sample_range,size=int(selection[1]*N),replace=False)
    else:
        raise NotImplementedError  
     
    return idx

def select_supercells(supercells,selection):
    print("Selecting using {}".format(selection))
    N = len(supercells)
    
    if isinstance(selection,tuple):
        sample_range = np.arange(N)
        idx = _select(selection,sample_range)                
        
    elif isinstance(selection,list):
        for i,sel in enumerate(selection):
            tmp_sample_range = np.array([v_i for v_i,v in enumerate(supercells) if sel[0] in v.name],dtype=int)
            tmp_selection = tuple(sel[1:])
            tmp_idx = _select(tmp_selection,tmp_sample_range)
            if i == 0:
                idx = np.array(tmp_idx)
            else:
                idx = np.hstack((idx,tmp_idx))
    else:
        raise NotImplementedError
    idx = set(idx)
    idx = sorted(list(idx))
    
    return idx

def get_ultracell(fpos,cell,species,r_cut,show=False,verbose=False,max_iter=20):
    if verbose:
        print("Generating ultracell with r_cut = {}".format(r_cut))
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
        
    r_center2corner = np.linalg.norm(corners - center,axis=1).max()
    r_search = (r_cut + r_center2corner) * 1.5
    
    Vsphere = 4./3.*np.pi*r_search**3
    if verbose:
        print("approx. num of required cells = {}".format(Vsphere/float(Vcell)))
    
    start = list(itertools.product(*[[-1,0,1] for v in range(3)]))
    ijks_accepted = set(start) # contains all ijks ever accepted
    ijks_to_test = set(start) # contains all ijks which should be tested
    ijks_saturated = set() # contains all ijks which have max number of neighbors
    
    allowed_moves = [v for v in itertools.product(*[[-1,0,1] for v in range(3)]) if not (v[0]==0 and v[1]==0 and v[2]==0)]
    if verbose: print("allowed moves {}".format(allowed_moves))
    
    i = 0
    while i<max_iter:
        if verbose:
            print("\n{}/{}".format(i+1,max_iter))
            print("cells: current = {} estimate for final = {}".format(len(ijks_accepted),Vsphere/float(Vcell)))
        
        # generate possible ijks by going through ijks_to_test comparing to ijks_saturated
        ijks_possible = [(i0+m0,i1+m1,i2+m2) for (i0,i1,i2) in ijks_to_test \
            for (m0,m1,m2) in allowed_moves if (i0+m0,i1+m1,i2+m2) not in ijks_saturated]
        if verbose: print("possible new cells: {}".format(len(ijks_possible)))
        
        # check which ijks are within the specified search radius and add those to ijks_accpeted
        ijks_possible = [(i0,i1,i2) for (i0,i1,i2) in ijks_possible if np.linalg.norm(i0*cell[0,:]+i1*cell[1,:]+i2*cell[2,:])<=r_search]
        if verbose: print("cells after r filter {}".format(len(ijks_possible)))
        if len(ijks_possible) == 0:
            print("Found all cells for r_cut {} => r_search = {}, terminating after {} iterations".format(r_cut,r_search,i+1))
            break

        # add all ijks_possible points to ijks_accepted
        ijks_accepted.update(ijks_possible)
        if verbose:print("accepted new cells: {}".format(len(ijks_accepted)))
        
        # all ijks_to_test points now are saturated, hence add to ijks_saturated
        ijks_saturated.update(ijks_to_test)
        if verbose:print("stored cells so far: {}".format(len(ijks_saturated)))
        
        # remove all previously tested points
        ijks_to_test.clear()
        
        # add all points which were not already known to ijks_to_test
        ijks_to_test.update(ijks_possible)
        if verbose:print("cell to test next round: {}".format(len(ijks_to_test)))
        
        i += 1
    if i == max_iter:
        warnings.warn("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
        raise 
    
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

class super_bond_info:
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
    
    def __init__(self):
        
        # setting input x and output t
        self.observables = {"x":["r","density","r_vec","emb_species","pair_species","neigh_idx_ultra","neigh_idx_super"],
                            "t":["energy","forces"]}
        for obs_type in self.observables:
            setattr(self,obs_type, {val: None for val in self.observables[obs_type]})
                
        # (ghost) atom info
        self.super_pos = None
        self.super_species = None
        self.species_set = None
        self.box = None
        self.N = None # number of atoms
        self.N_neigh = None # number of neighbors for a given atom
        self.ultracell_idx_ori = None # original supercell indices of given ultra cell atom.
        self.name = None # is the name of this structure in the list of parsed structures
        self.index = None # is the number of this structure in the list of parsed structures
                        
    def set_info(self,super_pos,super_species,neigh_idx,ultracell_pos,ultracell_idx_ori,ultracell_cell,
                 t,splines,name,index):
        """
        Calculates properties such as distance vectors, distances and bond angles. these
        properties are then stored according to what has been defined with "observables" 
        during initialization. 
        
        Parameters
        ----------
        
        super_pos : np.ndarray (3,)
            position in r space in supercell
        
        super_species : list of str
            chemical elements of atoms in supercell
        
        neigh_idx : list or np.ndarray of int
            ultracell indices found by neighboring search, not including the self reference of 
            the current atom to itself as neighbor
        
        ultracell_pos : np.ndarray of float (N,3)
            atom positions in rspace
        
        ultracell_idx_ori : list of integer
            length of N containing indices to original supercell atoms for each ultracell atom
        
        ultracell_cell : np.ndarray (3,3) of float
            simulation box for ultracell
        
        t : dict
            contains target values, i.e. {"density":[21.],"E":[42.],"F":[1.,2.,3.],...}
        
        splines : dict of interplate.interp1d
            contains element dependent spline functions of the rho(r) function
        
        name : str
            name of the structure given by the general input parser
        
        index : int
            integer value indicating the position of the structure in the parsed list of
            structures
        """
        self.name = name
        self.index = index
        self.super_pos = super_pos
        self.super_species = super_species
        self.species_set = set(super_species)
        self.all_pairs_set = set([tuple(sorted(list(v))) for v in itertools.product(self.species_set,self.species_set)])
        self.box = ultracell_cell
        self.N = N = len(super_pos)
        self.N_neigh = [len(v) for v in neigh_idx]
        self.t = t
        self.ultracell_idx_ori = ultracell_idx_ori
        
        if len(neigh_idx) == 0:
            r = {}
            r_vec = {}
            r_idx = {}
            density = []
            warnings.warn("neigh_idx is of length zero, nothing to do here...")
        else:
                        
            # initiate distance stuff
            r_vec = [np.array(ultracell_pos[neigh_idx[v]]) - super_pos[v] for v in range(N)]
            r  = [np.linalg.norm(v,axis=1) for v in r_vec]
            
            sorted_idx = [np.argsort(v) for v in r]
            neigh_idx = [[neigh_idx[iv][v] for v in v2] for iv,v2 in enumerate(sorted_idx)]            
            r = [r[v][sorted_idx[v]] for v in range(N)]
            
            r_vec = [r_vec[v][sorted_idx[v]] for v in range(N)]
            r_vec = [tmp_vec/np.reshape(tmp_r,(-1,1)) for tmp_vec,tmp_r in zip(r_vec,r)] # normalizing
            
            # storing super and ultra cell neighboring
            neigh_idx_ultra = neigh_idx
            neigh_idx_super = [[ultracell_idx_ori[neigh_idx[n][i]] for i in range(self.N_neigh[n])] for n in range(self.N)]
            
            # embedding density
            emb_species = [[] for v in range(N)]
            emb_density = np.zeros(N)
            for n in range(N): # atom n
                for i in range(self.N_neigh[n]): # neighbor i of atom n
                    idx = ultracell_idx_ori[neigh_idx[n][i]]
                    species = super_species[idx]
                    emb_density[n] += splines[species](r[n][i])
                    emb_species[n].append(species)
                        
            # pair things
            pair_species = [[] for v in range(N)]
            for n in range(N): # atom n
                curr_species = super_species[n]
                for i in range(self.N_neigh[n]): # neighbor i of atom n
                    idx = ultracell_idx_ori[neigh_idx[n][i]]
                    species = super_species[idx]
                    pair_species[n].append(tuple(sorted([curr_species,species])))
                                  
            self.x["r"] = r
            self.x["r_vec"] = r_vec
            self.x["density"] = emb_density
            self.x["emb_species"] = emb_species
            self.x["pair_species"] = pair_species
            self.x["neigh_idx_ultra"] = neigh_idx_ultra
            self.x["neigh_idx_super"] = neigh_idx_super
        
        for obs_type in self.observables:
            for val in self.observables[obs_type]:
                assert getattr(self,obs_type)[val] is not None, "Assertion failed - after processing self.{}['{}'] was not set!".format(obs_type,val)
    
    def __getitem__(self,key):
        return getattr(self,key)

def get_observations(supercells,rhos_dict,r_cut=6.,num_neigh=None,selection=("random",10.),
                     ultra_num=2,verbose=False,fill_value="extrapolate",kind="linear",idx=None):
    """Produces energies and related input X (embedding density, distances and angles).
    
    Parameters
    ----------
    
    s : instance of general input parser object
    
    rhos_dict : dict
        as created by fitelectrondensity.misc.save_regressed_rho
    
    r_cut : None or float
    
    num_neigh : None or int
    
    selection : tuple or list of tuples
        defines the selection process for parsed DFT structures, like this:
            * ("all") - uses all observations
            * ("uniform",21,"absolute") - choses uniformly at random 21 structures in total
            * ("uniform",42.,"relative") - choses uniformly at random 42 percent of all structures
            * [("Al_MD",x,y,z),("Ni_single",x1,y1,z1)] where the first entry in the tuple is used
                to filter all parsed structures by name and then select them according to the respective
                2nd to 4th tuple field (here denoted as x,y,z and x1,y1,z1) which are interpreted
                in the same way as the previous tuples like ("all") same order, example ("Al_MD","uniform",21,"absolute") ...  
    
    idx : None or np.ndarray of int or list of int
        specifies the supercells to chose by their index. if this is not None it takes precedence over 
        the specified 'selection' parameter 
    
    Notes
    -----
    Note if both r_cut and num_neigh are provided num_neigh is selected. Also, the search is done
    via KD-trees which return indices for the nearest positions, which may include the present atom
    the neighbors are being searched for.
    """
    E_ref = my_globals["E_ref"]
    if verbose: print("Generating observations from DFT configurations and electron density...")
    assert r_cut is not None or num_neigh is not None, "Assertion failed - r_cut and num_neigh are both None please provide valid input."
    if r_cut is not None: assert isinstance(r_cut,(float,int)), "Assertion failed - r_cut is neither int nor float..."
    if num_neigh is not None: assert isinstance(num_neigh,int), "Assertion failed - num_neigh has to be of int type..."
    assert "r" in rhos_dict and "rhos" in rhos_dict, "Assertion failed - keys 'r' and 'rhos' are missing from rhos_dict!"
    
    if r_cut is not None and num_neigh is not None: r_cut = None
    if r_cut is None and num_neigh is not None: num_neigh += 1 # neighborhood search will always find current atom as its own neighbor -> +1
    splines = {}
    # calculating splines
    if fill_value == "extrapolate": warnings.warn("Splining is done using the 'extrapolate' value for the fill_value parameter! This may cause problems, check for adequate cutoff and smoothing settings.")
    for k in rhos_dict["rhos"]:
        # isotropic rhos
        if isinstance(k,str):
            splines[k] = spline(rhos_dict["r"],rhos_dict["rhos"][k],ext=0)

    # looping through supercells   
    if idx is None: 
        idx = select_supercells(supercells,selection)
    
    print("idx {}".format(idx))
    print("selected structures for fitting: {}".format([supercells[v].name for v in idx]))
    
    bonds = []
    print("idx {}".format(idx))
    for i,ix in enumerate(idx):  
        s = supercells[ix]
        # generate ultra cell
        if verbose: print("supercell# {}".format(i))
                                                                        
        ultracell, ultracell_species, ultracell_idx_ori = get_ultracell(s["positions"],s["cell"],s["species"],r_cut=r_cut,verbose=verbose)#ultra_num=ultra_num,search_pos=s["positions"])
        
        eval_points = np.dot(s["positions"],s["cell"])
        
        # setting up neighborhood search
        if r_cut is not None:
            skd = spatial.KDTree(ultracell)
            neigh_idx = skd.query_ball_point(eval_points,r_cut)
            
        elif num_neigh is not None:
            skd = spatial.KDTree(ultracell)
            r,neigh_idx = skd.query(eval_points,k=num_neigh)
        
        neigh_idx = [[v for v in neigh_idx[v2] if np.linalg.norm(ultracell[v]-eval_points[v2])>1e-6] for v2 in range(len(eval_points))] # removing self reference
        
        # putting the info collected for all atoms of the current supercell into a bond
        bond = super_bond_info()
        _energy = s["energy"]/len(s["species"])
        _ref = sum([E_ref[v] for v in s["species"]])/len(s["species"]) #E_ref
        _e_form = _energy - _ref
        print("\nname {}\nenergy {}\nref {} elements {}\ne_form {}".format(s["name"],_energy,_ref,s["species"],_e_form))
        target = {"energy":np.array(_e_form,dtype=np.float64),
                 "forces":np.array(s["forces"],dtype=np.float64)}
        
        bond.set_info(eval_points,s["species"],neigh_idx,ultracell,ultracell_idx_ori,s["cell"],target,splines,s.name,ix)
        bonds.append(bond)
        
    if len(bonds)==0:
        raise ValueError("No bonds were set up! Check the specified path and selection criteria!")
    return bonds

def get_all_targets(bonds,target_types=set(["energy","forces"])):
    """Formats the values to be fitted reading from super_bond_info objects.
    
    """

    targets = {k:None for k in target_types}
    for bond in bonds:
        for k in target_types:
            if targets[k] is None:
                targets[k] = [bond.t[k]]
            else:
                targets[k].append(bond.t[k])
    
    for k in target_types:
        if k == "energy":
            targets[k] = np.array(targets[k],dtype=float)
        else:
            targets[k] = [np.array(v,dtype=float) for v in targets[k]]
    
    return targets

def get_all_sources(bonds,source_types="all",return_rho_bounds=False):
    """Formats the input values for the fitting reading from super_bond_info objects.
    
    Parameters
    ----------
    bonds : list of super_bond_info instances

    source_types : str or set of str
        if "all" then all info stored in [bond.x for bond in bonds] will be copied.
        if set of str then the specified strings are used to query the bond.x attributes
    """
    
    N_bonds = len(bonds)
    if source_types == "all":
        source_types = list(bonds[0].x.keys())
    
    N_bonds = len(bonds)
    sources = {k:[[] for n in range(N_bonds)] for k in source_types}
    sources["species"] = [bond.super_species for bond in bonds]
    sources["N_bonds"] = N_bonds
    sources["N_atoms"] = [bond.N for bond in bonds]
    sources["species_set"] = set(itertools.chain(*[bond.species_set for bond in bonds]))
    sources["species_pair_set"] = set(itertools.chain(*[bond.all_pairs_set for bond in bonds]))
    sources["names"] = [bond.name for bond in bonds]
    sources["index"] = [bond.index for bond in bonds]
    
    for k in source_types:
        for n in range(N_bonds):
            sources[k][n] = bonds[n].x[k]
    lb,ub = {el: 0 for el in sources["species_set"]},{el: 0 for el in sources["species_set"]}
    for n in range(N_bonds):
        print("\nn = {}:\ndensities {}\nspecies {}".format(n,sources["density"][n],sources["species"][n]))
        for el in set(sources["species"][n]):
            idx = np.where(np.array(sources["species"][n])==el)[0]
            tmp_lb = np.amin(sources["density"][n][idx])
            tmp_ub = np.amax(sources["density"][n][idx])
            if tmp_lb < lb[el]:
                lb[el] = tmp_lb
            if tmp_ub > ub[el]:
                ub[el] = tmp_ub
    
    print("lb {} ub {}".format(lb,ub))
    for el in ub:
        ub[el] *= 1
        
    if return_rho_bounds:
        return sources, lb, ub
    else:
        return sources

def get_mapper(k_pair,k_emb,X):
    """Computes a dict for parameter mapping.

    Parameters
    ----------

    k_pair : int

    k_emb : int
    
    X : dict
        Obtained from get_all_sources.

    Returns
    -------

    mapper : dict
        Example {"emb":{"Al":[0,1,2,3,4]}, "pair":{("Al","Al"):[5,6,7,8]}}
    """
    
    mapper = {"pair":{},"emb":{}}
    i = 0
    all_species = sorted(list(X["species_set"]))
    for species in all_species:
        mapper["emb"][species] = np.arange(i,i+k_emb)
        i += k_emb
    all_pairs = sorted(list(X["species_pair_set"]))
    for pair in all_pairs:
        if not isinstance(pair,tuple):
            continue
        mapper["pair"][pair] = np.arange(i,i+k_pair)
        i += k_pair
    return mapper

def get_initial_weights(mapper,seed=None):
    kinds = ["pair","emb"]
    c = 0
    np.random.seed(seed=seed)
    for kind in kinds:
        for key in mapper[kind]:
            c += len(mapper[kind][key])
    return np.random.randn(c)

def load_data(load_path):
    """Loads pre-processed data for EAM regression and
    EAM regression results.

    Parameters
    ----------

    load_path : str
        Path to *.pckl or *.Es file.

    Returns
    -------
    data : dict
    """
    if os.name != "posix": #win os
        _load_path = load_path.replace("\\","/")
    else:
        _load_path = copy.deepcopy(load_path)
    print("Loading data from {}...".format(_load_path))
    with open(_load_path,"rb") as f:
        data = pickle.load(f)
    return data

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

setup_var_names = ["dft_path","selection","load_path_rhos","dump_path",
        "num_neigh","r_cut","k_pair","k_emb","smooth_emb","smooth_pair",
        "type_pair","typ_emb","r_smooth","f_smooth","f_smooth_emb",
        "rho_scaling","r_lb","r_ub","return_rho_bounds","rho_conv_type",
        "rho_operations","rho_params","N_steps","aniso","ultra_num","save",
        "seed","split2groups","dump_fname"]

def sanity_check_eam_setup_vars(args):
    
    print("Sanity checking setup vars...")

    str_vars = ["dft_path","load_path_rhos","dump_path","type_pair","type_emb","dump_fname"]
    dir_vars = ["dft_path"]
    fname_vars = ["load_path_rhos","dump_path"]
    
    for k in str_vars: 
        if k == "dump_path" and args[k] is None and not args["save"]:
            continue
        assert isinstance(args[k],str), "Assertion failed - {} = {}: type {} != str!".format(k,args[k],type(k))
    for k in dir_vars: 
        assert os.path.exists(args[k]), "Assertion failed - {}: path {} does not exist!".format(k,args[k])
    for k in fname_vars: 
        if k == "dump_path" and args[k] == None and not args["save"]:
            continue
        tmp_dir = os.path.dirname(args[k])
        assert os.path.exists(tmp_dir), "Assertion failed - {}: path {} does not exist!".format(k,tmp_dir)
    
    float_vars = ["r_cut","r_smooth","f_smooth","f_smooth_emb","rho_scaling","r_lb","r_ub"]
    bool_vars = ["smooth_emb","smooth_pair","return_rho_bounds","aniso","save","split2groups"]
    int_vars = ["k_pair","k_emb","N_steps"]
    for k in float_vars:
        assert isinstance(args[k],(float,int)), "Assertion failed - type({}) = {} which is neither float nor int!".format(k,type(args[k]))
    for k in int_vars:
        assert isinstance(args[k],int), "Assertion failed - type({}) = {} which is not int!".format(k,type(args[k]))
    for k in bool_vars:
        assert isinstance(args[k],bool), "Assertion failed - type({}) = {} which is not boolean!".format(k,type(args[k]))

def save_Es(path,Es,X,t,calc,regressed_e=None,regressed_f=None,dft_path=None,load_path_rhos=None,rho_dict=None,num_neigh=None,r_cut=3.,aniso=False,
            ultra_num=2,selection=("random",95.),seed=42,k_pair=5,k_emb=5,smooth_emb=False,
            smooth_pair=True,type_pair="Fourier",type_emb="Fourier",r_smooth=3.,f_smooth=.1,
            minimize_method="Nelder-Mead",maxiter=10,lambda_=None,eam_path=None,reg_time=None,
            res=None,logbook=None,fitness_tuple=False,r_lb=0,r_ub=5.,rho_lb=-5,rho_ub=42.,
            N_steps=100,rho_scaling=1.,weights=None,alphas=None,betas=None,
            force_analytic=None,all_fitnesses=None,all_weights=None,
            fit_data_load_path=None,f_smooth_emb=None,mapper_complete=None,x0_complete=None,
            mapper=None,x0=None,all_pair_weights=None,mapper_pair=None,all_potty_energies=None,all_potty_forces=None):    
    """
    Saves all essential settings for the energy force regression to disk as a dict.
    """
    
    print("Saving settings and results to {}...".format(path))
    out = dict(load_path_rhos = load_path_rhos,
                r_rho = rho_dict["r"],
                rhos = rho_dict["rhos"],    
                dft_path = dft_path,
                num_neigh = num_neigh,
                r_cut = r_cut,
                aniso = aniso,
                ultra_num = ultra_num,
                selection=selection, 
                seed = seed,
                k_pair = k_pair,
                k_emb = k_emb,
                smooth_emb = smooth_emb,
                smooth_pair = smooth_pair,
                type_pair=type_pair,
                type_emb=type_emb,
                r_smooth = r_smooth,
                f_smooth = f_smooth,
                minimize_method = minimize_method,
                maxiter = maxiter,
                lambda_ = lambda_,
                eam_path = eam_path,
                reg_time = reg_time,
                res = res,
                logbook = logbook,
                fitness_tuple = fitness_tuple,
                r_lb=r_lb,
                r_ub=r_ub,
                rho_lb=rho_lb,
                rho_ub=rho_ub,
                N_steps = N_steps,
                Es = Es,
                X = X,
                t = t,
                regressed_e = regressed_e,
                regressed_f = regressed_f,
                rho_scaling=rho_scaling,
                weights = weights,
                alphas = alphas,
                betas = betas,
                calc = {"evl_fun":str(calc.evl_fun),"reg_fun":str(calc.reg_fun)},
                force_analytic = force_analytic,
                all_fitnesses = all_fitnesses,
                all_weights = all_weights,
                all_pair_weights = all_pair_weights,
                fit_data_load_path = fit_data_load_path,
                f_smooth_emb=f_smooth_emb,
                mapper_complete = mapper_complete,
                x0_complete = x0_complete,
                mapper = mapper,
                x0 = x0,
                mapper_pair = mapper_pair,
                all_potty_energies=all_potty_energies,
                all_potty_forces=all_potty_forces)
    with open(path,"wb") as f:
        pickle.dump(out,f)
    
def load_Es(Es_path):
    with open(Es_path,"rb") as f:
        Es = pickle.load(f)
    return Es

def get_ase_energies_forces(gip,potential):
    """
    Calculates the energies and forces for the parsed DFT files in the parser object 'gip'
    via the ase EAM calculator object in 'potential'.

    Parameters
    ----------
    
    gip : instance of parsers.GeneralInputParser
        contains all the relevant info as parsed from DFT files such as for castep and so on
    
    potential : instance of ase.calculators.eam
        empirical potential object to calculate energies, forces and such with 

    Returns
    -------

    energies : float np.ndarray of shape (N,)

    forces: list of float np.ndarrays of (Na,3) shapes
    """
    print("Calculating EAM energies and forces via ase...")
    ase_energies, ase_forces = [], []
    for i,n_gip in enumerate(gip):
        print("structure {}/{}".format(i+1,len(gip)))
        elements = n_gip.get_species()
        positions = n_gip.get_positions()
        cell = n_gip.get_cell()
        positions = np.dot(positions,cell)
        s = ase.Atoms(elements,positions=positions,cell=cell,pbc=[1,1,1])
        s.set_calculator(potential)
        ase_energies.append(s.get_potential_energy()/float(len(positions)))
        ase_forces.append(s.get_forces())

    return np.array(ase_energies,dtype=float), ase_forces


def get_complete_weights_split(M,data_pair,data_pure,embmod=False):
	"""Function to re-assemble the complete weights vector for separate alloy regression.

	This function is intended to be used after the regression in parts of an alloy
	by first regressing the pure atomic models and then the mixed systems.

	Parameters
	----------
	M : int
		number of weights in total
	
    data_pair : dict 
		obtained from fitenergy.load_data_for_fit containing the results of 
		mixed element atomic models
	
    data_pure : list of dicts
		each obtained from fitenergy.load_data_for_fit contianing each the 
		results of regressing a set of systems with one specific element

	Returns
	-------
	
    weights : np.ndarray of float of shape (M,)
	"""

	weights = np.zeros(M)
	for key in ["emb","pair"]:
		for k in data_pair["mapper"][key]:
		    mc = data_pair["mapper_complete"][key][k]
		    ms = data_pair["mapper"][key][k]
		    weights[mc] = data_pair["x0"][ms]
		if not embmod:
			for p in data_pure:
			    for k in p["mapper"][key]:
			        mc = p["mapper_complete"][key][k]
			        ms = p["mapper"][key][k]
			        weights[mc] = p["x0"][ms]
	if embmod:
		key = "pair"
		for p in data_pure:
		    for k in p["mapper"][key]:
		        mc = p["mapper_complete"][key][k]
		        ms = p["mapper"][key][k]
		        weights[mc] = p["x0"][ms]
	return weights

def get_complete_weights_joint(M,data_all):
    """Function to re-assemble the complete weights vector for joint alloy regression.

    This function is intended to be used after the regression of an alloy
    to the pure and mixed atomic models at the same time.

    Parameters
    ----------
    
    M : int
        number of weights in total
    
    data_all : dict 
        obtained from fitenergy.load_data_for_fit containing the results of 
        the regression
    
    Returns
    -------
    
    weights : np.ndarray of float of shape (M,)
    """
    weights = np.zeros(M)
    for key in ["emb","pair"]:
        for k in data_all["mapper"][key]:
            mc = ms = data_all["mapper_complete"][key][k]
            weights[mc] = data_all["x0"][ms]
    return weights

def least_square_measure_ref(t,energies,forces):
    """
    Least square measure provided the references and the new forces and energies.
    Energies and forces are NOT being calculated.
    """
    N = len(t["forces"])
    Natoms = np.array([len(v) for v in t["forces"]])
    e_error = np.linalg.norm( (energies - t["energy"]) )
    f_error = sum([np.sum(np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1))/Natoms[v_s] for v_s in range(N)]) / float(3)
    return e_error, f_error

def EAM_setup(seed=None, dft_path=None, selection=None, load_path_rhos=None,\
        dump_path=None, num_neigh=None, r_cut=6., k_pair=5, k_emb=5,\
        smooth_emb=False, smooth_pair=True, type_pair="Fourier",\
        type_emb="Fourier", r_smooth=6., f_smooth=.01, f_smooth_emb=0.01,\
        rho_scaling=1., r_lb=0, r_ub=6., return_rho_bounds=True,\
        rho_conv_type=None, rho_operations=[], rho_params=[6.,0.01], N_steps=1000,\
        ultra_num=None, aniso=False, save=True, split2groups=True, dump_fname="setup",\
        show=False):
    """Function to pre-process DFT files for EAM regression.

    Parameters
    ----------

    seed : int, optional, default None

    dft_path : str, optional, default None
        Path to DFT files.

    selection : list of tuples, optional, default None
        See get_observations() for description.

    load_path_rhos : str, optional, default None
        Path to previously generated *.rhos file containing
        electron density functions.

    dump_path : str, optional, default None
        Directory to write the data to.

    num_neigh : int, optional, default None
        Number of neighbors to consider during neighbourhood search.

    r_cut : float, optional, default 6.
        Cutoff radius to use during neighbourhood search.

    k_pair : int, optional, default 5
        Number of terms to use for pair energy functions.

    k_emb : int, optional, default 5
        Number of terms to use for embedding energy functions.

    smooth_emb : boolean, optional, default False
        If True embedding energy functions will be tapered to zero towards 
        an embedding density of zero.

    smooth_pair : boolean, optional, default True
        If True pair energy functions will be tapered to zero towards r_smooth.

    type_pair : str, optional, default "Fourier"
        Type of basis functions to use for the pair energy functions. 
        Only "Fourier" is currently recognized.

    type_emb : str, optional, default "Fourier"
        Type of basis functions to use for the embedding energy functions. 
        Only "Fourier" is currently recognized.

    r_smooth : float, optional, default 6.
        Distance to taper the pair distance functions to zero at.

    f_smooth : float, optional, default .01
        Pair energy smoothing/tapering factor for the implemented smoothing function x^4/(1+x^4) with
        x = (r-r_smooth)/f_smooth. 

    f_smooth_emb : float, optional, default 0.01
        Embedding energy smoothing/tapering factor for the implemented smoothing function x^4/(1+x^4) with
        x = (rho)/f_smooth_emb. 

    rho_scaling : float, optional, default 1.
        Scaling factor for embedding density terms f(pi*rho/rho_scaling)

    r_lb : float, optional, default 0
        Distance lower bound.

    r_ub : float, optional, default 6.
        Distance upper bound.

    return_rho_bounds : boolean, optional, default True
        Whether or not to return a dict of rho_bounds found during the setup.

    rho_conv_type : str, optional, default None
        Specifies the smoothing of the electron density functions. Note that this 
        is different to the smoothing of the energy functions.
        None: No smoothing takes place.
        "psi": Smoothing takes place using x^4/(1+x^4) for one location.
        "psi2": Smoothing takes place using x^4/(1+x^4) for two locations.

    rho_operations : list of str, optional, default []
        Relates to the smoothing of the electorn density functions. May contain
        "shift", "absolute" and "normalize". The order specifies the operations.
        "shift" leads to a shift for all electron density functions to be larger 
        than zero. "absolute" creates the absolute of all electron density functions.
        "normalize" normalizes all electron density functions to take values lower or 
        equal one everywhere.

    rho_params : list of float, optional, default [6.,0.01]
        Specifies the parameters required for the rho_conv_type. The length 
        depends on the rho_conv_type. For "psi" the first number specifies the
        cutoff whereas the second number is the scale.

    N_steps : int, optional, default 1000
        Number of steps to use when creating function splines.

    ultra_num : int or None, optional, default None
        Number of copies of the supercell to use to generate an ultracell. 
        Ultracells are used here to compute complete neighbourhood informations
        for any given atom and neighbouring condition. If None the ultra_num will 
        be determined automatically (recommended).

    aniso : boolean, optional, default False
        Whether or not to use anisotropic or higher than 2-body approximations.
        Not yet implemented.

    save : boolean, optional, default True
        Whether to save or return the data.

    split2groups : boolean, optional, default True
        Generates multiple *.pckl files one for each element and one for all elements
        if True.

    dump_fname : str , optional, default "setup"
        File name to use and modify to write *.pckl files to.

    show : boolean, optional, default False
        Shows embedding density functions if True.

    Returns
    -------
    Pre-processed data.

    Notes
    -----

    An example how this function is called can be found in the 
    'EAM regression tutorial.ipynb' tutorial notebook in 
    '2. Preparing an EAM regression. This function creates the pre-processed
    data for single element regressions and the multiple element regression
    in form of *.pckl files.
    The multiple element *.pckl file is marked with 'glipglobs'.
    """
    args = locals()
    print("Given EAM setup variables:")
    for k in sorted(args.keys()):
        print("{}:    {}".format(k,args[k]))

    # sanity check variables
    sanity_check_eam_setup_vars(args)

    # set seed
    np.random.seed(seed=seed)

    # parsing DFT files
    gip = parsers.general.GeneralInputParser()
    gip.parse_all(dft_path)
    gip.sort()

    # loading previously created rhos
    rho_dict = fed.misc.load_regressed_rho(load_path_rhos,operations=rho_operations,return_bounds=False,conv=rho_conv_type,params=rho_params,show=show)

    # get bonds
    bonds = get_observations(gip,rho_dict,ultra_num=ultra_num,num_neigh=num_neigh,r_cut=r_cut,verbose=False,selection=selection)

    # get domain and target values
    t = get_all_targets(bonds)
    X, rho_lb, rho_ub = get_all_sources(bonds,return_rho_bounds=return_rho_bounds)

    # rescale rho(r) and embedding densities by the largest observed embedding density
    if rho_conv_type is not None:
        obs_emb = np.amax([v for v2 in X["density"] for v in v2])
        X["density"] = [v/obs_emb for v in X["density"]]
        for _s in rho_dict["rhos"].keys():
            rho_dict["rhos"][_s] /= obs_emb
            rho_min = min([np.amin(v) for v in X["density"]])
            rho_lb[_s] = 0 if rho_min > 0 else rho_min
            rho_ub[_s] = 2*max([np.amax(v) for v in X["density"]])

        print("\nEmbedding densities after re-scaling {}".format([list(v) for v in X["density"]]))
        print("\na0 {}".format([bond.box[0,0] for bond in bonds]))
        print("\nnames {}".format([bond.name for bond in bonds]))
        print("\nBonds with negative densities:")
        for i,bond in enumerate(bonds):
            if any([v<0 for v in X["density"][i]]):
                print("bond {}".format(bond.name))
                print("densities {}".format(X["density"][i]))
            
        print("rho_lb {}\nrho_ub {}".format(rho_lb,rho_ub))
        assert all ([not v<0 for v in rho_lb.values()]), "Assertion failed - found density functions with negative values!"
    else:
        print(X["species"])
        for _s in rho_dict["rhos"].keys():
            _density = [[v2 for iv2,v2 in enumerate(v) if X["species"][iv][iv2]==_s] for iv,v in enumerate(X["density"]) if _s in X["species"][iv]]
            print(_s,": ",_density)
            rho_min = min([np.amin(v) for v in _density])
            rho_max = max([np.amax(v) for v in _density])
            drho = rho_max-rho_min
            rho_min -= .1*drho
            rho_lb[_s] = rho_min if rho_min <= 0 else 0
            rho_ub[_s] = rho_max + .1*drho
        print("rho_lb {}\nrho_ub {}".format(rho_lb,rho_ub))
        
    # parameter mapping and pre-calculation of values for subsequent optimization of model parameters
    mapper = get_mapper(k_pair,k_emb,X)
    params = get_precomputed_energy_force_values(X,f_smooth,r_smooth,rho_dict,type_pair=type_pair,type_emb=type_emb,                                                smooth_emb=smooth_emb,smooth_pair=smooth_pair,rho_scaling=rho_scaling,k_emb=k_emb,k_pair=k_pair,f_smooth_emb=f_smooth_emb)

    # initializing model parameters
    x0 = get_initial_weights(mapper)

    if f_smooth_emb is None:
        f_smooth_emb = f_smooth

    if not split2groups: # for fitting all embedding and pair contributions for all elements at the same time
        data = dict(rho_scaling=rho_scaling,
                    seed=seed,
                    t=t,
                    X=X,
                    mapper=mapper,
                    params=params,
                    x0=x0,
                    smooth_emb=smooth_emb,
                    smooth_pair=smooth_pair,
                    f_smooth=f_smooth,
                    f_smooth_emb=f_smooth_emb,
                    r_smooth=r_smooth,
                    rho_dict=rho_dict,
                    r_cut=r_cut,
                    num_neigh=num_neigh,
                    r_lb=r_lb,
                    r_ub=r_ub,
                    rho_lb=rho_lb,
                    rho_ub=rho_ub,
                    N_steps=N_steps,
                    dft_path=dft_path,
                    load_path_rhos=load_path_rhos,
                    aniso=aniso,
                    ultra_num=ultra_num,
                    selection=selection,
                    k_pair=k_pair,
                    k_emb=k_emb,
                    type_pair=type_pair,
                    type_emb=type_emb)
        if save:
            save_data_for_fit(dump_path,data)
        else:
            return data
    else: # for sequential fitting of first individual element systems and then pair element systems
        out_data = dict()

        species_groups = [list(sorted(bond.species_set)) for bond in bonds] # i.e. [["Al"],["Al","Ni"],["Al","Ni"]] - reflects all unique elements present in a phase
        print("species sets by structure:")
        for i in range(len(species_groups)):
            print("{} | {}".format(bonds[i].name,species_groups[i]))

        group_name = "glipglobs" # designation for all phases with more than one element

        eam_groups = {group_name:[]} # collection of structures by id and assignment to single element groups or the group specified by variable group_name
        print("species groups ",species_groups)
        for i,s in enumerate(species_groups):
            if len(s) == 1: # single element
                pure_name = s[0]
                if pure_name in eam_groups:
                    eam_groups[pure_name].append(i)
                else:
                    eam_groups[pure_name] = [i]
            #else:
            eam_groups[group_name].append(i) # collecting all structures for the glipglobs fit
        
        for eg in eam_groups:
            eam_groups[eg] = np.array(eam_groups[eg],dtype=int)
        print("eam_groups ",eam_groups)

        for eg,idx in eam_groups.items():
            print("\nProcessing group {}\n".format(eg))
            N_bonds = len(idx)
            species_set_eg = set(itertools.chain(*[species_groups[v] for v in idx]))
            species_pair_set_eg = set([tuple(sorted(list(v))) for v in itertools.product(species_set_eg,species_set_eg)])
            print("species pairs {}".format(species_pair_set_eg))
            
            # target values
            t_eg = {"energy":[t["energy"][v] for v in idx],
                    "forces":[t["forces"][v] for v in idx]}
                    
            # source values
            X_eg = {"species_set":species_set_eg,"species_pair_set":species_pair_set_eg,
                    "N_bonds":N_bonds}
            
            # updating source with remaining parameters using idx
            for k in X:
                if not k in X_eg:
                    X_eg[k] = [X[k][v] for v in idx]

            # "mapper_eg": specific weights mapper for single element or glipglobs cases (the global variable "mapper" already exists)
            # {"weights_eg","mapper_eg"} for all eg in "eam_groups" => obtain "weights" with "mapper"
            mapper_eg = dict(emb=dict(),pair=dict())
            sse = sorted(list(species_set_eg))
            print("sse ",sse)
            c = 0
            if True: #eg != group_name:
                for k in sse:
                    mapper_eg["emb"][k] = np.arange(c,c+len(mapper["emb"][k]))
                    c += len(mapper["emb"][k])
            spse = sorted(list(species_pair_set_eg))
            print("spse ",spse)
            for k in spse:
                if eg == group_name:
                    if k[0]==k[1]:
                        continue
                mapper_eg["pair"][k] = np.arange(c,c+len(mapper["pair"][k]))
                c += len(mapper["pair"][k])

            print("mapper_eg {}".format(mapper_eg))
            
            # initial guess for x0 specific for this case
            x0_eg = np.zeros(sum([len(mapper_eg["emb"][k]) for k in mapper_eg["emb"]])+\
                                sum([len(mapper_eg["pair"][k]) for k in mapper_eg["pair"]]))
            print("x0_eg {}".format(x0_eg.shape))
            # transferring of x0 from the global mapper to x0_eg
            for k in mapper_eg["emb"]:
                x0_eg[mapper_eg["emb"][k]] = x0[mapper["emb"][k]]
            for k in mapper_eg["pair"]:
                x0_eg[mapper_eg["pair"][k]] = x0[mapper["pair"][k]]

            params_eg = {"N_bonds": N_bonds}
            params_eg.update({k:[params[k][v] for v in idx] for k in params if not k in params_eg})

            
            data = dict(rho_scaling=rho_scaling,
                    seed=seed,
                    t=t_eg,
                    X=X_eg,
                    mapper=mapper_eg,
                    mapper_complete=mapper,
                    params=params_eg,
                    x0=x0_eg,
                    x0_complete=x0,
                    smooth_emb=smooth_emb,
                    smooth_pair=smooth_pair,
                    f_smooth=f_smooth,
                    f_smooth_emb=f_smooth_emb,
                    r_smooth=r_smooth,
                    rho_dict=rho_dict,
                    r_cut=r_cut,
                    num_neigh=num_neigh,
                    r_lb=r_lb,
                    r_ub=r_ub,
                    rho_lb=rho_lb,
                    rho_ub=rho_ub,
                    N_steps=N_steps,
                    dft_path=dft_path,
                    load_path_rhos=load_path_rhos,
                    aniso=aniso,
                    ultra_num=ultra_num,
                    selection=selection,
                    k_pair=k_pair,
                    k_emb=k_emb,
                    type_pair=type_pair,
                    type_emb=type_emb)
            if save:
                tmp = os.path.dirname(dump_path)
                save_data_for_fit(tmp+"/{}_{}.pckl".format(dump_fname,eg),data)
            else:
                out_data[ef] = data
        return out_data