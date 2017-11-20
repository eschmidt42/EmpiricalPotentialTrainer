import parsers
import time
import pickle
import numpy as np
from scipy import interpolate, spatial
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import itertools, copy, warnings, collections, pickle, ase
from ase.calculators.eam import EAM
import matplotlib.pylab as plt

from scoop import futures
from collections import deque
from deap import base, creator, tools, cma

my_globals = {"E_ref":{"Al":-107.344221-0.172105115032,
                       "Nb":-1647.496121,
                       "Ni":-1358.048079374}}

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
    
def get_ultracell_old(fpos,cell,species,ultra_num=None,r_cut=6,search_pos=None):
    
    ori_idx = np.arange(len(fpos))
    if ultra_num is not None:
        i_range = np.arange(-ultra_num,ultra_num+1)
        ijks = list(itertools.product(i_range,i_range,i_range))
        
        e0, e1, e2 = np.array([1,0,0],dtype=float), np.array([0,1,0],dtype=float), np.array([0,0,1],dtype=float)
        for h,(i,j,k) in enumerate(ijks):
            tmp_fpos = np.array(fpos) + i * e0 + j*e1 + k*e2
            tmp_species = np.array(species)

            if h == 0:
                ultracell_fpos = tmp_fpos
                ultracell_species = tmp_species
                ultracell_ori_idx = ori_idx
            else:
                ultracell_fpos = np.vstack((ultracell_fpos,tmp_fpos))
                ultracell_species = np.hstack((ultracell_species,tmp_species))
                ultracell_ori_idx = np.hstack((ultracell_idx,ori_idx))
        
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
        
        delta_comp = np.amin(np.diag(cell))
        delta_comp_max = np.amax(np.diag(cell))
        basis = np.eye(3)
        delta_int = int((delta_comp+r_cut)/delta_comp+.5)
        
        ijk = np.arange(-delta_int-1,delta_int+2)
        ijks = np.array(list(itertools.product(ijk,ijk,ijk)),dtype=int)
        
        for h,(i,j,k) in enumerate(ijks):
            if h == 0:
                grid_fpos = fpos + i*basis[0,:] + j*basis[1,:] + k*basis[2,:]
                ultracell_species = tmp_species
                ultracell_ori_idx = ori_idx
            else:
                grid_fpos = np.vstack((grid_fpos,fpos + i*basis[0,:] + j*basis[1,:] + k*basis[2,:]))
                ultracell_species = np.hstack((ultracell_species,tmp_species))
                ultracell_ori_idx = np.hstack((ultracell_ori_idx,ori_idx))
        
        grid_pos = np.dot(grid_fpos,cell)
        
        skd = spatial.KDTree(grid_pos)
        idx = skd.query_ball_point(np.dot(search_pos,cell),2*r_cut)
        all_idx = np.array(list(set([v for v2 in idx for v in v2])),dtype=int)
        
        ultracell_fpos = grid_fpos[all_idx]
    else:
        raise ValueError("In order for dynamic generation of the ultracell when ultra_num == None r_cut is required to be int or float! Got r_cut = {} ...".format(r_cut))     

    return np.dot(ultracell_fpos,cell), ultracell_species[all_idx], ultracell_ori_idx[all_idx]

def get_ultracell(fpos,cell,species,r_cut,show=False,verbose=False,max_iter=20):
    if verbose:
        print("Generating ultracell with r_cut = {}".format(r_cut))
    Vcell = np.absolute(np.linalg.det(cell))
    
    # find center and corners of the cell
    center = .5 * cell.sum(axis=1)
    
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
            if verbose:
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
                                                                        
        ultracell, ultracell_species, ultracell_idx_ori = get_ultracell(s["positions"],s["cell"],s["species"],r_cut=r_cut,verbose=True)#ultra_num=ultra_num,search_pos=s["positions"])
        
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

def get_mapper(k_pair,k_emb,X):
    # calculate : parameter mapping
    
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

def get_initial_weights(mapper):
    kinds = ["pair","emb"]
    c = 0
    for kind in kinds:
        for key in mapper[kind]:
            c += len(mapper[kind][key])
    return np.random.randn(c)

def calculate_all_energies(x,params,mapper):
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
    return tmp_emb_energy, tmp_pair_energy
    
def energy_wrapper(params,mapper):
    def wrapped_energy_fun(x): # x is an np.ndarray containing the weights
        E = calculate_all_energies(x,params,mapper)
        return E[0] + .5*E[1]
    return wrapped_energy_fun

def force_spline_wrapper(params,mapper,X,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,
                         r_lb=0,r_ub=5.,rho_lb=-5.,rho_ub=42.,N_steps=100,rho_scaling=1.): # temporary solution to a bug in the analytic force calculation...
    def wrapped_force_fun(x):
        return get_forces_from_splines(x,params,mapper,X,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,\
                         r_lb=r_lb,r_ub=r_ub,rho_lb=rho_lb,rho_ub=rho_ub,N_steps=N_steps,rho_scaling=rho_scaling)
    return wrapped_force_fun

def get_forces_from_splines(x,params,mapper,X,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,
                         r_lb=0,r_ub=5.,rho_lb=-5.,rho_ub=42.,N_steps=100,rho_scaling=1.):
    
    N_super = X["N_bonds"]
    N_atoms = X["N_atoms"]
    Es = get_splined_functions(x,mapper,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,r_lb=r_lb,
                          r_ub=r_ub,rho_lb=rho_lb,rho_ub=rho_ub,N_steps=N_steps,rho_scaling=rho_scaling)

    funs = {"pair":{p:spline(Es["r"],v) for p,v in Es["pair"].items()},
                   "emb":{s:spline(Es["rho"],v) for s,v in Es["emb"].items()},
                   "rhos":{r:spline(Es["r"],v) for r,v in Es["rhos"].items()},}
    derivatives = {"pair":{p:v.derivative(n=1) for p,v in funs["pair"].items()},
                   "emb":{s:v.derivative(n=1) for s,v in funs["emb"].items()},
                   "rhos":{r:v.derivative(n=1) for r,v in funs["rhos"].items()},}
    all_forces = [None for v in range(N_super)]
    for s in range(N_super): # iterate supercells
        energy = 0.
        forces = np.zeros((N_atoms[s],3),dtype=float)
        
        for n in range(N_atoms[s]): # iterate atoms
            
            N_neigh = len(X["r"][s][n])
            curr_species = X["species"][s][n]
            
            emb_dens = X["density"][s][n] 
                        
            fprho = np.zeros(3,dtype=float)
            emb_neigh = np.zeros(3,dtype=float) # neighborhood contribution to the embedding force
            forces_pair = np.zeros(3,dtype=float)
            for i in range(N_neigh): # iterate neighbors of atom
                idx = X["neigh_idx_super"][s][n][i] # index in supercell corresponding to neighbor found in ultracell
                species = X["emb_species"][s][n][i]
                r = X["r"][s][n][i]                
                vec = X["r_vec"][s][n][i]
                emb_dens_neigh =  X["density"][s][idx]
                
                # embedding
                fprho += vec * derivatives["rhos"][species](r)  
                emb_neigh += vec * derivatives["rhos"][curr_species](r) * derivatives["emb"][species](emb_dens_neigh) 
                
                # pair
                forces_pair += vec * derivatives["pair"][tuple(sorted([curr_species,species]))](r)
                
            forces_emb = fprho * derivatives["emb"][curr_species](emb_dens) + emb_neigh
            forces[n,:] = forces_pair + forces_emb
        all_forces[s] = forces
    return all_forces

def calculate_all_forces(x,params,mapper):
    """
    params = {"coskrho":coskrho,"psirho":psirho,"coskr":coskr,"psir":psir, # energy related
              "ksinkr":ksinkr,"psipr":psipr,"ksinkrho":ksinkrho,"psiprho":psiprho, # force related
              "N_bonds":N_super,"N_atoms":N_atoms,
              "emb_species":X["emb_species"],"pair_species":X["pair_species"]}
    """
    tmp_x = np.array(x,dtype=float)
    
    all_forces = [np.zeros((params["N_atoms"][v_s],3)) for v_s in range(params["N_bonds"])]
    
    tmp_emb_force = np.zeros(3,dtype=np.float64)
    tmp_pair_force = np.zeros(3,dtype=np.float64)
    
    tmp_emb_n = np.zeros(3,dtype=np.float64)
    tmp_emb_i = np.zeros(3,dtype=np.float64)
    tmp_pair_n = np.zeros(3,dtype=np.float64)
    tmp_pair_i = np.zeros(3,dtype=np.float64)
    for s in range(params["N_bonds"]):
        print("\ns {}".format(s))
        for n in range(params["N_atoms"][s]):
            
            print("n {}".format(n))
            curr_species = params["species"][s][n]
                                                
            ### embedding
            # dE(rho_n)/drho_n
            w_emb_n = tmp_x[mapper["emb"][curr_species]]

            fprho_n = params["fprho_n"][s][n]
            print("fprho_n {}".format(fprho_n))
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
                if i==0: print("fprho_i {}".format(fprho_i))

                psiprho_i = params["psiprho"][s][idx]
                coskrho_i = params["coskrho"][s][:,idx]
                psirho_i = params["psirho"][s][idx]
                ksinkrho_i = params["ksinkrho"][s][:,idx]

                tmp_emb_i += fprho_i * np.dot(w_emb_i, psiprho_i*coskrho_i - psirho_i*ksinkrho_i)                
            
            # total embedding force on atom n of supercell s
            print("emb_n {}\nemb_i {}".format(tmp_emb_n,tmp_emb_i))
            tmp_emb_force[:] = 0.
            tmp_emb_force = tmp_emb_n + tmp_emb_i
            
            print("tmp_emb_force {}".format(tmp_emb_force))
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
            
            # total pair force on atom n of supercell s
            tmp_pair_force[:] = tmp_pair_n #+ tmp_pair_i
            print("tmp_pair_force {}".format(tmp_pair_force))
            ### total
            all_forces[s][n] = tmp_emb_force + tmp_pair_force
    
    return all_forces
    
def force_wrapper(params,mapper):
    def wrapped_force_fun(x): # x is an np.ndarray containing the weights
        return calculate_all_forces(x,params,mapper)
    return wrapped_force_fun

def plot_potential(Es):
    
    fig = plt.figure(figsize=(15,10))
    
    ax1 = fig.add_subplot(121)
    plt.hold(True)
    for species in Es["emb"]:
        ax1.plot(Es["rho"],Es["emb"][species],'-',label=species,linewidth=2.,alpha=0.8)
    plt.hold(False)
    plt.grid()
    plt.legend(loc=0)
    ax1.set_xlabel("rho")
    ax1.set_ylabel("E_emb")
    
    ax2 = fig.add_subplot(122)
    plt.hold(True)
    for pair in Es["pair"]:
        plt.plot(Es["r"],Es["pair"][pair],'-',label=pair,linewidth=2.,alpha=0.8)
    plt.hold(False)
    plt.grid()
    ax2.set_xlabel("r")
    ax2.set_ylabel("E_pair")
    plt.legend(loc=0)
    plt.show()

def get_initial_hyperparameters(weights):
    betas = np.absolute(np.random.randn(2))
    alphas = np.absolute(np.random.randn(len(weights)))
    return alphas, betas

def bayesian_measure(x,alphas=None,betas=None,t=None,e_fun=None,f_fun=None):
    tmp_x = np.array(x,dtype=float)
    energies = e_fun(tmp_x)
    forces = f_fun(tmp_x)
    
    N = len(t["forces"])
    M = len(tmp_x)
        
    logp_e = -.5 * betas[0] * np.sum((energies - t["energy"])**2)
    logp_f = -.5 * betas[1] * np.sum([np.sum(np.linalg.norm(forces[v_s]-t["forces"][v_s],axis=1)**2)/3. for v_s in range(N)])
    logp_w = -.5 * np.dot(tmp_x**2,alphas)
    logp_hyper = N * .5 * np.sum(np.log(betas)) + .5 * np.sum(np.log(alphas))
    logp_const = -.5 * (N+M) * np.log(2*np.pi)
    
    fitness = - (logp_e + logp_f + logp_w + logp_hyper + logp_const)
    
    return - (logp_e + logp_f + logp_w + logp_hyper + logp_const) # the negative sign in front is on purpose, since this is input for a minimizer

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

def least_square_measure(x,t=None,e_fun=None,f_fun=None):
    """
    Least square measure provided the references, the functions and the weights.
    Energies and forces are being calculated.
    """
    tmp_x = np.array(x,dtype=float)
    energies = e_fun(tmp_x)
    forces = f_fun(tmp_x)
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
    def __init__(self,params,targets,opts=dict(),reg_fun=None,evl_fun=None,fitness_tuple=False):
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

def get_energies_forces_from_splines(X,Es,verbose=False):
    print("\nCalculating energies and forces from splines...")
    N_super = X["N_bonds"]
    N_atoms = X["N_atoms"]

    funs = {"pair":{p:spline(Es["Es"]["r"],v) for p,v in Es["Es"]["pair"].items()},
                   "emb":{s:spline(Es["Es"]["rho"],v) for s,v in Es["Es"]["emb"].items()},
                   "rhos":{r:spline(Es["Es"]["r"],v) for r,v in Es["Es"]["rhos"].items()},}
    derivatives = {"pair":{p:v.derivative(n=1) for p,v in funs["pair"].items()},
                   "emb":{s:v.derivative(n=1) for s,v in funs["emb"].items()},
                   "rhos":{r:v.derivative(n=1) for r,v in funs["rhos"].items()},}
    all_energies = np.zeros(N_super,dtype=float)
    all_forces = [np.zeros((s,3),dtype=float) for s in N_atoms]

    for s in range(N_super): # iterate supercells
        energy = 0.
        forces = np.zeros((N_atoms[s],3),dtype=float)
        if verbose: print("\ns {}".format(s))
        for n in range(N_atoms[s]): # iterate atoms
            
            N_neigh = len(X["r"][s][n])
            curr_species = X["species"][s][n]
            
            emb_dens = X["density"][s][n] 
            energy_emb = funs["emb"][curr_species](emb_dens)
            energy_pair = .5 * np.sum([funs["pair"][X["pair_species"][s][n][v]](X["r"][s][n][v])  for v in range(N_neigh)])
            energy += energy_emb + energy_pair
            
            fprho = np.zeros(3,dtype=float)
            emb_neigh = np.zeros(3,dtype=float) # neighborhood contribution to the embedding force
            forces_pair = np.zeros(3,dtype=float)
            for i in range(N_neigh): # iterate neighbors of atom
                idx = X["neigh_idx_super"][s][n][i] # index in supercell corresponding to neighbor found in ultracell
                species = X["emb_species"][s][n][i]
                r = X["r"][s][n][i]                
                vec = X["r_vec"][s][n][i]
                emb_dens_neigh =  X["density"][s][idx]
                
                # embedding
                fprho += vec * derivatives["rhos"][species](r)  
                emb_neigh += vec * derivatives["rhos"][curr_species](r) * derivatives["emb"][species](emb_dens_neigh) 
                if i == 0 and verbose: print("fprho_i {}".format(vec * derivatives["rhos"][curr_species](r)))
                # pair
                forces_pair += vec * derivatives["pair"][tuple(sorted([curr_species,species]))](r)
                
            forces_emb = fprho * derivatives["emb"][curr_species](emb_dens) + emb_neigh
            forces[n,:] = forces_pair + forces_emb
            if verbose:
                print("fprho_n {}".format(fprho))
                print("emb_n {}\nemb_i {}".format(fprho * derivatives["emb"][curr_species](emb_dens),emb_neigh)) 
                print("forces_emb {}\nforces_pair {}\n".format(forces_emb,forces_pair))
        all_forces[s] = forces
        all_energies[s] = energy 
        if verbose: print("\ns {} energy = {}\nforces = {}".format(s,energy/float(N_atoms[s]),forces))
    return all_energies, all_forces

def get_splined_functions(weights,mapper,smooth_emb,smooth_pair,f_smooth,r_smooth,rho_dict,r_lb=0,
                          r_ub=5.,rho_lb=-5,rho_ub=42.,N_steps=100,rho_scaling=1.,verbose=False):
    
    def _simple(x):
        if isinstance(x,np.ndarray):
            return np.ones(x.shape)
        else:
            return 1.

    if verbose: print("Splining energy functions...")
    if smooth_emb:
        smooth_fun_rho = wrapped_smooth(f_smooth,0,kind="<") # the contribution of the embedding energy is zero if the background density is zero
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

    drho = np.round((rho_ub - rho_lb)/float(N_steps),decimals=6)
    dr = np.round((r_ub - r_lb)/float(N_steps),decimals=6)
    if verbose:
        print("drho {}".format(drho))
        print("dr {}".format(dr))   

    # rho and embedding energy
    rho = np.array([rho_lb+v*drho for v in range(N_steps)])
    Es["rho"] = rho
    f_rho = np.pi/float(rho_scaling)
    
    if smooth_emb:
        E_emb = lambda w,x: np.dot(w,np.array([smooth_fun_rho(x)*np.cos(f_rho*v*x) for v in range(len(w))]))
    else:
        E_emb = lambda w,x: np.dot(w,np.array([np.cos(f_rho*v*x) for v in range(len(w))]))
    
    for species in mapper["emb"]:
        Es["emb"][species] = E_emb(weights[mapper["emb"][species]],rho)
    
    # r and pair energy
    r = np.array([r_lb+v*dr for v in range(N_steps)])
    Es["r"] = r
    f_r = np.pi/float(r_smooth)
    if smooth_pair:
        E_pair = lambda w,x: np.dot(w,np.array([smooth_fun_r(x)*np.cos(f_r*v*x) for v in range(len(w))]))
    else:
        E_pair = lambda w,x: np.dot(w,np.array([np.cos(f_r*v*x) for v in range(len(w))]))
    
    for pair in mapper["pair"]:
        Es["pair"][pair] = E_pair(weights[mapper["pair"][pair]],r)

    # density functions
    for species in mapper["emb"]:
        dens_spline = spline(rho_dict["r"],rho_dict["rhos"][species],ext=0)
        Es["rhos"][species] = dens_spline(r)

    return Es

def save_Es(path,Es,X,t,calc,regressed_e=None,regressed_f=None,dft_path=None,load_path_rhos=None,rho_dict=None,num_neigh=None,r_cut=3.,aniso=False,
            ultra_num=2,selection=("random",95.),seed=42,k_pair=5,k_emb=5,smooth_emb=False,
            smooth_pair=True,type_pair="Fourier",type_emb="Fourier",r_smooth=3.,f_smooth=.1,
            minimize_method="Nelder-Mead",maxiter=10,lambda_=None,eam_path=None,reg_time=None,
            res=None,logbook=None,fitness_tuple=False,r_lb=0,r_ub=5.,rho_lb=-5,rho_ub=42.,
            N_steps=100,rho_scaling=1.,weights=None,alphas=None,betas=None,
            force_analytic=None,all_fitnesses=None,all_weights=None,
            fit_data_load_path=None,f_smooth_emb=None):    
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
                fit_data_load_path = fit_data_load_path,
                f_smooth_emb=f_smooth_emb,)
    with open(path,"wb") as f:
        pickle.dump(out,f)

def deap_cma_optimizer(fun,N,MAXITER=100,verbose=True,init_lb=-4,init_ub=4,sigma=10.,lambda_=None):
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
    creator.create("FitnessMin", base.Fitness, weights=(-1.,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("map",futures.map)
    toolbox.register("evaluate", fun)

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    t = 0
    if lambda_ is None:
        #lambda0 = 4 + int(3 * np.log(N))
        #lambda_ = int(lambda0 * (0.5**(np.random.rand()**2)))
        lambda_ = 20*int(N)
    else:
        lambda_ = int(lambda_)
    print("lambda_ {}".format(lambda_))
    #sigma = 2 * 10**(-2 * np.random.rand())

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

def write_EAM_setfl_file(eam_path,Es,show=False,
                        atom_info={'Ni':{"mass":58.6934,"number":28,"lattice":"fcc","a0":3.52},
                                   'Al':{"mass":26.9815,"number":13,"lattice":"fcc","a0":4.02},
                                   'Ti':{"mass":47.88,"number":22,"lattice":"fcc","a0":2.95},
                                   'Nb':{"mass":92.90637,"number":41,"lattice":"bcc","a0":3.3}}):
    """
    This function takes the regressed empirical potential in 'Es' and writes it to 
    disk in form of a setfl file with .eam.alloy ending    
    """
    if eam_path[-10:] != ".eam.alloy":
        warnings.warn("The specified file doesn't end on '.eam.alloy'! The file is modified accordingly...")
        eam_path += ".eam.alloy"
    
    elements = list(sorted(Es["Es"]["emb"].keys()))
    N_ele = len(elements)
    pair_map = {tuple([el1,el2]):tuple(sorted([el1,el2])) for el1,el2 in itertools.product(elements,elements)}
    
    dr = Es["Es"]["r"][1] - Es["Es"]["r"][0]
    drho = Es["Es"]["rho"][1] - Es["Es"]["rho"][0]
    
    Nrho = Es["N_steps"]
    Nr = Es["N_steps"]  

    EAM_obj = EAM(elements=elements, embedded_energy=np.array([spline(Es["Es"]["rho"],Es["Es"]["emb"][el]) for el in elements]),
                    electron_density=np.array([spline(Es["Es"]["r"],Es["Es"]["rhos"][el]) for el in elements]),
                    phi=np.array([[spline(Es["Es"]["r"],Es["Es"]["pair"][(pair_map[(el2,el1)])]) for el2 in elements] for el1 in elements]), 
                    cutoff=Es["r_cut"], form='alloy',
                    Z=[atom_info[el]["number"] for el in elements], nr=Nr, nrho=Nrho, dr=dr, drho=drho,
                    lattice=[atom_info[el]["lattice"] for el in elements], 
                    mass=[atom_info[el]["mass"] for el in elements], 
                    a=[atom_info[el]["a0"] for el in elements])
    setattr(EAM_obj,"Nelements",N_ele)
    
    print("Writing setfl file to {}...".format(eam_path))
    if show:
        EAM_obj.plot()
        plt.show()
    EAM_obj.write_potential(eam_path)

def load_Es(Es_path):
    with open(Es_path,"rb") as f:
        Es = pickle.load(f)
    return Es

def get_ase_energies_forces(gip, potential, verbose=False):
    """Function for energy and force calculation using an EAM calculator.
    
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
        N energy values, one for each crystal.
    
    forces : list of N float np.ndarrays of shape (Na,3)
        Forces for each crystal. Each crystal has Na atoms.
    """
    if verbose:
        print("Calculating EAM energies and forces via ase...")
    
    ase_energies, ase_forces = [], []
    for i,n_gip in enumerate(gip):
        if verbose:
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