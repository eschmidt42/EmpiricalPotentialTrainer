import numpy as np
import copy, warnings
from .misc import translating_params, symmetrize_params

#-------------------------------------------------------------------#
# DEVELOPER NOTES                                                   #
# If adding a potential, must alter these functions in the file     #
#                                                                   #
# - calculate_supercell_bonds                                       #
#-------------------------------------------------------------------#

import itertools, copy, random, pickle
import numpy as np

def load_rhos(path):
    with open(path,'rb') as f:
        rhos = pickle.load(f)
    return rhos

def damping(x,f=10.):
    x4 = (f*x)**4
    return x4/(1+x4)

def damping_prime(x,f=10.):
    x4 = (f*x)**4
    x3 = (f*x)**3
    return 4*x3/(1+x4)**2

def _derivative_of_spline(x,f,shift=1e-4,lb=None,ub=None):
    """
    Parameters
    ----------
    x : numpy.ndarray
        argument for f
    f : function object
    shift : float
        two axes will be generated which will be used to calculate the symmetric
        finite difference derivative of f. shift specifies by how much these
        axes are shifted relatively.
    """
    #print("derivative bounds lb {} ub {}".format(lb,ub))
    xp = [x-shift,x+shift]
    if lb is not None and ub is not None:
        xp = np.array([[v0,v1] for v0,v1 in zip(xp[0],xp[1]) if v0>=lb and v1<=ub])
    elif lb is not None:
        xp = np.array([[v0,v1] for v0,v1 in zip(xp[0],xp[1]) if v0>=lb])
    elif ub is not None:
        xp = np.array([[v0,v1] for v0,v1 in zip(xp[0],xp[1]) if v1<=ub])
    #print("min xp {} max xp {}".format(xp[0],xp[-1]))
    yp = f(xp)
    yp = (yp[:,1]-yp[:,0])/(xp[0,1]-xp[0,0])
    xp2 = np.mean(xp,axis=1)
    return interp1d(xp2,yp,bounds_error=False,fill_value=np.nan)

def iso_contribution(pair,pair_distances,pair_elements,iso_rho):
    """Calculates isotropic density contribution for given pair.
    
    Returns
    -------
    densities : np.ndarray of float
        shape (N,)
    """
        
    #outer loop over atoms in configuration and inner loop over a single atom's neighborhood
    densities = [np.sum([iso_rho[pair_elements[v][v2]](pair_distances[v][v2]) for v2 \
                         in range(len(pair_distances[v]))]) for v in range(len(pair_distances))]
    return np.array(densities)

def ani_contribution(triplet,triplet_distances,triplet_elements,ani_rho):
    
    densities = [np.sum(ani_rho[tuple(triplet)](triplet_distances[v])) for v in range(len(triplet_distances))]    
    return np.array(densities)

def _rho_Duff2015(dens_atom):
    return np.array([np.sqrt(dens_atom), dens_atom**2, dens_atom**3])

def _emb_energy(rho,p):
    return np.dot(np.array(p),rho)

def _emb_energy_damped(rho,p,dens):
    return np.dot(np.array(p),rho)*damping(dens)

def _rho_Fourier(dens_atom,num_k):
    return np.array([np.cos(v*dens_atom) for v in range(num_k)])

def calculate_embedding_energy(pair_distances,pair_elements,crys_elements,rhos,params={"k":[1,1e-1,1e-2]},
                               triplet_distances=None,triplet_elements=None,edensity=None):
    """Function to calculate the embedding energy.
    
    In with this function the embedding energies are calculate for all atoms.
    
    Parameters
    ----------
    pair_distances : dict
        {("A","A"):[np.array([...],dtype=float),np.array([...],dtype=float),...]}
    pair_elements : dict
        {("A","A"):[np.array([...],dtype=str),np.array([...],dtype=str),...]}
    rhos : dict
        {"iso":{"A":...,...},"aniso":{("A","B"):...,...}
    params : dict
        {"abc":[a,b,c], "t":[t_1,t_2,...]}
    
    """
    
    num_atoms = len(crys_elements)
    
    if edensity is None:
        _calc_edensity = True
        iso_dens_atom = np.zeros(num_atoms)
        ani_dens_atom = np.zeros(num_atoms)
        pairs = pair_distances.keys()
        if triplet_distances is not None:
            triplets = triplet_distances.keys()
        assert set(pairs) == set(pair_elements.keys()), "Assertion failed - pair_distances and pair_elements have different keys: {} and {}".format(pairs,pair_elements.keys())
    
        #iso contribution - looping over pairs
        for pair in pairs:
            iso_dens_atom += iso_contribution(pair,pair_distances[pair],pair_elements[pair],rhos["iso"])

        iso_dens_atom = np.array(iso_dens_atom)
        ani_dens_atom = np.zeros(len(iso_dens_atom))
        ani_dens_atom = np.array(ani_dens_atom)

        #ani contribution
        if triplet_distances is not None:
            for triplet in triplets:
                ani_dens_atom += ani_contribution(triplet,triplet_distances[triplet],triplet_elements[triplet],rhos["aniso"])
        dens_atom = iso_dens_atom + ani_dens_atom
    else:
        _calc_edensity = False
        dens_atom = edensity
    ###calc embedding energy - expansion with rho    
    e_emb = np.zeros(num_atoms)
    elements = set(crys_elements)
    
    rho = _rho_Fourier(dens_atom,len(params[next(iter(elements))]["k"]))
    
    for ele in elements: #calculate element specific embedding energies for each atom
        idx = np.where(np.array(crys_elements)==ele)[0]
        e_emb[idx] = _emb_energy_damped(rho[:,idx],np.array(params[ele]["k"]),dens_atom[idx])
        
    if _calc_edensity:
        return e_emb, dens_atom
    else:
        return e_emb

def iso_contribution_bondstyle(pair,grp_bond_list,iso_rho):
    """Calculates isotropic density contribution for given pair.
    
    Parameters
    ----------
    pair : tuple of str
        element pair (first entry is always the central atom and the second is the neighborhood)
    grp_bond_list : list of bond objects
        bond objects for a specific bond_group
    iso_rho : dict of splines
        contain density function obtained from RVM regression of DFT electron densities
    
    Returns
    -------
    densities : np.ndarray of float
        shape (N,) where N is the number of atoms
    """
        
    #outer loop over atoms in configuration and inner loop over a single atom's neighborhood
    num_atoms = len(grp_bond_list)
    densities = np.array([np.nan]*num_atoms,dtype=float)
    for i in range(num_atoms):
        b = grp_bond_list[i]
        r = b["bonded_atoms"][1:,:]-b["bonded_atoms"][0,:]
        r = np.dot(r,b["cell"])
        r = np.linalg.norm(r,axis=1)
        densities[i] = np.sum([iso_rho[pair[1]](v) for v in r])    
    return np.array(densities)

def calculate_embedding_energy_bondstyle(bond_grps,bond_list,crys_elements):
    """Function to calculate the embedding energy.
    
    In with this function the embedding energies are calculate for all atoms.
    
    Parameters
    ----------
    bond_grps : list of bond_group objects
    
    bond_list : list of lists of bond objects
        
    Notes
    -----
    bond_grps and bond_list are expected to be ordered in the same way, i.e. in pseudocode
    bond_grps = [bond_group(('Al','Al'),EAM),bond_group(('Ni','Ni'),EAM),...]
    bond_list = [[('Al','Al') list with bond objects for each atom in the original cell one object],[('Ni','Ni') list with ...]]
    
    """
    
    def _get_pair(elements):
        return (elements[0],elements[1])
    
    num_atoms = len(bond_list[0])    
    iso_dens_atom = np.zeros(num_atoms,dtype=float)
    pairs = [_get_pair(v.bond_species) for v in bond_grps]
    
    #iso contribution - looping over pairs
    num_bond_types = len(bond_grps)
    for i in range(num_bond_types):
        pair,bond_grp = pairs[i],bond_grps[i]
        iso_dens_atom += iso_contribution_bondstyle(pair,bond_list[i],bond_grp.rhos["iso"])
    
    iso_dens_atom = np.array(iso_dens_atom)
    #ani contribution - not implemented
    dens_atom = iso_dens_atom
    
    ###calc embedding energy - expansion with rho    
    e_emb = np.zeros(num_atoms)
    elements = set(crys_elements)
        
    rho = _rho_Fourier(dens_atom,len(bond_grp.dict_parameters["emb"][next(iter(elements))]["k"])) #this only works properly is all the element have the same number of k values
    for ele in elements: #calculate element specific embedding energies for each atom
        #select density values relevant for the embeddign energy of the current element
        idx = np.where(np.array(crys_elements)==ele)[0]
        e_emb[idx] = _emb_energy_damped(rho[:,idx],np.array(bond_grp.dict_parameters["emb"][ele]["k"]),dens_atom[idx])
    return e_emb

def _pair_energy_bondstyle(r,p,r_cut=None):
    cosr = np.array([np.cos(vi*r) for vi in range(len((p)))])
    if r_cut is not None:
        return np.dot(p,cosr)*damping(r-r_cut)
    else:
        return np.dot(p,cosr)

def calculate_pair_energy_bondstyle(bond_grps,bond_list,r_cut=None):
    
    num_atoms = len(bond_list[0])
    num_bond_types = len(bond_grps)
    
    e_pair = np.zeros(num_atoms,dtype=float)
    #loop over bond types
    for i in range(num_bond_types):
        #get energies for all atom using the given bond type
        bond_grp = bond_grps[i]
        
        for j in range(num_atoms):
            b = bond_list[i][j]
            r = b["bonded_atoms"][0,:]-b["bonded_atoms"][1:,:]
            r = np.dot(r,b["cell"])
            r = np.linalg.norm(r,axis=1)
            pair = (b["bond_species"][0],b["bond_species"][1])
            if not pair in bond_grp.dict_parameters["pair"]:
                pair = (pair[1],pair[0])
            e_pair[j] += np.sum(_pair_energy_bondstyle(r,bond_grp.dict_parameters["pair"][pair]["k"],r_cut=r_cut))    
    return e_pair

def calculate_pair_energy_bondstyle_lj(bond_grps,bond_list,r_cut=None):
    num_atoms = len(bond_list[0])
    num_bond_types = len(bond_grps)
    
    e_pair = np.zeros(num_atoms,dtype=float)
    #loop over bond types
    
    def _pair_energy(r,sig,eps):
        r3 = (sig/r)**3
        return 4*eps*(r3**4-r3**2)
    
    for i in range(num_bond_types):
        #get energies for all atom using the given bond type
        bond_grp = bond_grps[i]
        
        for j in range(num_atoms):
            b = bond_list[i][j]
            r = b["bonded_atoms"][0,:]-b["bonded_atoms"][1:,:]
            r = np.dot(r,b["cell"])
            r = np.linalg.norm(r,axis=1)
            pair = (b["bond_species"][0],b["bond_species"][1])
            if not pair in bond_grp.dict_parameters["pair"]:
                pair = (pair[1],pair[0])
            sig = bond_grp.dict_parameters["pair"][pair]["sig"][0]
            eps = bond_grp.dict_parameters["pair"][pair]["eps"][0]
            e_pair[j] += np.sum(_pair_energy(r,sig,eps))    
    return e_pair


def calculate_pair_energy_Duff(pair_distances,pair_elements,
                          params={("Al","Al"):{"b":[1,1],"s":[1,1]}}):
    
    def _calc(r,params):
        #the pair energy for a specific element pair
        thetas = np.array([np.where(s>r,1,0) for s in params["s"]]) #how does this cutoff (s>r = 0) make sense?
        cube = np.array([(s-r)**3 for s in params["s"]])
        return np.sum(np.dot(params["b"],cube*thetas))
    
    num_atoms = len(list(pair_distances.values())[0])
    
    e_pair = np.zeros(num_atoms)
    pairs = pair_distances.keys()
    assert set(pairs) == set(pair_elements.keys()), "Assertion failed - pair_distances and pair_elements have different keys: {} and {}".format(pairs,pair_elements.keys())
    for pair in pairs:
        e_pair += np.array([_calc(r,params[pair]) for r in pair_distances[pair]])
    
    return e_pair
    
def _pair_energy(r,p,r_cut=None):
    cosr = np.array([np.cos(vi*r) for vi in range(len((p)))]) #this can be pre-calculated
    if r_cut is not None:
        return np.dot(p,cosr)*damping(r-r_cut) #damping can also be pre-calculated
    else:
        return np.dot(p,cosr)


def calculate_pair_energy(pair_distances,pair_elements,
                          params={("Al","Al"):{"k":[1,1]}},r_cut=None):
    
    num_atoms = len(list(pair_distances.values())[0])
    
    e_pair = np.zeros(num_atoms)
    pairs = pair_distances.keys()
    assert set(pairs) == set(pair_elements.keys()), "Assertion failed - pair_distances and pair_elements have different keys: {} and {}".format(pairs,pair_elements.keys())
    for pair in pairs:
        e_pair += np.array([np.sum(_pair_energy(r,params[pair]["k"],r_cut=r_cut)) for r in pair_distances[pair]])
    
    return e_pair

def _pair_energy_fast(cosr,damping_r,p):
    return np.dot(p,cosr)*damping_r #damping can also be pre-calculated
    


def calculate_pair_energy_fast(e_pair,all_cosr,all_dampingr,pair_elements,
                          params={("Al","Al"):{"k":[1,1]}}):
    
    #cosr = np.array([np.cos(vi*r) for vi in range(len((p)))]) #this can be pre-calculated
    #damping_r = damping(r-r_cut)
    
    pairs = all_cosr.keys()
    #assert set(pairs) == set(pair_elements.keys()), "Assertion failed - pair_distances and pair_elements have different keys: {} and {}".format(pairs,pair_elements.keys())
    for pair in pairs:
        e_pair += np.array([np.sum(_pair_energy_fast(all_cosr[pair][v],all_dampingr[pair][v],params[pair]["k"])) for v in range(len(all_cosr[pair]))])
    
    return e_pair

def calculate_pair_forces(pair_params,elements,psipr,psir,coskr,ksinkr,
                                  dr,sup_idx,fpr_neighs_idx_orders):
    num_atoms = len(elements)

    W = [np.array([pair_params[tuple(sorted([elements[n],elements[sup_idx[v]]]))]["k"] for v in fpr_neighs_idx_orders[n]]) 
         for n in range(num_atoms)] #this is a bit messy - makes sure that the parameters are called using the correct element order

    F_n = np.zeros((num_atoms,3),dtype=np.float64)
    F_i = np.zeros((num_atoms,3),dtype=np.float64)
    for n in range(num_atoms):
        num_neigh = psipr[n].shape[0]
        
        L_n = psipr[n] * np.einsum('ij,ij->i',W[n],coskr[n])
        R_n = psir[n] * np.einsum('ij,ij->i',W[n],ksinkr[n])
        
        F_n[n] += np.sum(dr[n]*((L_n-R_n).reshape((num_neigh,1))), axis=0)
        
        idx = fpr_neighs_idx_orders[n]
        L_i = np.array([psipr[n][v] * np.dot(W[n][v,:], coskr[n][v,:]) for v,vidx in enumerate(idx)])
        R_i = np.array([psir[n][v] * np.dot(W[n][v,:], ksinkr[n][v,:]) for v,vidx in enumerate(idx)])
        F_i[n] += np.sum( dr[n]*((L_i-R_i).reshape((num_neigh,1))), axis=0 )
    """
    F_n = np.array([np.sum(dr[n]*((psipr[n] * np.einsum('ij,ij->i',W[n],coskr[n]) - psir[n] * np.einsum('ij,ij->i',W[n],ksinkr[n])).reshape((psipr[n].shape[0],1))), axis=0)\
                   for n in range(num_atoms)])
    
    F_i = np.array([np.sum( dr[n]*((\
                                    np.array([psipr[n][sup_idx[vidx]] * np.dot(W[n][sup_idx[vidx],:], coskr[n][sup_idx[vidx],:]) for vidx in fpr_neighs_idx_orders[n]])\
                                    - np.array([psir[n][sup_idx[vidx]] * np.dot(W[n][sup_idx[vidx],:], ksinkr[n][sup_idx[vidx],:]) for vidx in fpr_neighs_idx_orders[n]])).reshape((psipr[n].shape[0],1))), axis=0 ) for n in range(num_atoms)])
    """
    return F_n + F_i

def calculate_embedding_forces(emb_params,elements,fpr,fpr_neighs,fpr_neighs_idx_orders,
                                   psiprho,psirho,coskrho,ksinkrho,sup_idx):
    """
    Parameters
    ----------
    psiprho :
        phi^prime(rho_n)
    psirho :
        psi(rho_n)
    coskrho :
        cos(k rho_n)
    ksinkrho :
        k sin(k rho)
    fpr :
        f^prime_alpha(r_ni)
    ultra2super_map : dict of int keys and values
        maps the index of an (original supercell) ultracell atom from the ultra cell indexing to the 
        supercell indexing
    """

    num_atoms = len(fpr)
    W = np.array([emb_params[ele]["k"] for ele in elements])      
    F_n =  fpr * (psiprho*np.einsum('ij,ij->i',W,coskrho) - psirho*np.einsum('ij,ij->i',W,ksinkrho)).reshape((num_atoms,1))

    F_i = np.zeros((num_atoms,3))
    for n in range(num_atoms):
        idx = fpr_neighs_idx_orders[n]
        L = np.array([psiprho[sup_idx[vidx]] * np.dot(W[sup_idx[vidx],:], coskrho[sup_idx[vidx],:]) for vidx in idx])
        R = np.array([psirho[sup_idx[vidx]] * np.dot(W[sup_idx[vidx],:], ksinkrho[sup_idx[vidx],:]) for vidx in idx])
        F_i[n] += np.sum( fpr_neighs[n]*(L-R).reshape((len(idx),1)), axis=0 )
    """
    F_i = np.array([np.sum(fpr_neighs[n]*(\
                                          np.array([psiprho[sup_idx[vidx]] * np.dot(W[sup_idx[vidx],:], coskrho[sup_idx[vidx],:]) for vidx in fpr_neighs_idx_orders[n]])                                         \
                                          - np.array([psirho[sup_idx[vidx]] * np.dot(W[sup_idx[vidx],:], ksinkrho[sup_idx[vidx],:]) for vidx in fpr_neighs_idx_orders[n]]) ).reshape((len(fpr_neighs_idx_orders[n]),1)), axis=0 ) for n in range(num_atoms)])
    """
    #print("d/dr_n E_emb,i -> F {}".format(F_i))
    return F_n+F_i

def _iso_contribution_edensity(num_atoms,ordered_pairs,pair_distances,pair_indices,sup_ele,elements,rhos_iso):
    edensity = np.zeros(num_atoms,dtype=np.float64)
    edensity_neighs_idx_orders = [[]]*num_atoms
    for n in range(num_atoms):
        #tmp_fpr_neighs = []
        tmp_edensity_neighs_order = []
        for pair in ordered_pairs:
            for i in range(len(pair_distances[pair][n])):
                idx = pair_indices[pair][n][i+1]
                neigh_ele = sup_ele[idx]
                curr_ele = elements[n]
                r_ni = pair_distances[pair][n][i]

                #isotropic density contribution
                edensity[n] += rhos_iso[neigh_ele](r_ni) ##contribution d/d_rn E_emb(rho_n)
                tmp_edensity_neighs_order.append(idx)
        #self.fr_neighs[n] = np.array(tmp_fpr_neighs,dtype=np.float64)
        edensity_neighs_idx_orders[n] = np.array(tmp_edensity_neighs_order,dtype=int)
    return edensity, edensity_neighs_idx_orders

def _calculate_embedding_energies(emb_params,elements,edensity,edensity_neighs_idx_orders,
                                 psirho,coskrho,num_atoms):

    #print("\nCalculating embedding forces...")
    W = np.array([emb_params[ele]["k"] for ele in elements])      
    E_emb = psirho*np.einsum('ij,ij->i',W,coskrho)
    return E_emb

def _calculate_pair_energies(pair_params,elements,psir,coskr,sup_idx,edensity_neighs_idx_orders,num_atoms):

    W = [np.array([pair_params[tuple(sorted([elements[n],elements[sup_idx[v]]]))]["k"] for v in edensity_neighs_idx_orders[n]]) 
         for n in range(num_atoms)] #this is a bit messy - makes sure that the parameters are called using the correct element order
    """
    E_pair = np.zeros(num_atoms,dtype=np.float64)

    for n in range(num_atoms):
        num_neigh = psir[n].shape[0]
        M = psir[n] * np.einsum('ij,ij->i',W[n],coskr[n])
        E_pair[n] += np.sum(M)
    """
    E_pair = np.array([np.sum(psir[n] * np.einsum('ij,ij->i',W[n],coskr[n])) for n in range(num_atoms)])
    return E_pair

class structure_template(translating_params):
    """Template for structures duh...
    
    Parameters
    ----------
    rho_paths : list of str
        list of paths to relevant densities obtained from RVM regression
    
    """
    
    def __init__(self,fpositions,elements,cell,energy,rho_paths,forces=None,stress=None):
        
        #input
        self.fpositions = fpositions
        self.elements = elements
        self.sorted_elements = np.array(sorted(list(set(self.elements))),dtype=str)
        self.edensity = None
        self.cell = cell
        self.invcell = np.linalg.inv(cell)
        self.num_atoms = len(self.fpositions)
        self.energy = energy/float(self.num_atoms) #ENERGY PER ATOM
        self.forces = forces
        self.stress = stress
        
        if self.fpositions.shape[0]==3:
            self.fpositions = fpositions.T
        self.positions = np.dot(self.fpositions,cell)
                    
        #geometries
        self.pair_distances = None
        self.pair_vectors = None #normalized distance vectors
        self.pair_elements = None
        self.triplet_distances = None
        self.triplet_elements = None
        self.r_cut = None
        self.aniso = None
        self.sup_pos = None #supercell positions
        self.sup_fpos = None #supercell fractional positions
        self.sup_ele = None #supercell elements
        self.sup_idx_map = None #map which index belongs to which fractional coordinate of the supercell
        
        #a structure is made up of bonds...
        self.bond_list = None
        self.bond_grps = None
        
        #EAM
        self.all_cosr = None
        self.all_damping = None
        self.e_pair = np.zeros(len(self.fpositions))
        self.e_emb = np.zeros(len(self.fpositions))
        self.all_params = None
        self._num_all_params = None
        self._params_info = None
        self.maps = None
        self.rhos = {"iso":{},"aniso":{}}
        for rho_path in rho_paths:
            tmp_rhos = load_rhos(rho_path)
            self.rhos["iso"].update(tmp_rhos["iso"])
            self.rhos["aniso"].update(tmp_rhos["aniso"])
            print("Loaded rhos from {}...".format(rho_path))
        #EAM - force related for fast calculation during evo
        self.rhos_prime = None
        self.psiprho = None
        self.psirho = None
        self.coskrho = None
        self.ksinkrho = None
        self.fpr = None
        self.fpr_neighs = None
        self.fpr_neighs_idx_orders = None #since distances and such is split into element combinations this variable stores in which order the neighbors of an atom were accessed
        self.ordered_pairs = None
        self.psipr = None
        self.psir = None
        self.coskr = None
        self.ksinkr = None
        self.dr = None
    
    def _calc_bonds(self):
        """Setting up bonds the bond list and the corresponding bond groups.
        
        Notes
        -----
        
        The code may be a bit confusing. This is because pair_* are generated
        with parameter optimization using energy only in mind whereas bond_list 
        and bond_grps are useful for geometry optimization and force / stress
        calculations.
        
        """
        PD = PairDistances(self.fpositions,self.elements,self.cell,aniso=self.aniso)

        self.pair_distances, self.pair_elements, self.pair_vectors, self.triplet_distances,self.triplet_elements = PD(r_cut=self.r_cut)
        self.pair_indices = PD.pair_indices
        #print("pair_indices {}".format(self.pair_indices))
        self.sup_pos = PD.sup_pos #supercell
        self.sup_fpos = PD.sup_fpos
        self.sup_ele = PD.sup_ele
        self.sup_idx = PD.sup_idx
        self.sup_idx_map = {tuple(fpos):v for v,fpos in enumerate(self.sup_fpos)}
        self.ultra2super_map = {v:self.sup_idx_map[tuple(self.fpositions[v])] for v in range(self.num_atoms)} #maps the index of an atom within the ultracell to the corresponding index of the aotm in the origin supercell
        self.ultra2super_map = {v:k for k,v in self.ultra2super_map.items()}
        #iso part
        self.bond_list = []
        self.bond_grps = []
        self.ordered_pairs = sorted(self.pair_distances.keys())
        #print("self.pair_distances {}".format(self.pair_distances))
        for pair in self.pair_distances.keys():
            num_atoms = len(self.pair_distances[pair])
            tmp_bond_list = []
            
            bond_grp = bonded_group()
            bond_grp.set_potential_form(2)
            bond_grp.set_bond_species(pair)
            bond_grp.set_parameters(self.all_params) #the same set of parameters is set for all bonded groups to avoid having to split and merge
            bond_grp.set_dict_parameters({"emb":self.emb_params,"pair":self.pair_params})
            bond_grp.set_rhos(self.rhos)
            self.bond_grps.append(bond_grp)
            for i in range(num_atoms):
                if len(self.pair_distances[pair])>0:
                    b = bond()
                    b["bond_species"] = self.sup_ele[self.pair_indices[pair][i]]
                    b["pair_distances"] = self.pair_distances[pair][i]
                    b["pair_elements"] = self.pair_elements[pair][i]
                    b["bonded_atoms"] = np.array(self.sup_fpos[self.pair_indices[pair][i]],dtype=float)
                    b["indices"] = self.pair_indices[pair][i] #includes the "center" atom of the bond in first place
                    b["Nk"] = len(b["bond_species"])
                    b["cell"] = self.cell
                    tmp_bond_list.append(b)
                    #print("bond:{}".format('\n'.join([str(k)+'->'+str(b[k]) for k in b.keys()])))
            self.bond_list.append(tmp_bond_list)
    
    def eam_setup(self,emb_params,pair_params,r_cut=None,aniso=False):
        """
        emb_params = {"Nb":{"k":np.ones(10)}}
        pair_params = {("Nb","Nb"):{"k":np.ones(10)},}
        """
        
        self.aniso = aniso
        assert r_cut is not None and isinstance(r_cut,(int,float)), "Assertion failed - expected r_cut to be of int or float type, got {} instead.".format(type(r_cut))
        self.r_cut = float(r_cut)
        
        self.emb_params = emb_params
        self.pair_params = symmetrize_params(pair_params)
        
        #params sorting
        self._elements_sorted = sorted(self.emb_params.keys())
        self._pairs_sorted = sorted(self.pair_params.keys())
        self._emb_params_sorted = sorted(self.emb_params[self._elements_sorted[0]])
        self._pair_params_sorted = sorted(self.pair_params[self._pairs_sorted[0]])
        
        self.maps = {ele: "emb_params" for ele in self._elements_sorted}
        self.maps.update({pair:"pair_params" for pair in self._pairs_sorted})
        
        self._check_input_params()
        self._map_to_all_params()
                
        assert set(self.elements).issubset(self._elements_sorted), "Assertion failed - the elements in the atomic model are not a subset of the ones specifying the potential"+\
            ": potential {}, atomic model {}".format(set(self._elements_sorted),set(self.elements))
        
        self._calc_bonds()
        
        #aniso part - not yet implemented
        
    def update_params(self,params):
        self._clear_params()
        assert len(params)==self._num_all_params, "Assertion failed - given params mismatch in number - given = {}, expected = {}".format(len(params),self._num_all_params)
        self.all_params = params
        self._map_from_all_params()
        for item in self.bond_grps:
            item.update_parameters(params)
                                
    def get_structure_forces(self,initialize=True):
        """Calculates analytical forces for the EAM potential.
        
        """
                            
        # ===== Embedding =====        
        if initialize: #if true then pre-calculated variables are unknown
            # variable names for storage for embedding: phi^prime(rho_n) = psiprho, psi(rho_n) = psirho, cos(k rho_n) = coskrho, k sin(k rho) = ksinkrho, f^prime_alpha(r_ni) = fpr        
            if self.rhos_prime is None:
                # check whether rho derivatives exist
                r_range = np.linspace(0,self.r_cut,10000)
                self.rhos_prime = {"iso":{k:None} for k in self.rhos["iso"].keys()}
                for key in self.rhos["iso"].keys():
                    self.rhos_prime["iso"][key] = _derivative_of_spline(r_range,self.rhos["iso"][key],lb=0,ub=self.r_cut)
            
            # check whether self.edensity exists, if not calculate
            if self.edensity is None:
                e_emb,self.edensity = calculate_embedding_energy(self.pair_distances,self.pair_elements,self.elements,
                                                   self.rhos,params=self.emb_params,triplet_distances=self.triplet_distances,
                                                   triplet_elements=self.triplet_elements)
            
            self.psiprho = damping_prime(self.edensity)
            self.psirho = damping(self.edensity)
            
            self.fpr = np.zeros((self.num_atoms,3))
            self.fpr_neighs = [[]]*self.num_atoms #sizes of respective arrays depend upon number of neighbors of atom
            self.fpr_neighs_idx_orders = [[]]*self.num_atoms
            for n in range(self.num_atoms):
                tmp_fpr_neighs = []
                tmp_fpr_neighs_order = []
                for pair in self.ordered_pairs:
                    for i in range(len(self.pair_distances[pair][n])):
                        idx = self.pair_indices[pair][n][i+1]
                        neigh_ele = self.sup_ele[idx]
                        curr_ele = self.elements[n]
                        r_ni = self.pair_distances[pair][n][i]
                        dr_ni = self.pair_vectors[pair][n][0][i]
                        self.fpr[n,:] += self.rhos_prime["iso"][neigh_ele](r_ni)*dr_ni ##contribution d/d_rn E_emb(rho_n)
                        tmp_fpr_neighs.append(self.rhos_prime["iso"][curr_ele](r_ni)*dr_ni) #contribution d/d_rn E_emb(rho_i)
                        tmp_fpr_neighs_order.append(idx)
                self.fpr_neighs[n] = np.array(tmp_fpr_neighs,dtype=np.float64)
                self.fpr_neighs_idx_orders[n] = np.array(tmp_fpr_neighs_order,dtype=int)
            #print("self.fpr {}".format(self.fpr))
            #print("self.fpr_neighs {}".format(self.fpr_neighs))
            #print("self.fpr_neighs_idx_orders {}".format(self.fpr_neighs_idx_orders))
            
            max_k = [len(v["k"]) for v in self.emb_params.values()]
            num_max_k = len(max_k)
            max_k = max(max_k)
            if num_max_k:                 
                warnings.warn("Noticed that 'k' was specified for the embedding energy for multiple elements. Using only the largest k = {}.".format(max_k))            
            #print("max_k {}".format(max_k))
            #print("self.edensity {}".format(self.edensity))
            self.coskrho = np.array([[np.cos(k*e) for k in range(max_k)] for e in self.edensity],dtype=np.float64)
            #self.coskrho = np.array([[np.cos(np.arange(max_k)*e)] for e in self.edensity],dtype=np.float64)
            self.ksinkrho = np.array([[k*np.sin(k*e) for k in range(max_k)] for e in self.edensity],dtype=np.float64)
            #self.ksinkrho = np.array([[np.arange(max_k)*np.sin(np.arange(max_k)*e)] for e in self.edensity],dtype=np.float64)
                        
        # forces for each atom due to embedding
        F_emb = calculate_embedding_forces(self.emb_params,self.elements,self.fpr,self.fpr_neighs,self.fpr_neighs_idx_orders,
                                           self.psiprho,self.psirho,self.coskrho,self.ksinkrho,self.sup_idx)
        #print("F_emb {}".format(F_emb))
        # ===== Pair =====
        # variable names for storage for embedding: psi^prime(r_ni-r_cut) = psipr, psi(r_ni-r_cut) = psir, cos(k*r_ni) = coskr, k sin(k r_ni) = ksinkr
        
        if initialize: #if true then pre-calculated variables are unknown
            max_k = [len(v["k"]) for v in self.pair_params.values()]
            num_max_k = len(max_k)
            max_k = max(max_k)
            if num_max_k:                 
                warnings.warn("Noticed that 'k' was specified for the pair energy for multiple elements. Using only the largest k = {}.".format(max_k))            
            #print("max_k {}".format(max_k))
            
            self.psipr = [[]]*self.num_atoms #each atom has an (N_n,) array 
            self.psir = [[]]*self.num_atoms #each atom has an (N_n,) array
            self.coskr = [[]]*self.num_atoms #each atom has an (max_k,N_n) array
            self.ksinkr = [[]]*self.num_atoms #each atom has an (max_k,N_n) array
            self.dr = [[]]*self.num_atoms #each atom has an (N_n,3) array
            
            for n in range(self.num_atoms):
                tmp_r = []
                tmp_dr = []
                for pair in self.ordered_pairs:
                    for i in range(len(self.pair_distances[pair][n])):
                        idx = self.pair_indices[pair][n][i+1]
                        neigh_ele = self.sup_ele[idx]
                        curr_ele = self.elements[n]
                        r_ni = self.pair_distances[pair][n][i]
                        dr_ni = self.pair_vectors[pair][n][0][i]
                        tmp_dr.append(dr_ni)
                        tmp_r.append(r_ni)
                #print("tmp_r {}".format(tmp_r))
                self.psipr[n] = damping_prime(np.array(tmp_r,dtype=np.float64)-self.r_cut)
                self.psir[n] = damping(np.array(tmp_r,dtype=np.float64)-self.r_cut)
                #for k in range(max_k):
                #    print("np.cos(k*np.array(tmp_r)) {}".format(np.cos(k*np.array(tmp_r))))
                #print("self.coskr {}".format(self.coskr))
                self.coskr[n] = np.array([[np.cos(k*vr) for k in range(max_k)] for vr in tmp_r],dtype=np.float64)
                self.ksinkr[n] = np.array([[k*np.sin(k*vr) for k in range(max_k)] for  vr in tmp_r],dtype=np.float64)
                self.dr[n] = np.array(tmp_dr,dtype=np.float64)
                
        # forces for each atom due to pair interacton
        F_pair = calculate_pair_forces(self.pair_params,self.elements,self.psipr,self.psir,self.coskr,
                                       self.ksinkr,self.dr,self.sup_idx,self.fpr_neighs_idx_orders)
        #print("F_pair {}".format(F_pair))
        #return F_emb + F_pair
        return F_emb+F_pair
    
    def get_structure_energy(self):
        """Method to calculate the energy.
        
        This method is not using the bond style formalism. This method is optimal for
        parameter optimzation, since no modification of the atomic models is expected.
        
        """
        
        #calculate the electron density here once
        self.e_emb[:] = 0
        if self.edensity is None:
            e_emb,self.edensity = calculate_embedding_energy(self.pair_distances,self.pair_elements,self.elements,
                                               self.rhos,params=self.emb_params,triplet_distances=self.triplet_distances,
                                               triplet_elements=self.triplet_elements)
        else:
            e_emb = calculate_embedding_energy(self.pair_distances,self.pair_elements,self.elements,
                                               self.rhos,params=self.emb_params,triplet_distances=self.triplet_distances,
                                               triplet_elements=self.triplet_elements,edensity=self.edensity)
        self.e_emb[:] = e_emb
        
        
        if self.all_cosr is None and self.all_damping is None:
            pairs = self.pair_distances.keys()
            self.all_cosr = {pair: [ np.array([np.cos(vi*r) for vi in range(len((self.pair_params[pair]["k"])))]) for r in self.pair_distances[pair]] for pair in pairs}
            self.all_damping = {pair: [ damping(r-self.r_cut) for r in self.pair_distances[pair]] for pair in pairs}
            num_atoms = len(list(self.all_cosr.values())[0])
                
        self.e_pair[:] = 0
        self.e_pair[:] = calculate_pair_energy_fast(self.e_pair,self.all_cosr,self.all_damping,self.pair_elements,params=self.pair_params)
        return (np.sum(e_emb)+.5*np.sum(self.e_pair))/float(self.num_atoms)
    
    def get_structure_energy_new(self,initialize=True):
        """
        
        Returns
        -------
        E - float
            average potential energy per atom
        """
        
        # ===== Embedding =====
        if initialize:
            
            self.edensity = np.zeros(self.num_atoms,dtype=np.float64)
            #self.fr_neighs = [[]]*self.num_atoms
            self.edensity_neighs_idx_orders = [[]]*self.num_atoms
            
            self.edensity, self.edensity_neighs_idx_orders = _iso_contribution_edensity(self.num_atoms,self.ordered_pairs,
                                                                                        self.pair_distances,self.pair_indices,
                                                                                        self.sup_ele,self.elements,self.rhos["iso"])
            
            if self.psirho is None:
                self.psirho = damping(self.edensity)
            if self.coskrho is None:
                max_k = [len(v["k"]) for v in self.emb_params.values()]
                num_max_k = len(max_k)
                max_k = max(max_k)
                if num_max_k:                 
                    warnings.warn("Noticed that 'k' was specified for the embedding energy for multiple elements. Using only the largest k = {}.".format(max_k))            
                
                self.coskrho = np.array([[np.cos(k*e) for k in range(max_k)] for e in self.edensity],dtype=np.float64)
                
        E_emb = _calculate_embedding_energies(self.emb_params,self.elements,self.edensity,self.edensity_neighs_idx_orders,
                                              self.psirho,self.coskrho,self.num_atoms)
        # ===== Pair =====
        if initialize:
            if any([self.psir is None, self.coskr is None, self.ksinkr is None]):
                warnings.warn("Found that one or more of the following self.psir, self.coskr, self.ksinkr were not previously calculated. Calculating all of them now now!")
                self.psir = [[]]*self.num_atoms #each atom has an (N_n,) array
                self.coskr = [[]]*self.num_atoms #each atom has an (max_k,N_n) array
                self.ksinkr = [[]]*self.num_atoms #each atom has an (max_k,N_n) array
            
                for n in range(self.num_atoms):
                    tmp_r = []
                    for pair in self.ordered_pairs:
                        for i in range(len(self.pair_distances[pair][n])):
                            idx = self.pair_indices[pair][n][i+1]
                            neigh_ele = self.sup_ele[idx]
                            curr_ele = self.elements[n]
                            r_ni = self.pair_distances[pair][n][i]
                            dr_ni = self.pair_vectors[pair][n][0][i]
                            tmp_r.append(r_ni)
                                        
                    self.psir[n] = damping(np.array(tmp_r,dtype=np.float64)-self.r_cut)
                    self.coskr[n] = np.array([[np.cos(k*vr) for k in range(max_k)] for vr in tmp_r],dtype=np.float64)
                    self.ksinkr[n] = np.array([[k*np.sin(k*vr) for k in range(max_k)] for  vr in tmp_r],dtype=np.float64)
        
        E_pair = _calculate_pair_energies(self.pair_params,self.elements,self.psir,self.coskr,self.sup_idx,self.edensity_neighs_idx_orders,self.num_atoms)
        #print("E = E_emb + 1/2 E_pair = {} + {} = {}".format(E_emb,E_pair,E_emb+.5*E_pair))
        return np.mean(E_emb + .5*E_pair)
    
    def get_structure_energy_bond_style(self,eam=True,cumulative=True):
        """Method to calculate the energy.
        
        This method is using the bond style formalism, particularly useful for force and 
        geometry optimizations where small pertubations are made to the atomic model.
        Using the indices determined during the setup distances are recalculated to get
        energies.        
        
        """
        if eam:
            e_emb = calculate_embedding_energy_bondstyle(self.bond_grps,self.bond_list,self.elements)
            e_pair = calculate_pair_energy_bondstyle(self.bond_grps,self.bond_list,r_cut=self.r_cut)
        else:
            e_emb = np.zeros(2)
            e_pair = calculate_pair_energy_bondstyle_lj(self.bond_grps,self.bond_list,r_cut=self.r_cut)
        if cumulative:
            return (np.sum(e_emb)+.5*np.sum(e_pair))/float(self.num_atoms)
        else:
            return e_emb+.5*e_pair
        
    def get_forces_ase(self,delta=1e-8,eam=True):
        """Calculates numerical forces
        
        Inpsiration:
        https://wiki.fysik.dtu.dk/ase/epydoc/ase.calculators.test-pysrc.html#numeric_force
        
        Parameters
        ----------
        delta : float
            real space displacement of an atom in Angstrom.
            
        Notes
        -----
        Though this method can be used for evolution note that this is ridiculously bad scaling. 
        Since the distances have to be recomputed again and again.
        """
        
        num_atoms = len(self.bond_list[0])
        dr = np.array([[delta,0,0],[0,delta,0],[0,0,delta]],dtype=np.float64)
        fdr = np.dot(dr,self.invcell)
        sup_fpos = copy.deepcopy(self.sup_fpos)
        num_bondtypes = len(self.bond_grps)
        num_bonds = len(self.bond_list[0])
        
        virial_tensor = np.zeros((3,3),dtype=np.float64)        
        forces = np.zeros((num_atoms,3))
        for i,fpos in enumerate(self.fpositions):
            pos_idx = self.sup_idx_map[tuple(fpos)]
            sup_fpos_ori = copy.deepcopy(sup_fpos[pos_idx])
            
            for d in range(3):
                #sup_fpos[pos_idx] += fdr[d]
                
                for j in range(num_bondtypes):
                    #loop over all bonds
                    for k in range(num_bonds):
                        #bond = self.bond_list[j][k]
                        atom_present, ix = self.bond_list[j][k].has(pos_idx)
                        if atom_present: self.bond_list[j][k]["bonded_atoms"][ix] = sup_fpos_ori + fdr[d] #sup_fpos[pos_idx]
                
                E_plus = self.get_structure_energy_bond_style(eam=eam,cumulative=True)
                
                for j in range(num_bondtypes):
                    #loop over all bonds
                    for k in range(num_bonds):
                        #bond = self.bond_list[j][k]
                        atom_present, ix = self.bond_list[j][k].has(pos_idx)
                        if atom_present: self.bond_list[j][k]["bonded_atoms"][ix] = sup_fpos_ori - fdr[d] #sup_fpos[pos_idx]
                
                E_minus = self.get_structure_energy_bond_style(eam=eam,cumulative=True)
                
                forces[i,d] = (E_minus - E_plus) / (2 * delta)
        return forces
                
    def get_stress_ase(self,delta=1e-3,eam=True):
        """Calculates numerical stress.
        
        Inspiration:
        https://wiki.fysik.dtu.dk/ase/epydoc/ase.calculators.calculator-pysrc.html#Calculator.calculate_numerical_stress
        
        The numerical stress is calculated by squeezing and stretching
        the cell along its base vectors.
        
        Parameters
        ----------
        delta : float
            Modifies the cell along a given axis by multiplying with 1+delta.
        """
        dr = np.array([[delta,0,0],[0,delta,0],[0,0,delta]],dtype=np.float64)
        fdr = np.dot(dr,self.invcell)
        sup_fpos = copy.deepcopy(self.sup_fpos)
        num_bondtypes = len(self.bond_grps)
        num_bonds = len(self.bond_list[0])
        
        stress = np.zeros((3,3),dtype=np.float64)
        Ep1 = 0#np.zeros(num_bonds,dtype=np.float64) #energy moving atom in + direction
        Em1 = 0#np.zeros(num_bonds,dtype=np.float64) #energy moving atom in - direction
        V = np.linalg.det(self.cell)
        #print("V {}".format(V))
        #print("original cell {}".format(self.cell))
        
        def _update_cell(x):
            new_cell = np.dot(self.cell,x)
            #print("new_cell {}".format(new_cell))
            #update bonds
            for i0 in range(len(self.bond_list)):
                for i1 in range(len(self.bond_list[i0])):
                    self.bond_list[i0][i1]["cell"] = new_cell #ase scales the atom positions, which we dont have to do because of fractional coordinates CHECK THAT
        
        for i in range(3):
            x = np.eye(3)
            x[i,i] += delta
            _update_cell(x)
            E_plus = self.get_structure_energy_bond_style(eam=eam,cumulative=True)
            
            x[i,i] = 1.-delta
            _update_cell(x)
            
            E_minus = self.get_structure_energy_bond_style(eam=eam,cumulative=True)
            stress[i,i] = (E_plus - E_minus) / (2*delta*V)
            
            x[i,i] = 1
            j = (i+1)%3
            x[i,j] = delta
            x[j,i] = delta
            _update_cell(x)
            
            E_plus = self.get_structure_energy_bond_style(eam=eam,cumulative=True)
            
            x[i,j] = -delta
            x[j,i] = -delta
            _update_cell(x)
            
            E_minus = self.get_structure_energy_bond_style(eam=eam,cumulative=True)
            stress[i,j] = (E_plus - E_minus) / (4*delta*V)
            stress[j,i] = stress[i,j]
            
        _update_cell(np.eye(3))
        return stress
    
    def get_forces_and_stress(self,delta=1e-9,eam=True):
        """Calculates numerical forces and stress.
        
        Numerical forces are calculated by displacing an atom by +\Delta r and -\Delta r and calculate the 
        corresponding energy each. Hence the position of that atom in the supercell needs to be changed.
        Then the energy can be computed looping over all bond types and bonds. 
        
        XXXXX bond["bonded_atoms"] will need to be updated after modifying the supercell
        
        """
        dr = np.array([[delta,0,0],[0,delta,0],[0,0,delta]],dtype=np.float64)
        fdr = np.dot(dr,self.invcell)
        sup_fpos = copy.deepcopy(self.sup_fpos)
        num_bondtypes = len(self.bond_grps)
        num_bonds = len(self.bond_list[0])
        forces = np.zeros((num_bonds,3),dtype=np.float64)
        virial_tensor = np.zeros((3,3),dtype=np.float64)
        Ep1 = 0#np.zeros(num_bonds,dtype=np.float64) #energy moving atom in + direction
        Em1 = 0#np.zeros(num_bonds,dtype=np.float64) #energy moving atom in - direction
        V = np.linalg.det(self.cell)
        #print("V {}".format(V))
                          
        #loop over all atoms -> select current atom
        for i,fpos in enumerate(self.fpositions):
            #print("i {} fpos {}".format(i,fpos))
            #select dimension
            F = np.zeros((num_bonds,3),dtype=np.float64) #force
            pos_idx = self.sup_idx_map[tuple(fpos)]
            sup_fpos_ori = copy.deepcopy(sup_fpos[pos_idx])
            for d in range(3):
                Ep1 = 0 #[:] = 0
                Em1 = 0 #[:] = 0
                
                #======== change supercell in + direction
                sup_fpos[pos_idx] += fdr[d]
                
                #loop over all bond types
                for j in range(num_bondtypes):

                    #loop over all bonds
                    for k in range(num_bonds):
                        #bond = self.bond_list[j][k]
                        atom_present, ix = self.bond_list[j][k].has(pos_idx)
                        if atom_present: self.bond_list[j][k]["bonded_atoms"][ix] = sup_fpos[pos_idx]
                  
                #calculate energy
                Ep1 = self.get_structure_energy_bond_style(eam=eam,cumulative=True) #CHECK: were the bonds REALLY updated?

                #======== change supercell in - direction
                sup_fpos[pos_idx] = sup_fpos_ori - fdr[d,:]
                
                #loop over all bond types
                for j in range(num_bondtypes):

                    #loop over all bonds
                    for k in range(num_bonds):
                        #bond = self.bond_list[j][k]
                        atom_present, ix = self.bond_list[j][k].has(pos_idx)
                        if atom_present: self.bond_list[j][k]["bonded_atoms"][ix] = sup_fpos[pos_idx]

                #calculate energy
                Em1 = self.get_structure_energy_bond_style(eam=eam,cumulative=True) #CHECK: were the bonds REALLY updated?
                
                #======== update force and reset
                F[i,d] = -(Ep1-Em1)/(2*delta*V)
                
                sup_fpos[pos_idx] = sup_fpos_ori #should be the original position (floating point error a problem?)
                
                #restore bonds to original
                for j in range(num_bondtypes): #loop over all bond types
                    #loop over all bonds
                    for k in range(num_bonds):
                        #bond = self.bond_list[j][k]
                        atom_present, ix = self.bond_list[j][k].has(pos_idx)
                        if atom_present: self.bond_list[j][k]["bonded_atoms"][ix] = sup_fpos_ori
            #store force on current atom
            forces += F
        for j in range(num_bonds):
            virial_tensor += np.outer(self.positions[j],forces[j])
        return forces, virial_tensor