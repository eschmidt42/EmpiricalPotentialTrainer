import numpy as np
from scipy import interpolate
from scipy import spatial
import itertools, sys, warnings, time, timeit
from fitelectrondensity import misc
import fortran.interface as f90
import parsers
import copy
#import memory_profiler

def wrapped_polynomial(k,r_s):
    # polynomial basis set
    def poly(x):
        return (x/r_s)**k
    return poly

def wrapped_cos(k,r_s,usefortran=False):
    if usefortran:
        def cos_fun(r):
            return np.array([f90.wrapped_cos(k,r_s,v) for v in r])
    else:
        # for r
        f = np.pi /float(r_s)
        def cos_fun(r):
            return np.cos(f*k*r)
    
    return cos_fun

def wrapped_cos_q(k,r_s):
    # for r
    f = np.pi /float(r_s)
    def cos_fun_q(r):
        return np.cos(f*k*r)
    return cos_fun_q

def wrapped_sin_q(k,r_s):
    # for r
    f = np.pi /float(r_s)
    def sin_fun_q(r):
        return np.sin(f*k*r)
    return sin_fun_q

def wrapped_sin(k,r_s):
    # for r
    f = np.pi /float(r_s)
    def sin_fun(r):
        return np.sin(f*k*r)
    return sin_fun

def wrapped_cos2(k):
    # for phi
    def cos_fun(theta): # phi -> ct
        return np.cos(k*theta) # this needs to be changed to k*np.pi*ct
    return cos_fun

def wrapped_sin2(k):
    # for phi
    def sin_fun(theta): # phi -> ct
        return np.sin(k*theta) # this needs to be changed to k*np.pi*ct
    return sin_fun
"""
def wrapped_smooth(f_s,r_s):
    def smooth_fun(r):
        x = (r-r_s)/float(f_s)
        x4 = x**4
        return x4/(1.+x4)
    return smooth_fun
"""
def wrapped_smooth(f_s,r_s,kind=">",usefortran=False):
    """
    Parameters
    ----------
    kind : str or None
        ">" zero for r > r_s
        "<" zero for r < r_s
    """
    if isinstance(r_s,(tuple,list,np.ndarray)):
        def smooth_fun(r,tmp_kind=["<",">"]):
            x = (r-r_s[0])/float(f_s)
            xp = (r-r_s[1])/float(f_s)  # NEW FIX
            if kind is not None:
                if isinstance(x,np.ndarray):
                    lo = np.where(r<r_s[0])[0]
                    hi = np.where(r>r_s[0])[0]
                    if (tmp_kind[0] == ">" and len(hi)>0):
                        x[hi] = 0.                        
                    elif (tmp_kind[0] == "<" and len(lo)>0):
                        x[lo] = 0.
                    lo = np.where(r<r_s[1])[0]
                    hi = np.where(r>r_s[1])[0]
                    if (tmp_kind[1] == ">" and len(hi)>0):
                        xp[hi] = 0.                        
                    elif (tmp_kind[1] == "<" and len(lo)>0):
                        xp[lo] = 0.

                else:
                    if (tmp_kind[0] == ">" and r>r_s[0]):
                        x = 0.
                    elif (tmp_kind[0] == "<" and r<r_s[0]):
                        x = 0.
                    if (tmp_kind[1] == ">" and r>r_s[1]):
                        xp = 0.
                    elif (tmp_kind[1] == "<" and r<r_s[1]):
                        xp = 0.
                            
            x4 = x**4
            xp4 = xp**4
            return x4/(1.+x4) * xp4/(1.+xp4)
    else:
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

def iso_wrapped(fun_basis,f_s,r_s,smooth):
    
    if smooth:
        def iso_fun_smooth(r):
            return fun_basis(r)*wrapped_smooth(f_s,r_s)(r)
        return iso_fun_smooth
    else:
        def iso_fun(r):
            return fun_basis(r)
        return iso_fun
        
def ani_wrapped_like(fun_basis_r0,fun_basis_ct,f_s,r_s,smooth):
    assert isinstance(f_s,(np.ndarray,list)),'expecting an array'
    assert len(f_s)==2,'expecting an array of length two'
    if smooth:
        def ani_fun_smooth(*args):
            # f0(r)*f0(r')*g(theta)*psi_rcut(r)*psi_rcut(r')*psi_0(r)*psi_0(r')
            return fun_basis_r0(args[0])*fun_basis_r0(args[1])*fun_basis_ct(args[2])*\
                    wrapped_smooth(f_s[0],r_s)(args[0])*wrapped_smooth(f_s[0],r_s)(args[1])*\
                    wrapped_smooth(f_s[1],0.0,kind="<")(args[0])*wrapped_smooth(f_s[1],0.0,kind="<")(args[1])
                    
            
        return ani_fun_smooth
    else:
        def ani_fun(*args):
            return fun_basis_r0(args[0])*fun_basis_r0(args[1])*fun_basis_ct(args[2])
        return ani_fun

def ani_wrapped(fun_basis_r0,fun_basis_r1,fun_basis_ct,f_s,r_s,smooth):
    assert isinstance(f_s,(np.ndarray,list)),'expecting an array'
    assert len(f_s)==2,'expecting an array of length two'
    if smooth:
        #def ani_fun_smooth(r0,r1,ct):
            #return fun_basis_r0(r0)*fun_basis_r1(r1)*fun_basis_ct(ct)*wrapped_smooth(f_s,r_s)(r0)*wrapped_smooth(f_s,r_s)(r1)
        def ani_fun_smooth(*args):
            return fun_basis_r0(args[0])*fun_basis_r1(args[1])*fun_basis_ct(args[2])*\
                    wrapped_smooth(f_s[0],r_s)(args[0])*wrapped_smooth(f_s[0],r_s)(args[1])*\
                    wrapped_smooth(f_s[1],0.0,kind="<")(args[0])*wrapped_smooth(f_s[1],0.0,kind="<")(args[1])
            
        return ani_fun_smooth
    else:
        #def ani_fun(r):
        #    return fun_basis_r0(r0)*fun_basis_r1(r1)*fun_basis_ct(ct)
        def ani_fun(*args):
            return fun_basis_r0(args[0])*fun_basis_r1(args[1])*fun_basis_ct(args[2])
        return ani_fun

def ani_wrapped_binary(fun_basis_r,fun_basis_q,f_s,r_s,smooth):
    
    if smooth:
        #def ani_fun_smooth(r0,r1,ct):
            #return fun_basis_r0(r0)*fun_basis_r1(r1)*fun_basis_ct(ct)*wrapped_smooth(f_s,r_s)(r0)*wrapped_smooth(f_s,r_s)(r1)
        def ani_fun_smooth(*args):
            return fun_basis_r(args[0])*fun_basis_q(args[1])*wrapped_smooth(f_s,r_s)(args[0])
            
        return ani_fun_smooth
    else:
        def ani_fun(r):
            return fun_basis_r(args[0])*fun_basis_q(args[1])
        return ani_fun
        
def valid_k_ani(k_ani):
	"""Checks k_ani for validity and returns processable k_ani_dict.
	
	The purpose of this function is to allow for more fine grained control over
	the anisotropic approximation, instead of specifying the k_ani for all 
	element pairs, r and theta.
	
	Parameters
	----------
	k_ani - int or dict
	
	Examples of valid k_anis:
	
	k_ani = 42 # int, default
	
	k_ani = {"like":42,"unlike":21} # functions of r,rp and theta are expanded to different degrees for like and unlike pairs
	
	k_ani = {"like":{"r":21,"theta":84},"unlike":{"r":1,"theta":2}}
	
	
	Returns
	-------
	k_ani_dict : dictionary
		* if k_ani == int then k_ani_dict = {"like":{"r":k_ani,"theta":k_ani},"unlike":{"r"...}}
		* elif k_ani == dict with "like" and "unlike" each with integer values
			then k_ani_dict = {"like":{"r":k_ani["like"],"theta":k_ani["like"]},"unlike"...}
		* elif k_ani == dict of dicts with "like" and "unlike" keys the lowest dicts have
			to have "r" and "theta" specified then k_ani_dict == k_ani
	
	"""
	
	keys = ["like","unlike"]
	
	if isinstance(k_ani,int):
		return {"like":{"r":k_ani,"theta":k_ani},"unlike":{"r":k_ani,"theta":k_ani}}
	elif isinstance(k_ani,dict) and all([k in k_ani for k in keys]):
		if all([isinstance(k_ani[k],int) for k in keys]):
			return {k:{"r":k_ani[k],"theta":k_ani[k]} for k in keys}
		elif all([isinstance(k_ani[k],dict) for k in keys]):
			if all([all([isinstance(v,int) for v in k_ani[k].values()]) for k in keys]):
				return k_ani
			else:
				raise NotImplementedError("Parse fail k_ani = {}".format(k_ani))
		else:
			raise NotImplementedError("Parse fail k_ani = {}".format(k_ani))
	else:
		raise NotImplementedError("Parse fail k_ani = {}".format(k_ani))

def get_basis(gips, k_iso=10, k_ani=10, type_iso="Fourier", type_ani="Fourier",\
              aniso=False, ani_type=None, ani_specification={}, rcut=6.,\
              num_neigh=None, smooth=True, r_smooth=6., f_smooth=1, verbose=False,\
              self_contribution=True, usefortran=False, r_scale=1., r_smooth_low=0.):
    """Generates a list of basis functions.
    
    Function accepts a list of GeneralInputParsers and creates all basis functions,
    allowing for sequential update of the design matrix structure by structure.

    Parameters
    ----------
    
    gips : list of parsers.general.GeneralInputParser instances
        Each instance contains crystals in form of parsed DFT files.

    k_iso : int, optional, default 10
        Number of 2-body basis functions for each element.

    k_ani : int, optional, default 10
        Number of 3-body basis functions for each element pair.

    aniso : boolean, optional, default False
        If True a basis will be generated with isotropic (2-body) and 
        anisotropic (3-body) terms.

    type_iso : str, optional, default "Fourier"
        Series to use for the isotropic basis.

    type_ani : str, optional, default "Fourier"
        Series to use for the anisotropic basis.

    ani_type : str or None, optional, default None

    ani_specification : dict, optional, default {}
    
    rcut : float, optional, default 6.

    num_neigh : None or int, optional, default None
    
    smooth : boolean, optional, default True
    
    r_smooth : float, optional, default 6.
    
    f_smooth : float, optional, default 1
    
    verbose : boolean, optional, default False

    self_contribution : boolean, optional, default True
        Excludes atoms with near zero distances if True.
    
    usefortran : boolean, optional, default False
    
    r_scale : float, optional, default 1.
        Scaling factor for r.    

    r_smooth_low : float, optional, default 0.
        Used as the lower tapering distance below which all
        contributions are zero. This value is used in the case
        that self_contribution = False.

    Returns
    -------

    basis : dict
        "functions" : list of basis functions
        "info" : information on the basis setup

    mapper : list of tuples
        Indicates, by index, which functions in basis["functions"] relates
        to which element and body approximation. 
        Example : [('Al', 0, 49), ('Ni', 49, 98)]


    """
    if verbose: 
        print("Generating basis...")
    
    implemented_types = ["Fourier","Fourier s+c"]
    
    # safety checks
    assert isinstance(k_iso,int), "Assertion failed - k_iso ({}) is not of int type.".format(k_iso)
    #assert isinstance(k_ani,int), "Assertion failed - k_ani ({}) is not of int type.".format(k_ani)
    k_ani_dict = valid_k_ani(k_ani)
    if verbose:
        print("k_ani {} => k_ani_dict {}".format(k_ani,k_ani_dict))
    
    assert type_iso in implemented_types, "Assertion failed - got an unexpected type_iso value ({}), expected one of: {}".format(type_iso,implemented_types)
    assert type_ani in implemented_types, "Assertion failed - got an unexpected type_ani value ({}), expected one of: {}".format(type_ani,implemented_types)
    assert isinstance(gips,list), "Assertion failed - gips must be a list of GeneralInputParser instances" 
    assert all([isinstance(_g,type(parsers.general.GeneralInputParser())) for _g in gips]), "Assertion failed - gips must be a list of GeneralInputParsers instances"
    assert all([all([_s["species"] is not None for _s in _gip]) for _gip in gips]), "Assertion failed - atom species must be given for all structures in gips"


    # generate set of all species present in all structures
    species = set([])
    pairs = set()
    for _gip in gips:
        for _s in _gip:
            tmp = _s["species"]
            species = species.union(set(_s["species"]))
            ij = itertools.product(range(len(_s["species"])),range(len(_s["species"])))
            
            _pairs = [tuple(sorted([_s["species"][i],_s["species"][j]])) \
                for i,j in ij if j>=i]
            
            pairs.update(_pairs)

    species = sorted(list(species))
    pairs = sorted(list(pairs))
    if verbose:
        print("species {}".format(species))       
        print("pairs {}".format(pairs))

    if verbose: 
        print("species present {}...".format(species))
    
    # genereate ani_selected_keys for BOP approximations
    if aniso:
        if ani_type in ["BOP","BOP-invr","BOP-r","R","BOP4atoms","BOP4atoms-ref","BOP4atoms-ref2",\
                "BOP-taper1"]:
            assert "l_range" in ani_specification,"Assertion Failed - incorrect ani_specification"
                
            if "usage" in ani_specification:
                if isinstance(ani_specification["usage"],str):
                    if ani_specification["usage"] == "everything":
                        ani_selected_keys = "everything"
                    else:
                        ani_selected_keys = [ani_specification["usage"]]
                elif isinstance(ani_specification["usage"],(list,tuple)):
                    if "all" in ani_specification["usage"]:
                        ani_selected_keys = ["all"] + sorted([v for v in ani_specification["usage"] if \
                                v!="all"])
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
    if ani_selected_keys == "everything":
        ani_selected_keys = ["all"] + [v for v in sorted(species) if v!="all"]


    if type_iso == "Fourier":
        if self_contribution:
            iso_basis = [[iso_wrapped(wrapped_cos(v,r_scale,usefortran),f_smooth,r_smooth,smooth) for v in range(k_iso)] for vspec in species]            
        else:
            iso_basis = [[iso_wrapped(wrapped_cos(v,r_scale,usefortran),f_smooth,[r_smooth_low,r_smooth],smooth) for v in range(k_iso)] for vspec in species]
            
    elif type_iso == "Fourier s+c":
        if self_contribution:
            iso_basis = [[iso_wrapped(wrapped_cos(v,r_scale,usefortran),f_smooth,[r_smooth_low,r_smooth],smooth) \
                        for v in range(k_iso)]+[iso_wrapped(wrapped_sin(v,r_scale),f_smooth,[r_smooth_low,r_smooth],smooth) \
                        for v in range(1,k_iso)]for vspec in species]            
        else:
            iso_basis = [[iso_wrapped(wrapped_cos(v,r_scale,usefortran),f_smooth,r_smooth,smooth) \
                        for v in range(k_iso)]+[iso_wrapped(wrapped_sin(v,r_scale),f_smooth,r_smooth,smooth) \
                        for v in range(1,k_iso)]for vspec in species]
    else:
        raise NotImplementedError("type_iso '{}' not understood!".format(type_iso))
    
    basis = {"functions":[]}
    mapper = []
    count = 0
    for i,tmp_basis in enumerate(iso_basis):
        basis["functions"].extend(tmp_basis)
        mapper.append(tuple([species[i],count,count+len(tmp_basis)]))
        count += len(tmp_basis)
    
    # anisotropic part
    if aniso:
        if ani_type in ["MEAM","polynomial"]: 
            
            like_pairs = [v for v in pairs if v[0]==v[1]]
            unlike_pairs = [v for v in pairs if v[0]!=v[1]]
            
            if verbose: 
                print("pairs present {}...".format(pairs))
                print("    >>> like {}\nunlike {}".format(like_pairs,unlike_pairs))
                
            k_tuples = list(itertools.product(range(k_ani_dict["unlike"]["r"]),range(k_ani_dict["unlike"]["r"]),range(k_ani_dict["unlike"]["theta"])))
            k_tuples_like = list(itertools.product(range(k_ani_dict["like"]["r"]),range(k_ani_dict["like"]["theta"])))
           

            if "f_smooth" in ani_specification and (ani_type == "MEAM" or ani_type == "polynomial"):
                smoothing = np.array([f_smooth,ani_specification["f_smooth"]])
            elif ani_type == "MEAM" or ani_type == "polynomial":
                smoothing = np.array([f_smooth,0.1])

            if type_ani == "Fourier" and ani_type == "MEAM":
                # like pairs
                ani_basis_like = [[ani_wrapped_like(wrapped_cos(v0,ani_specification["r_ani"]),\
                        wrapped_cos2(v1),smoothing,ani_specification["r_ani"],smooth) for v0,v1 in \
                        k_tuples_like] for pair in like_pairs]
                # unlike pairs
                ani_basis_unlike = [[ani_wrapped(wrapped_cos(v0,ani_specification["r_ani"]),\
                        wrapped_cos(v1,ani_specification["r_ani"]),\
                        wrapped_cos2(v2),smoothing,ani_specification["r_ani"],smooth) \
                        for v0,v1,v2 in k_tuples] for pair in unlike_pairs]
                            
                # combining both
                ani_basis = ani_basis_like + ani_basis_unlike
                pairs = like_pairs + unlike_pairs
            elif type_ani == "Fourier s+c" and ani_type == "MEAM":
                # like pairs
                ani_basis_like = [[[ani_wrapped_like(wrapped_cos(v0,ani_specification["r_ani"]),\
                        wrapped_cos2(v1),f_smooth,ani_specification["r_ani"],smooth),\
                        ani_wrapped_like(wrapped_cos(v0,ani_specification["r_ani"]),\
                        wrapped_sin2(v1),f_smooth,ani_specification["r_ani"],smooth)] for v0,v1 \
                        in k_tuples_like] for pair in like_pairs]
                ani_basis_like = [[v for v2 in _basis for v in v2] for _basis in ani_basis_like]
                
                # unlike pairs
                ani_basis_unlike = [[[ani_wrapped(wrapped_cos(v0,ani_specification["r_ani"]),\
                        wrapped_cos(v1,ani_specification["r_ani"]),wrapped_cos2(v2),f_smooth,\
                        ani_specification["r_ani"],smooth),\
                        ani_wrapped(wrapped_cos(v0,ani_specification["r_ani"]),\
                        wrapped_cos(v1,ani_specification["r_ani"]),\
                        wrapped_sin2(v2),f_smooth,ani_specification["r_ani"],smooth)] for \
                        v0,v1,v2 in k_tuples] for pair in unlike_pairs]
                ani_basis_unlike = [[v for v2 in _basis for v in v2] for _basis in ani_basis_unlike]
                
                # combining both
                ani_basis = ani_basis_like + ani_basis_unlike
            elif ani_type == "polynomial":
                if verbose:
                    print ("generating polynomial basis set...")

                # like pairs
                ani_basis_like = [[ani_wrapped_like(wrapped_polynomial(v0,ani_specification["r_ani"]),\
                        wrapped_polynomial(v1,np.pi),smoothing,\
                        ani_specification["r_ani"],smooth) for v0,v1 in k_tuples_like] for pair in like_pairs]

                # unlike pairs
                ani_basis_unlike = [[ani_wrapped(wrapped_polynomial(v0,ani_specification["r_ani"]),\
                        wrapped_polynomial(v1,ani_specification["r_ani"]),\
                        wrapped_polynomial(v2,np.pi),smoothing,ani_specification["r_ani"],smooth) \
                        for v0,v1,v2 in k_tuples] for pair in unlike_pairs]

                # combine like and unlike contributions
                ani_basis = ani_basis_like + ani_basis_unlike
                pairs = like_pairs + unlike_pairs

                if verbose:
                    print ("finished generating polynomial basis set")
            else:
                raise NotImplementedError
            if verbose:
                print("ani_basis {}".format([len(v) for v in ani_basis]))

            for i,tmp_basis in enumerate(ani_basis):
                basis["functions"].extend(tmp_basis)
                mapper.append(tuple([pairs[i],count,count+len(tmp_basis)]))
                count += len(tmp_basis)
        
        elif ani_type == "BOP4atoms-ref2":
            k_tuples = list(itertools.product(range(k_ani),range(k_ani))) # (r,q)
            
            elements = [v for v in ani_selected_keys if v!="all"] + ["all"]
            assert len(elements)>0,"Assertion Failed - No elements found with ani_specification['usage']"
            
            #elements = sorted(list(set(list(itertools.chain(*[list(v.x["ani"].keys()) for v in bonds])))))
            if verbose:
                print("elements {}".format(elements))
            if type_ani == "Fourier":
                ani_basis = []
                for i,l in enumerate(ani_specification["l_range"]):
                    ani_basis.append([[ani_wrapped_binary(wrapped_cos(v0,r_scale),wrapped_cos(v1,1),f_smooth,r_smooth,smooth) for v0,v1 in k_tuples]\
                                for el in elements])
            else:
                raise NotImplementedError
            if verbose:
                print("ani_basis {}".format([len(v) for v in ani_basis]))

            for i,l in enumerate(ani_specification["l_range"]):
                for j,tmp_basis in enumerate(ani_basis[i]):
                    basis["functions"].extend(tmp_basis)
                    mapper.append(tuple([(elements[j],"atom_q-r-ref2",l),count,count+len(tmp_basis)]))
                    count += len(tmp_basis)
            
        elif ani_type in {"BOP","BOP-invr","BOP-r","BOP4atoms","BOP4atoms-ref","BOP-taper1"}:
            
            ani_keys = ani_selected_keys
            Nk = len(ani_keys)
            if type_ani in ["Fourier","Fourier s+c"]:
                ani_basis = []
                for i,l in enumerate(ani_specification["l_range"]):
                    if type_ani == "Fourier":
                        ani_basis.append([[wrapped_cos_q(v,1.) for v in range(k_ani)] for el in ani_keys])
                    elif type_ani == "Fourier s+c":
                        ani_basis.append([[wrapped_cos_q(v,1.) for v in range(k_ani)] for el in ani_keys] +\
                                         [[wrapped_sin_q(v,1.) for v in range(k_ani)] for el in ani_keys])
                    else:
                        raise NotImplementedError

            else:
                raise NotImplementedError
            if ani_type == "BOP": 
                bop_designation = "q"
            elif ani_type == "BOP-invr":
                bop_designation = "q-1/r"
            elif ani_type == "BOP-r":
                bop_designation = "q-r"
            elif ani_type == "BOP4atoms":
                bop_designation = "atom_q-r"
            elif ani_type == "BOP4atoms-ref":
                bop_designation = "atom_q-r-ref"
            elif ani_type == "BOP-taper1":
                bop_designation = "q-r-taper1"
            
            for i,l in enumerate(ani_specification["l_range"]):
                
                for j,tmp_basis in enumerate(ani_basis[i]):
                    
                    basis["functions"].extend(tmp_basis)
                    if type_ani == "Fourier":
                        mapper.append(tuple([tuple([ani_keys[j],bop_designation,l]),count,count+len(tmp_basis)]))
                    elif type_ani == "Fourier s+c":
                        if j<Nk:
                            mapper.append(tuple([tuple([ani_keys[j],bop_designation+"-cos",l]),count,count+len(tmp_basis)]))
                        else:
                            mapper.append(tuple([tuple([ani_keys[j%Nk],bop_designation+"-sin",l]),count,count+len(tmp_basis)]))
                    count += len(tmp_basis)
        
        elif ani_type == "R":
            
            R_designation = "R"
            ani_basis = [[wrapped_cos_q(v,1.) for v in range(k_ani)] for el in ani_selected_keys]
            for j,tmp_basis in enumerate(ani_basis):
                basis["functions"].extend(tmp_basis)
                mapper.append(tuple([tuple([ani_selected_keys[j],R_designation]),count,count+len(tmp_basis)]))
                count += len(tmp_basis)

        else:
            raise NotImplementedError("any_type '{}' not understood!".format(ani_type))
    if verbose:
        print('ansio type is : {}'.format(type_ani))

    # include basis function info for fortran in get_design_matrix
    basis["info"] = {"iso":True,"aniso":aniso,"k_iso":k_iso,"k_ani":k_ani_dict,"type_iso":type_iso,
                     "type_ani":type_ani,"smooth":smooth,"r_smooth":r_smooth,"f_smooth":f_smooth,
                     "usefortran":usefortran,"self_contribution":self_contribution,"rcut":rcut,
                     "num_neigh":num_neigh,"ani_type":ani_type,"ani_specification":ani_specification} 
    
    if verbose: 
        print("finalized basis with {} terms...".format(len(basis)))
    return basis, mapper

def update_design_matrix(basis, mapper, structure, Phi, reference_densities,\
                         stochastic=("all",), seed=None, selection=("random",10.),\
                         ultra_num=None, verbose=False, usefortran=False, \
                         multithreading=True):
    """Updates the Design matrix.

    Given a predetermined basis including all elements in every structure for the 
    regression, add the section of Phi matrix which is associated with the given 
    structure.

    Parameters
    ----------

    basis : list of instances of basis functions
        obtained from fitelectrondensity.rvm.get_basis
    
    mapper : dict
        contains the information which basis functions correspond to what
        kind of contribution to the electrondensity, i.e. isotropic/anisotropic, elements,...
    
    structure : supercell instance
        one of the structures parsed using the GeneralInputParser
    
    Phi : None, empty list or rectangular np.ndarray
        design matrix. if supplied the matrix will be extended along axis 0
    
    stochastic : tuple
        ("all",) enforces the selection of all entries in bonds.
        ("uniform",N) selects N entries from bonds uniformly at random without repetition
    
    reference_densities : empty list or np.ndarray
        contains the reference density values for each bond
    
    seed : int
        value numpy.random.seed
    
    usefortran : boolean
        switches fortran computation on or off. Note that the fortran routines in folder "fortran" 
        need to be compiled beforehand.
    
    multrithreading : boolean
        if usefortran = True this can be used to allow for openmp parallelisation

    Returns
    -------
    Phi : float np.ndarray
        Updated design matrix.

    reference_densities : float np.ndarray
        The corresponding reference densities.
    """
    if stochastic[0] != "all":
        # behaviour for stochastic behaviour not defined yet
        raise NotImplementedError

    print('entering bond generation for {}'.format(structure["name"]))

    bonds = misc.get_observations(s=structure,r_cut=basis["info"]["rcut"],num_neigh=basis["info"]["num_neigh"],
        selection=selection,iso=basis["info"]["iso"],aniso=basis["info"]["aniso"],
        ultra_num=ultra_num,verbose=verbose,seed=seed,ani_type=basis["info"]["ani_type"],
        ani_specification=basis["info"]["ani_specification"],usefortran=usefortran)
    
    print('bond generation succesful ({}) many, entering design matrix generation for {}'.\
        format(len(bonds),structure["name"]))

    Phi =  get_design_matrix(bonds=bonds,basis=basis,mapper=mapper,Phi=Phi,
                stochastic=stochastic,return_t=False,seed=seed,usefortran=usefortran,
                multithreading=multithreading)

    if len(reference_densities)==0 and len(bonds)!=0:
        # first time function is called, array is empty 
        reference_densities = np.array([v.t["density"] for v in bonds],dtype=np.float64)
    elif len(bonds)!=0:
        # do not stack if no new bonds are found
        reference_densities = np.vstack((reference_densities,np.array([v.t["density"] for v in bonds],\
                                          dtype=np.float64)))
    
    return Phi, reference_densities

def get_design_matrix(bonds,basis,mapper,Phi=None,verbose=False,stochastic=("all",),return_t=False,
                      seed=None,usefortran=False,multithreading=True):
    """Calculates the design matrix using the info in bonds.
    
    To generate a design matrix for the density regression we make use of the
    additivty of the electron density. Hence a basis component is applied to all
    relevant values stored in x and then summed.

    Parameters
    ----------
    bonds : list of instances of the local_bond_info class
        required to find the number of elements present, whether or not 
        to generate terms for anisotropic approximation, ...
    
    basis : list of instances of basis functions
        obtained from fitelectrondensity.rvm.get_basis
    
    mapper : dict
        contains the information which basis functions correspond to what
        kind of contribution to the electrondensity, i.e. isotropic/anisotropic, elements,...
    
    Phi : None, empty list or rectangular np.ndarray
        design matrix. if supplied the matrix will be extended along axis 0
    
    stochastic : tuple
        ("all",) enforces the selection of all entries in bonds.
        ("uniform",N) selects N entries from bonds uniformly at random without 
        repetition
    
    return_t : boolean
        decides whether or not the target vector (np.ndarray of shape (N,)) 
        will be returned
    
    seed : int
        value numpy.random.seed
    
    usefortran : boolean
        switches fortran computation on or off. Note that the fortran routines 
        in folder "fortran" need to be compiled beforehand.
    
    multrithreading : boolean
        if usefortran = True this can be used to allow for openmp 
        parallelisation

    Returns
    -------

    Phi : np.ndarray of shape (N,M)
        Design matrix for N observations and M basis functions/features.
    
    t : np.ndarray of shape (N,)
        Only returned if return_t is True. Contains electron density target values.
    """
    if verbose: 
        print("Generating design matrix...")
        print("mapper {}".format(mapper))
    aniso = basis["info"]["aniso"]

    # sanity check Phi
    assert Phi is None or isinstance(Phi,(list,np.ndarray)), "Assertion failed - Phi is supposed to be either None, of type list or np.ndarray!"
    
    N = len(bonds)

    if aniso:
        assert all([v.ani for v in bonds]) and any([(isinstance(v,tuple)) for v in mapper]), "Assertion failed - "+\
            "expected to find bonds and basis both generated with aniso = True..."
    
    if not seed is None:
        np.random.seed(seed=seed)
    
    # sampling of the bonds (could potentially be deprecated)
    stochastic_types = ["all","uniform"]
    
    assert stochastic[0] in stochastic_types, "Assertion failed - got {} as stochastic[0], expected one of {}...".format(stochastic_types)
    
    if stochastic[0] == "all": # using all observations
        Nsamples = N
        idx_bonds = list(range(N))
    elif stochastic[0] == "uniform": # using observations sampled uniformly at random
        Nsamples = int(stochastic[1])
        if Nsamples > N:
            warnings.warn("The number of samples (Nsamples) specified with {} as {} exceeds the number of available samples N {}. Hence Nsamples is set equal to N...".format(stochastic,Nsamples,N))
            Nsamples = N
        idx_bonds = np.random.choice(range(N),size=Nsamples,replace=False)
    else:
        raise NotImplementedError
    
    tmp_bonds = [bonds[v] for v in idx_bonds]

    print("Generating design matrix for {} samples...".format(Nsamples))

    q_types = {"q","q-r","q-1/r","atom_q-r","atom_q-r-ref","q-r-taper1"}
    for _q in list(q_types):
        q_types.add(_q+"-cos")
        q_types.add(_q+"q-sin") # why the inconsistency?

    firstcall = True

    for s,ix_s,ix_e in mapper: # 's' contains the key for bond.x["r"] (isotropic) or bond.x["ani"] (anisotropic) as well as further specifications such as the lower case L value for BOPs
        print("mapping >> s {}, ix_s {}, ix_e {}".format(s,ix_s,ix_e))
        if isinstance(s,str): # the isotropic case, rows are bonds, columns are basis functions - if a bond has nothing related to variable 's' the entry is zero
        
            #---------------------------#
            # isotropic basis functions #
            #---------------------------#
            
            if usefortran:
                tmp_Phi = f90.isotropic_phi(basis["info"],bonds=tmp_bonds,species=s,multithreading=multithreading)
            else:
                tmp_Phi = np.array([[np.sum(f(bond.x["r"][s])) if s in bond.x["r"] else 0 \
                    for f in basis["functions"][ix_s:ix_e]] for bond in tmp_bonds],dtype=np.float64)

        elif isinstance(s,tuple): # anisotropic cases
            print("s {}".format(s))
            if s[1] in q_types: # BOPs
                l_map = {l:v for v,l in enumerate(bonds[0].ani_specification["l_range"])}
                                
                tmp_Phi = np.array([[f(bond.x["ani"][s[0]][l_map[s[2]]]) if s[0] in bond.x["r"] \
                    else 0 for f in basis["functions"][ix_s:ix_e]] for bond in bonds],dtype=np.float64)
                
                print("tmp_Phi {} {}".format(tmp_Phi.shape,np.sum(tmp_Phi)))
            
            elif s[1] == "R": # rows are bonds, columns are basis functions
                # values to apply the basis functions to
                vals = np.array([[bond.x["ani"][s[0]] if s[0] in bond.x["r"] else 0 \
                    for f in basis["functions"][ix_s:ix_e]] for bond in bonds],dtype=np.float64)

                # applying the basis functions
                tmp_Phi = np.array([f(vals[:,v]) for v,f in enumerate(basis["functions"][ix_s:ix_e])]).T
                
                print("tmp_Phi {} {}".format(tmp_Phi.shape,np.sum(tmp_Phi)))
            
            elif s[1] == "atom_q-r-ref2":
                
                lmap = {l:v for v,l in enumerate(bonds[0].ani_specification["l_range"])}

                print("bond {}".format(bonds[0].x["ani"]["all"].shape))
                                
                tmp_Phi = np.array([[np.sum(\
                    f(bond.x["ani"][s[0]][:,lmap[s[2]],0],bond.x["ani"][s[0]][:,lmap[s[2]],1])) \
                    if s[0] in bond.x["ani"] else 0 for f in basis["functions"][ix_s:ix_e]] \
                    for bond in tmp_bonds],dtype=np.float64)
                
                print("tmp_Phi {} {}".format(tmp_Phi.shape,tmp_Phi.sum()))
                                
            else: 
                #------------------------------------#
                # MEAM or polynomial basis functions #
                #------------------------------------#
                
                if usefortran:
                    t1 = time.time()
                    tmp_Phi = f90.anisotropic_phi(basis["info"],tmp_bonds,s,multithreading)    
                    t2 = time.time()
                else:
                    t1 = time.time()
                    tmp_Phi = np.array([[np.sum(f(*bond.x["ani"][s].T)) if s in bond.x["ani"] \
                        else 0 for f in basis["functions"][ix_s:ix_e]] for bond in tmp_bonds],dtype=np.float64)
                    t2 = time.time()
                
                print ('time doing aniso design matrix: {}'.format(t2-t1))
        else:
            raise NotImplementedError
        if np.isnan(tmp_Phi).any():
            idx_nan = np.where(np.isnan(tmp_Phi))[0]
            print("found nans {} ...".format(idx_nan))
            
        if verbose: print("tmp_Phi shape {}".format(tmp_Phi.shape))
        
        if np.isnan(tmp_Phi).any():
            raise ValueError("Found NaN value in tmp_Phi, something went wrong...")
        
        
        if firstcall:
            Phi_new = np.array(tmp_Phi,dtype=np.float64)
            firstcall = False
        else:
            Phi_new = np.hstack((Phi_new,tmp_Phi))

    if Phi is None or len(Phi)==0:
        # first time update of design matrix has been called
        Phi = np.array(Phi_new)
    elif isinstance(Phi,np.ndarray):
        # stack new contribution to design matrix below existing block
        Phi = np.vstack((Phi,Phi_new))
    else:
        raise ValueError("Phi is supposed to be None, of type list or np.ndarray!")
    
    if verbose: print("final design matrix dimensions {}".format(Phi.shape))
    if return_t:
        t = np.array([bond.t["density"] for bond in tmp_bonds],dtype=float)
        return Phi, t
    else:
        return Phi

def get_covariance_after_evidence(alpha,beta,Phi):
    """Calculates the covariance of p(w|t).
    
    Parameters
    ----------
    Phi : np.array of float (N,M)
        design matrix
    """
    N,M = Phi.shape
    A = np.eye(M,dtype=np.float64)
    for i in range(M):
        A[i,i] = alpha[i]
    
    inv_covariance = A + beta * np.dot(Phi.T,Phi)
    covariance = np.linalg.inv(inv_covariance)
    return inv_covariance, covariance

#@memory_profiler.profile
def get_mean_after_evidence(beta,Sn,Phi,t):
    """Calculates the mean of p(w|t).
    
    Parameters
    ----------
    Phi : np.array of float (N,M)
        design matrix
    """
    N,M = Phi.shape
    mean = np.zeros(M,dtype=np.float64)
    mean = beta * np.dot(Sn ,np.dot(Phi.T,t))
    mean = np.reshape(mean,(-1,))
    return mean

def get_updated_hyperparameters(t,mean,Phi,Sigma,N,M,alpha_old):
    
    gamma = np.array([1-alpha_old[v]*Sigma[v,v] for v in range(M)],dtype=float)
    alpha = np.array([gamma[v]/mean[v]**2 for v in range(M)])
    inv_beta = np.linalg.norm(t-np.dot(Phi,mean))**2 / (N-np.sum(gamma))
    beta = 1./inv_beta
    return beta, alpha

#@memory_profiler.profile
def get_updated_hyperparameters_sparse(t,mean_sparse,Phi_sparse,Phi,Sigma_sparse,
                                       N,M,alpha_old,beta_old,max_memory=None):
    """Making use of section 7.2.2 in Bishop's book.
    
    Considering all alpha at the same time
    
    Parameters
    ----------
    mean_sparse : np.ndarray (M_sp,)
        active model weights with M_sp <= M
    Phi_sparse : np.ndarray (N,M_sp)
        N - number of observations and M is the total number of basis functions
    Sigma_sparse : np.ndarray (M_sp,M_sp)
    alpha_old : np.ndarray (M_sp,)
    beta_old : float
    max_memory : optional
    """
    """
    if max_memory is not None:
        tmp_max_mem = max_memory
        mem_proj = float(Phi_sparse.shape[0] ** 2 * sys.getsizeof(np.float64())) / 3. * 1e-9
        if mem_proj > tmp_max_mem:
            raise MemoryError("The required RAM is likely to exceed the provided cap of {} GB (projects {} GB), hence the process is stopped...".format(tmp_max_mem,mem_proj)+\
                              "Either increase the memory cap by increasing max_memory or change the number of processed density points via the 'selection' or 'stochastic' variable...")
    """
    #tmp0 = np.dot(Phi_sparse, np.dot(Sigma_sparse, Phi_sparse_trans))
    #print("tmp0 {}".format(sys.getsizeof(tmp0)))
    #tmp1 = beta_old * Phi_trans

    #tmp2 = beta_old**2 * np.dot(Phi_trans,tmp0)
    #tmp2 = beta_old**2 * np.dot(np.dot(Phi_trans, Phi_sparse),np.dot(Sigma_sparse,Phi_sparse_trans))

    #print("tmp1 {} tmp2 {}".format(sys.getsizeof(tmp1),sys.getsizeof(tmp2)))
    #new_tmp2 = beta_old**2 * np.dot(np.dot(Phi_trans, Phi_sparse),np.dot(Sigma_sparse,Phi_sparse_trans))
    #print("A*(B*C) == (A*B)*C {}".format(np.allclose(tmp0,np.dot(np.dot(Phi_sparse,Sigma_sparse), Phi_sparse_trans))))
    #print("tmp2 and new_tmp2 {}".format(np.allclose(tmp2,new_tmp2)))
    #raise
    
    #tmp = tmp1 - tmp2
    tmp = beta_old * Phi.T - beta_old**2 * np.dot(np.dot(Phi.T, Phi_sparse),np.dot(Sigma_sparse,Phi_sparse.T))
    Q = np.dot(tmp,t)
    Q = np.reshape(Q,(-1,))
    S = np.einsum("ij,ji->i",tmp,Phi)
    
    # calculate q and s, yes notice the lower case 
    q = np.zeros(M)
    s = np.zeros(M)
    alpha_inf = np.where(np.isinf(alpha_old))[0]
    alpha_fin = np.where(np.isfinite(alpha_old))[0]
    alpha_sp = alpha_old[alpha_fin]
    q[alpha_inf] = Q[alpha_inf]
    q[alpha_fin] = alpha_sp*Q[alpha_fin] / (alpha_sp - S[alpha_fin])
    s[alpha_inf] = S[alpha_inf]
    s[alpha_fin] = alpha_sp*S[alpha_fin] / (alpha_sp - S[alpha_fin])
    

    #gamma = np.array([1.-alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))],dtype=float)
    alpha_update = s**2/(q**2-s)

    # calculate the new beta value
    t_pred = np.dot(Phi_sparse,mean_sparse)
    t_pred = np.reshape(t_pred,(-1,1))

    #inv_beta = np.linalg.norm(t-t_pred)**2 / (N - M + np.sum([alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))]))
    #beta = 1./inv_beta 
    dt = np.linalg.norm(t-t_pred)**2
    top = N - M + np.dot(alpha_sp,np.diag(Sigma_sparse))
    beta = top/dt
    #inv_beta = np.linalg.norm(t-np.dot(Phi_sparse,mean_sparse))**2 / (N-np.sum(gamma))
    #beta = 1./inv_beta
    #print("beta: N = {}, M = {}, ||t-y||^2 = {}, N - M + sum_m alpha_m Sigma_mm = {}".format(N,M,np.linalg.norm(t-t_pred)**2,\
    #                np.sum([alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))])))
    
    q2_smaller_equal_s = np.where(q**2<=s)[0]
    q2_larger_s = np.where(q**2>s)[0]
    # q**2 > s and alpha is finite/infinte - update with 7.101
    alpha = np.array(alpha_old)
    alpha[q2_larger_s] = alpha_update[q2_larger_s]
    
    ## q**2 < s and alpha is finite set new alpha to infinite
    alpha[np.intersect1d(q2_smaller_equal_s,alpha_fin)] = np.inf
    
    if (alpha<0).any(): # if this happens there is a problem since alphas can't be negative by definition
        
        print("to update with 7.101 {}".format(sorted(list(q2_larger_s))))
        print("q**2>s -> alpha {}".format(alpha[q2_larger_s]))
        print("alpha_old {}".format(alpha_old))
        print("alpha {}".format(alpha))
        print("q2 <= s {}".format(q2_smaller_equal_s))
        print("alpha finite {}".format(alpha_fin))
        print("intersection q2 <= s and alpha finite {}".format(np.intersect1d(q2_smaller_equal_s,alpha_fin)))
        print("s {}".format(s))
        print("q**2 {}".format(q**2))
        raise
    
    return alpha, beta

def get_updated_hyperparameters_sparse_sequential(i,t,mean_sparse,Phi_sparse,Phi_sparse_trans,Phi,Phi_trans,Sigma_sparse,
                                       N,M,alpha_old,beta_old,max_memory=None):
    """Making use of section 7.2.2 in Bishop's book.
    
    Considering all alpha at the same time

    """
    if max_memory is not None:
        tmp_max_mem = max_memory * 1e9
        mem_proj = float(Phi_sparse.shape[0] ** 2 * sys.getsizeof(np.float64())) / 3.
        if mem_proj > tmp_max_mem:
            raise MemoryError("The required RAM is likely to exceed the provided cap of {} GB (projects {} GB), hence the process is stopped...".format(tmp_max_mem,mem_proj))

    # calculate useful quantities
    tmp1 = beta_old * Phi_trans[i,:]
    tmp2 = beta_old**2 * np.dot(Phi_trans[i,:],np.dot(Phi_sparse, np.dot(Sigma_sparse, Phi_sparse_trans)))
    tmp = tmp1 - tmp2

    # calculate Q and S
    Q = np.dot(tmp,t)
    S = np.dot(tmp,Phi[:,i])
    
    # calculate q and s
    idx_fin = np.where(np.isfinite(alpha_old))
    idx_inf = np.where(np.isinf(alpha_old))
    M_sp = len(idx_fin)
    alpha_sp = np.array(alpha_old[idx_fin])
    if np.isinf(alpha_old[i]):
        q = Q
        s = S
    else:
        q = alpha_old[i]*Q / (alpha_old[i] - S)
        s = alpha_old[i]*S / (alpha_old[i] - S)    
        
    # calculate gamma and the alpha_update value which is used to judge how to update the alpha
    #gamma = np.array([1.-alpha_sp[v]*Sigma_sparse[v,v] for v in range(M_sp)],dtype=float)
    alpha_update = s**2/(q**2-s)

    # calculate the new beta value    
    N = float(len(t))
    t_pred = np.dot(Phi_sparse,mean_sparse)
    t_pred = np.reshape(t_pred,(-1,1))
    inv_beta = np.linalg.norm(t-t_pred)**2 / (N - M + np.sum([alpha_sp[v]*Sigma_sparse[v,v] for v in range(M_sp)]))
    beta = 1./inv_beta

    print("beta: N = {}, M = {}, ||t-y||^2 = {}, N - M + sum_m alpha_m Sigma_mm = {}".format(N,M,np.linalg.norm(t-t_pred)**2,\
                    np.sum([alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))])))
    
    q2_smaller_equal_s = np.where(q**2<=s)[0]
    q2_larger_s = np.where(q**2>s)[0]
    
    # updating alpha   
    alpha = np.array(alpha_old)
    
    if q**2 > s: # q**2 > s and alpha is finite/infinte - update with 7.101
        alpha[i] = alpha_update
    elif q**2 < s and np.isfinite(alpha_old[i]): # q**2 < s and alpha is finite set new alpha to infinite
        alpha[i] = np.inf
    if np.isinf(alpha).all():
        alpha[0] = 1.
    if (alpha<0).any():
        
        print("to update with 7.101 {}".format(sorted(list(q2_larger_s))))
        print("q**2>s -> alpha {}".format(alpha[q2_larger_s]))
        print("alpha_old {}".format(alpha_old))
        print("alpha {}".format(alpha))
        print("q2 <= s {}".format(q2_smaller_equal_s))
        print("alpha finite {}".format(alpha_fin))
        print("intersection q2 <= s and alpha finite {}".format(np.intersect1d(q2_smaller_equal_s,alpha_fin)))
        print("s {}".format(s))
        print("q**2 {}".format(q**2))
        raise
    
    return alpha, beta

def get_loglikelihood(t,weights,Phi,alpha,beta,N,M):
    # total log likelihood for all observations given current weights, alpha and beta
    
    t_pred = np.dot(Phi,weights)
    t_pred = np.reshape(t_pred,(-1,1))
    
    hyper = np.sum(np.log(alpha)) + N * np.log(beta)
    evidence = beta * np.sum((t-t_pred)**2)
    ln2pi = np.log(2*np.pi)
    regularizer = np.dot(alpha,weights**2)
    
    return - .5*(M+N)*ln2pi + .5*N*np.log(beta) + .5*np.sum(np.log(alpha) - alpha*(weights**2)) - .5*evidence  

def get_mse(t,weights,Phi):
    t_pred = np.dot(Phi,weights)
    t_pred = np.reshape(t,(-1,1))
    
    return np.linalg.norm(t-t_pred)**2

def iterate(Phi, t, niter=10, verbose=False, alpha_init=None, beta_init=None,\
            tol=1e-6, fix_beta=False, sequential=False, n_steps_beta=1, \
            max_memory=None, seed=None):
    """Iteratively updating weights and hyperparameters using an Relevance Vector Machine.

    Parameters
    ----------
    
    Phi : float np.ndarray of shape (N,M)
        Design matrix with M basis functions/features and N observations.
    
    t : float np.ndarray of shape (N,)
        Vector of target values.

    verbose : boolean, optional, default False

    alpha_init : None or np.ndarray of shape (M,), optional, default None
        Vector of initial alpha values.
    
    beta_init : None or float, optional, efault None
        Initial beta value.

    tol : float, optional, default 1e-6
    
    fix_beta : boolean, optional, default False
    
    sequential : boolean, optional, default False
    
    n_steps_beta : int, optional, default 1
        How often to update beta.
    
    max_memory : None or float, optional, default None
        If not None then specifies the maximum RAM in GB
        to be used. (outdated)
        
    seed : None or int, optional, default None

    Returns
    -------

    logbook : dict
        Contains loglikelihood ("L"), alpha hyperparameters ("alphas"),
        sparse weights ("weights"), complete weights vectors ("weights_full"),
        mean squared error ("mse"), total squared error ("tse"), 
        minimum deviation ("min"), maximum deviation ("max"), mean of deviations 
        ("dev_est"), one standard deviation of deviations ("dev_std"),
        median squared error ("median_se")
    """
    print("Starting RVM iteration...")
    
    N,M = Phi.shape

    if not seed is None:
        np.random.seed(seed=seed)
    N, M = Phi.shape
    
    # initializing hyper parameters
    if alpha_init is None:
        alpha_init = np.ones(M)
        alpha_init[1:] = np.inf
    if beta_init is None:
        beta_init = 75.

    if verbose:
        print("Starting with alpha = {}".format(alpha_init))
        print("Starting with beta = {}".format(beta_init))
    
    logbook = {"L":[],"alphas":[],"beta":[],"weights":[],"weights_full":[],
               "mse":[],"tse":[],"min":[],"max":[],"Sigma":[],"dev_est":[],
               "dev_std":[],"median_se":[]}
    
    # iteratively calculate: 1) weights based on hyper parameters, 2) hyperparameters based on weights
    if verbose:
        print("beta_init {}".format(beta_init))
        print("alpha_init {}".format(alpha_init))
    
    t_sig, t_w, t_q, t_hyp = 0, 0, 0, 0
    
    for i in range(niter):
        
        alpha_fin = np.where(np.isfinite(alpha_init))[0]
        alpha_inf = np.where(np.isinf(alpha_init))[0]
        if verbose: 
            print("iteration {}/{}".format(i+1,niter))
            print("    >> active phis {}...".format(len(alpha_fin)))
        
        # get the sparse version of Phi and the alpha vector
        Phi_sparse = Phi[:,alpha_fin]
        alpha_sparse = alpha_init[alpha_fin]
        
        # calculate the covariance matrix and its inverse for the normal distribution p(w|X,t,hyper)
        try:
            t0 = time.time()
            inv_Sn_sparse, Sn_sparse = get_covariance_after_evidence(alpha_sparse,beta_init,Phi_sparse)
        except np.linalg.linalg.LinAlgError:
            print("ERROR: encountered singular matrix in get_covariance_after_evidence! Stopping iteration...")
            alpha_curr = alpha_init
            beta_curr = beta_init
            weights_new = np.zeros(M)
            weights_new[alpha_fin] = weights_new_sparse
            
            break
        # calculate the weights using the newly obtained covariance matrix
        t0 = time.time()
        weights_new_sparse = get_mean_after_evidence(beta_init,Sn_sparse,Phi_sparse,t)
        t_w += time.time()-t0
        
        if np.isnan(weights_new_sparse).any():
            print("\nSn {}".format(Sn_sparse))
            print("\ninv_Sn {}".format(inv_Sn_sparse))
            print("\nalpha_init {}".format(alpha_init_sparse))
            print("\nbeta_init {}".format(beta_init))
            raise
        
        # calculate and store quality measures alongside key properties such as the weights
        t0 = time.time()
        qualities = misc.get_quality_measures(t,weights_new_sparse,Phi_sparse,alpha_sparse,beta_init,N,M)
        t_q += time.time()-t0
        
        for _key in qualities.keys(): 
            logbook[_key].append(qualities[_key])
        logbook["alphas"].append(alpha_init)
        logbook["beta"].append(beta_init)
        logbook["weights"].append(weights_new_sparse)
        weights_full = np.zeros(M)
        weights_full[alpha_fin] = weights_new_sparse
        logbook["weights_full"].append(weights_full)
        
        if verbose: 
            print("    >> log_joint {} mse {} tse {} min {} max {}".format(round(logbook["L"][i],6),round(logbook["mse"][i],6),round(qualities["tse"],6),round(qualities["min"],6),round(qualities["max"],6)))
                
        if sequential:
            # update a single alpha
            i_alpha = i%M
            print("i_alpha {}".format(i_alpha))
            alpha_new, beta_new = get_updated_hyperparameters_sparse_sequential(i_alpha,\
                t, weights_new_sparse, Phi_sparse, Phi_sparse_trans, Phi, Phi_trans,\
                Sn_sparse, N, M, alpha_init, beta_init, max_memory=max_memory)
        else:
            # update all alpha at one
            alpha_new, beta_new = get_updated_hyperparameters_sparse(t,\
                weights_new_sparse, Phi_sparse, Phi, Sn_sparse, N, M,\
                alpha_init, beta_init, max_memory=max_memory)

        if i<niter-1:            
            alpha_init = copy.deepcopy(alpha_new)
            if (not fix_beta) and i%n_steps_beta==0 and i>0:
                beta_init = copy.deepcopy(beta_new)
        else:
            alpha_curr = alpha_init
            beta_curr = beta_init
            weights_new = np.zeros(M)
            weights_new[alpha_fin] = weights_new_sparse
    print("time spent on:\nSigma {} s...\nweights {} s...\nqualities {} s...\nhyperparameters {} s...".format(t_sig, t_w, t_q, t_hyp))

    if verbose: 
        print("RVM iteration completed")
    return logbook

def predict(Phi,w):
    """Predict y given the design matrix 'Phi' and the model parameters 'w'

    Parameters
    ----------
    
    Phi : float np.ndarray of shape (N,M)

    w : float np.ndarray of shape (M,)

    Returns
    -------

    y : float of np.ndarray of shape (N,)
    """
    return np.dot(Phi,w)