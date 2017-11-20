"""
This script is intended to set up the input necessary for the fitting of an EAM potential
to DFT data. Hence this script does all the parsing and pre-calculations the fitting
process in EAM_fit.py depends on.
"""
import sys
import time
import numpy as np
import pickle

# Potty specific
sys.path.insert(0,"../")
import fitenergy as fe
import fitelectrondensity as fed
import parsers

# info loading from disk
dft_path = "./unittest_files/ideal_dis_Ni3Al/" #small, small_dis, small_dis_singleMD
#dft_path = "./single_Al/"
load_path_rhos = "./unittest_files/test_energyforce.rhos"

# storing data for the fitting process
dump_path = "./unittest_files/test_energyforce.pckl" # default suffix is ".pckl", if it's not present it will be added

# info for bonds
num_neigh = None # number of neighoring atoms for (ghost) atom
r_cut = 6.
aniso = False
ultra_num = None #number of additional super cells in each sim box vector direction
#selection=[("Al_4d","all",),("Al_3d","all",),("Al_dis","all"), ("Al_100","all"),("Al_300","all"),("A1_c","all"),("A2_c","all"),("A3_c","all")] # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
selection=[("A","all"),("Ni","all")]#[("A1","all"),("A2","all"),("A3","all")]#,("A1_c","all"),("A2_c","all"),("A3_c","all")]
seed = 42

# info for basis set
k_pair = 25 # parameter for basis size of pair energy functions
k_emb = 30 # parameter for basis size of embedding energy functions
smooth_emb = False
smooth_pair = True 
type_pair="Fourier" # type of basis for pair energy functions
type_emb="Fourier" # type of basis for embedding energy functions
r_smooth = 6.
f_smooth = .01 # fudge factor for smoothing distances
f_smooth_emb = .01 # fudge factor for smoothing embedding densities

# info for EAM
r_lb=0.
r_ub=float(r_cut)
#rho_lb=0.
#rho_ub=42.
return_rho_bounds = True
rho_conv_type = None #"psi", None
rho_operations = []#["absolute","normalize"]
rho_params = [r_cut,.01] # parameters for the convolution of rho functions
N_steps = 100000

def main(f_smooth_emb):
    rho_scaling = 1.
    np.random.seed(seed=seed)

    # parsing DFT files
    gip = parsers.general.GeneralInputParser()
    gip.parse_all(dft_path)
    gip.sort()

    # loading previously created rhos
    rho_dict = fed.misc.load_regressed_rho(load_path_rhos,operations=rho_operations,return_bounds=False,conv=rho_conv_type,params=rho_params)

    # get observation input and output: X, t    
    t0 = time.time()
    bonds = fe.nonlin.get_observations(gip,rho_dict,ultra_num=ultra_num,num_neigh=num_neigh,r_cut=r_cut,verbose=True,selection=selection)
    print("generated bonds {}s...".format(time.time()-t0))
    
    #print("pos {}".format([v.super_pos for v in bonds]))

    a0 = [bond.box[0,0] for bond in bonds]

    t = fe.nonlin.get_all_targets(bonds)
    X, rho_lb, rho_ub = fe.nonlin.get_all_sources(bonds,return_rho_bounds=return_rho_bounds)
    print("embedding densities {}...".format(X["density"]))
    
    # rescale rho(r) and embedding densities by the largest observed embedding density
    
    obs_emb = np.amax([v for v2 in X["density"] for v in v2])
    X["density"] = [v/obs_emb for v in X["density"]]
    #max_obs_emb = float(max(rho_ub.values()))
    for _s in rho_dict["rhos"].keys():
        rho_dict["rhos"][_s] /= obs_emb
        rho_min = min([np.amin(v) for v in X["density"]])
        rho_lb[_s] = 0 if rho_min > 0 else rho_min
        rho_ub[_s] = 2*max([np.amax(v) for v in X["density"]])

    print("\nEmbedding densities after re-scaling {}".format([list(v) for v in X["density"]]))
    print("\na0 {}".format(a0))
    print("\nnames {}".format([bond.name for bond in bonds]))
    print("rho_lb {}\nrho_ub {}".format(rho_lb,rho_ub))

    # parameter mapping and pre-calculation of values for subsequent optimization of model parameters
    print("preparing optimization process...")
    mapper = fe.nonlin.get_mapper(k_pair,k_emb,X)
    print("mapper {}".format(mapper))
    #rho_scaling = max(rho_ub.values())
    print("f_smooth {}".format(f_smooth))
    print("f_smooth_emb {}".format(f_smooth_emb))
    params = fe.nonlin.get_precomputed_energy_force_values(X,f_smooth,r_smooth,rho_dict,type_pair=type_pair,type_emb=type_emb,                                                smooth_emb=smooth_emb,smooth_pair=smooth_pair,rho_scaling=rho_scaling,k_emb=k_emb,k_pair=k_pair,f_smooth_emb=f_smooth_emb)
    #print("params {}".format(params))
    #print(params["r_vec"])
    
    # initializing model parameters
    print("initializing weights and energy/force functions...")
    x0 = fe.nonlin.get_initial_weights(mapper)
    
    if f_smooth_emb is None:
        f_smooth_emb = f_smooth
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
    

    fe.nonlin.save_data_for_fit(dump_path,data)

if __name__=="__main__":
    
    main(f_smooth_emb)
