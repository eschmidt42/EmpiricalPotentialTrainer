import sys
import time
import numpy as np
import pickle

# Potty specific
sys.path.insert(0,"../")
import parsers
import fitelectrondensity as fed

# info loading from disk
dft_path = "./unittest_files/ideal_dis_Ni3Al/" #EV_A1_h_LDA

# info for bonds
num_neigh = None # number of neighoring atoms for (ghost) atom
r_cut = 6.
aniso = False
ani_type = "MEAM" # "MEAM", BOP, BOP-r or BOP-invr, "R", "BOP4atoms", "BOP4atoms-ref", "BOP4atoms-ref2", "BOP-taper1"
# useful l values: 3, 4, 5 (bad), 6 (very bad), 7 (bad), 8 (very bad), 9, 10, 11 (bad), 12 (very bad), 42
ani_specification = {"l_range": np.array([4,6,8],dtype=int),
                     "usage":"everything","r_ani":4.} #"usage": "everything" (is "usage" is not present this is assumed as default, all" and aphabetically sorted elements), "all" (only use q obtained for "all"), "Ni" (only usq q obtained for "Ni")
q_tol = .1 # plus and minus tolerance for the final output data

ultra_num = None #number of additional super cells in each sim box vector direction
selection=("atom",.5,"r") # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
seed = 42
np.random.seed(seed=seed)

# info for basis set
k_iso=50 # parameter for iso basis size
k_ani= 5 # parameter for ani basis size
type_iso="Fourier" # type of basis for iso
type_ani="Fourier" # type of basis for ani
smooth = True # apply smoothing of basis functions towards r_smooth
r_smooth = 6.
self_contribution = False # whether or not an atom itself is considered contribution to the valence elenctron density
f_smooth = .1 # fudge factor for smoothing

# plotting rho(r)
Nsteps_iso = 10000 #number of steps to to plot rho(r) from r = 0 to r = r_smooth

# RVM initiate hyper parameters
niter = 1
tol = 1e-6
fix_beta = False
n_steps_beta = 1 # integer specifying every nth step to update beta
sequential = False
stochastic = ("all",) # ("all",), ("uniform",5000) switch for rvm.get_design_matrix, specifies how many and which observations from the ones already processed are actually used in a given run 
niter_stochastic = 1 # number of iterations generating new Phis according to variable 'stochastic' if stochastic[0] != "all"

# RAM cap
max_memory = 25. # [GB] - cap on maximum size of most memory consuming variable in rvm.get_updated_hyperparameters

# writing paths
save_path_rhos = "./unittest_files/test_energyforce.rhos"
#save_path_predicted = "./predicted_Ni3Al_ideal_dis.pred"
save_path_Phi = "./unittest_files/test_energyforce.phi"
save_path_bond_info = "./unittest_files/test_energyforce.bonds"

usefortran = False

def main():
    # parsing DFT files
    gip = parsers.general.GeneralInputParser()
    gip.parse_all(dft_path)
    gip.sort()
    # get observation input and output: X, t    
    t0 = time.time()
    bonds = []
    gips = iter(gip)

    for tmp_gip in gips:
        print("name {}".format(tmp_gip.get_name()))
        tmp_bonds = fed.misc.get_observations(tmp_gip,ultra_num=ultra_num,num_neigh=num_neigh,r_cut=r_cut,aniso=aniso,verbose=True,selection=selection,seed=seed,ani_type=ani_type,ani_specification=ani_specification,usefortran=usefortran)
        bonds.extend(tmp_bonds)
    
    neg_dens = [bond.t["density"] for bond in bonds if bond.t["density"]<0.]
    
    print("found {} negative densities...: {}".format(len(neg_dens),neg_dens))
    print("generated bonds {}s...".format(time.time()-t0))
    
    # get basis
    basis, mapper = fed.rvm.get_basis(bonds,k_iso=k_iso,k_ani=k_ani,type_iso=type_iso,type_ani=type_ani,
                            smooth=smooth,r_smooth=r_smooth,f_smooth=f_smooth,verbose=True,self_contribution=self_contribution)

    # "stochastic gradient descent"
    beta_init = 1e2
    np.random.seed(seed=seed)
    if stochastic[0] == "all":
        print("\nProcessing all observations during regression...")

        # get design matrix
        t0 = time.time()
        Phi, t = fed.rvm.get_design_matrix(bonds,basis,mapper,verbose=True,return_t=True,seed=seed,usefortran=usefortran)

        print("generated design matrix (Phi) {}s...".format(time.time()-t0))

        # do the RVM
        M = Phi.shape[1]
        alpha_init = np.ones(M)
        alpha_init[1:] = np.inf
        #alpha_init[np.random.randint(0,M,size=np.random.randint(0,M))] = np.inf
        first_alphas = np.array(alpha_init)
        #beta_init = 10**np.random.uniform(-5,5)
        
        t0 = time.time()
        logbook = fed.rvm.iterate(Phi,t,niter=niter,verbose=True,alpha_init=alpha_init,beta_init=beta_init,tol=tol,
                                  fix_beta=fix_beta,sequential=sequential,n_steps_beta=n_steps_beta,max_memory=max_memory,seed=seed)
        print("completed RVM regression {}s...".format(time.time()-t0))
    else:
        print("\nProcessing a subset of all observations during regression as specified: {}".format(stochastic))
        t0s = time.time()
        
        #beta_init = 10**np.random.uniform(-5,5)
        for i in range(niter_stochastic):
            
            print("\nStochastic iteration {}/{}...".format(i+1,niter_stochastic))
            # get design matrix
            t0 = time.time()
            Phi,t = fed.rvm.get_design_matrix(bonds,basis,mapper,aniso,verbose=True,stochastic=stochastic,return_t=True,seed=seed)

            if i == 0:
                M = Phi.shape[1]
                alpha_init = np.ones(M)
                #alpha_init[np.random.randint(0,M,size=np.random.randint(0,M))] = np.inf
                alpha_init[1:] = np.inf
                first_alphas = np.array(alpha_init)

            print("generated design matrix (Phi) {}s...".format(time.time()-t0))

            # do the RVM
            t0 = time.time()
            logbook = fed.rvm.iterate(Phi,t,niter=niter,verbose=True,alpha_init=alpha_init,beta_init=beta_init,tol=tol,
                                      fix_beta=fix_beta,sequential=sequential,n_steps_beta=n_steps_beta,max_memory=max_memory,seed=seed)
            alpha_init[:] = logbook["alphas"][-1]
            beta_init = logbook["beta"][-1]
            print("completed RVM regression {}s...".format(time.time()-t0))
        print("completed all stochastic RVM iterations {}s...".format(time.time()-t0s))
    
    logbook["first_alphas"] = first_alphas
    # some plotting - performance
    #fed.misc.show_rvm_performance(niter,logbook) # likelihood and beta vs iteration
    
    ##### Last
    # some plotting
    print("\Minimum tse...")
    weights = np.zeros(M)
    #idx = len(logbook["L"])-1
    idx = np.nanargmin(logbook["tse"])
    print("Selected iteration {} for output:\nL = {}, mse = {}, tse = {}".format(idx,logbook["L"][idx],logbook["mse"][idx],logbook["tse"][idx]))
    alpha_fin = np.where(np.isfinite(logbook["alphas"][idx]))[0]
    weights[alpha_fin] = logbook["weights"][idx]
    r, rhos = fed.predict.predict_rho_iso(mapper,weights,basis,r_smooth,Nsteps_iso)
    fed.misc.show_rho_iso(r,rhos,title="Last") # rho(r)
    
    print("finished regressing {} density points...".format(len(bonds)))
    
    # saving final rhos to disk 
    fed.misc.save_regressed_rho(r,rhos,save_path_rhos,lb=0,ub=r_smooth,dft_path=dft_path,logbook=logbook,i=idx,niter=niter,
                           tol=tol,fix_beta=fix_beta,beta_init=beta_init,sequential=sequential,k_iso=k_iso,k_ani=k_ani,
                           type_iso=type_iso,type_ani=type_ani,ultra_num=ultra_num,selection=selection,num_neigh=num_neigh,
                           r_cut=r_cut,aniso=aniso,smooth=smooth,r_smooth=r_smooth,f_smooth=f_smooth)

    with open(save_path_bond_info,"wb") as f:
        pickle.dump([bond.x for bond in bonds],f)
    # predicting density
    #s = next(iter(gip)) # predicting the density for the same structure as used for the training
    #print("log likelihood {} num parameters {}".format(logbook["L"][idx],len(logbook["weights"][idx])))
    #xyz, pred_density = fed.predict.predict_n_of_r(r_smooth,r,rhos,gip[0],ultra_num=ultra_num,r_cut=r_cut,num_neigh=num_neigh,aniso=aniso)
    # write predicted density
    #fed.misc.save_predicted_density(save_path_predicted,xyz, pred_density)
    
    # some more plotting
    #fed.misc.show_densities_near_atoms(s,xyz,pred_density,title="Last")

    # saving more results to disk
    
    # write design matrix
    fed.misc.save_Phi(save_path_Phi,Phi)

if __name__=="__main__":
    
    main()
    
