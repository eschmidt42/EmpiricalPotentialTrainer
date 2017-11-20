"""
This is a unittest/tdda script for testing that the eam regression pipeline
is functional. Although the optimization itself is not carried out, resulting
energies and forces as predicted by a stored optimization are compared
against ase's EAM calculator.

In order to refresh the reference data files run:
>>> python test_edens_regresspredict.py --write-all

Just test using existing reference data files:
>>> python test_edens_regresspredict.py

Notes
-----
The regression is carried out in the recent implementation using scoop.

Eric Schmidt
e.schmidt@cantab.net
2017-10-20
"""
from __future__ import print_function
import sys, pickle, os, copy
import unittest
import numpy as np
from scipy import stats

sys.path.append("..")
import parsers
import fitelectrondensity as fed
from fitelectrondensity import linear_model
from fitelectrondensity.linear_model import RelevanceVectorMachine, distribution_wrapper
import fitenergy as fe

ref_name = lambda s,m: "regresspredict_{}_{}.csv".format(s,m)
tmp_name = lambda s,m: "tmp_regresspredict_{}_{}.csv".format(s,m)

# identify platform
if os.name == "posix":
    win = False
else:
    win = True

# current directory of this script
curr_dir = os.path.abspath(os.path.dirname(__file__))
if win:
   curr_dir = curr_dir.replace("\\","/")

# reference dir where files for comparison are stored, relative to script
reference_data_dir = curr_dir+"/unittest_files/reference/"
if win:
    reference_data_dir = reference_data_dir.replace("\\","/")

# info loading from disk
dft_path = curr_dir+"/unittest_files/EAM_test_Al-Ni/few/"

# info for bonds
num_neigh = None # number of neighoring atoms for (ghost) atom
r_cut = 6.
aniso = False
ultra_num = None #number of additional super cells in each sim box vector direction
selection=("atom",.5,"r") # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
seed = None
np.random.seed(seed=seed)

# info for basis set
k_iso=25 # parameter for iso basis size
type_iso="Fourier s+c" # type of basis for iso
smooth = True # apply smoothing of basis functions towards r_smooth
r_smooth = r_cut
r_scale = r_cut
self_contribution = False # whether or not an atom itself is considered contribution to the valence elenctron density
f_smooth = .1 # fudge factor for smoothing
usefortran = False

# parsing DFT files
print("\nparsing DFT files...")
gip = parsers.general.GeneralInputParser(verbose=False)
gip.parse_all(dft_path)
gip.sort()
print("done")

print("\nsetting up bonds...")
bonds = []
for g in gip:
    print("\nprocessing: {}".format(g.get_name()))
    _bonds = fed.misc.get_observations(g,ultra_num=ultra_num,num_neigh=num_neigh,
                                       r_cut=r_cut,aniso=aniso,verbose=False,
                                       selection=selection,seed=seed,usefortran=usefortran)
    bonds.extend(_bonds)

# shifting electron densities to be all positive    
bonds, _ = fed.misc.shift_densities(bonds,verbose=False)
print("done")

# get basis
print("\nsetting up basis and mapper...")
basis, mapper = fed.rvm.get_basis([gip],k_iso=k_iso,type_iso=type_iso,
                                  smooth=smooth,r_smooth=r_smooth,f_smooth=f_smooth,
                                  verbose=False,self_contribution=self_contribution,
                                  r_scale=r_scale)
print("done")

# get design matrix
print("\nsetting up design matrix and target vector...")
Phi, t = fed.rvm.get_design_matrix(bonds,basis,mapper,verbose=False,return_t=True,seed=seed,usefortran=usefortran)
print("design matrix dimension = {} x {}".format(Phi.shape[0],Phi.shape[1]))
print("done")

# RVM initiate hyper parameters
niter = 300
tol = 1e-3
fix_beta = False
n_steps_beta = 1 # integer specifying every nth step to update beta
sequential = False
niter_stochastic = 1 # number of iterations generating new Phis according to variable 'stochastic' if stochastic[0] != "all"

# initialize priors
N, M = Phi.shape
## manually fixed
alphas_init = np.ones(M)
alphas_init[1:] = np.inf # default
beta_init = 1./np.var(t) # default
## randomly drawn (useful for repetitions to check for regression convergence behavior)
# beta_init = distribution_wrapper(stats.halfnorm(scale=1),size=1,single=True)
# alphas_init = distribution_wrapper(stats.halfnorm(scale=1),single=False)

print("regressing...")
## old version
# logbook = fed.rvm.iterate(Phi,t,niter=niter,verbose=False,alpha_init=alpha_init,beta_init=beta_init,tol=tol,
#                           fix_beta=fix_beta,sequential=sequential,n_steps_beta=n_steps_beta,seed=seed)
## new version
model = RelevanceVectorMachine(n_iter=niter,init_alphas=alphas_init,init_beta=beta_init,compute_score=True,
                               do_logbook=True,fit_intercept=False,verbose=False,tol=tol)
model.fit(Phi,t.ravel())
logbook = model.get_logbook()
print("done")

class EAMRegressPipelineTest(unittest.TestCase):

    def test_EAM_setup(self):
        # info for bonds
        num_neigh = None # number of neighoring atoms for (ghost) atom
        r_cut = 6.
        aniso = False
        ultra_num = None #number of additional super cells in each sim box vector direction
        selection=("atom",.5,"r") # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
        seed = None
        np.random.seed(seed=seed)

        # info for basis set
        k_iso=25 # parameter for iso basis size
        type_iso="Fourier s+c" # type of basis for iso
        smooth = True # apply smoothing of basis functions towards r_smooth
        r_smooth = 6.
        self_contribution = False # whether or not an atom itself is considered contribution to the valence elenctron density
        f_smooth = .1 # fudge factor for smoothing
        usefortran = False

        Nsteps_iso = 10000 #number of steps to to plot rho(r) from r = 0 to r = r_smooth
        save_path_rhos = reference_data_dir+"/Al-Ni.rhos"

        # select weights according to some criterion
        idx = np.nanargmin(logbook["tse"])
        print("Selected iteration {} for output:\nL = {}, mse = {}, tse = {}".format(idx,logbook["L"][idx],logbook["mse"][idx],logbook["tse"][idx]))
        weights = logbook["weights_full"][idx]

        r, rhos = fed.predict.predict_rho_iso(mapper,weights,basis,r_smooth,Nsteps_iso)
        
        # saving final rhos to disk 
        fed.misc.save_regressed_rho(r,rhos,save_path_rhos,lb=0,ub=r_smooth,dft_path=dft_path,logbook=logbook,i=idx,
                                    niter=niter,tol=tol,fix_beta=fix_beta,sequential=sequential,
                                    k_iso=k_iso,type_iso=type_iso,ultra_num=ultra_num,selection=selection,
                                    num_neigh=num_neigh,r_cut=r_cut,aniso=aniso,smooth=smooth,r_smooth=r_smooth,
                                    f_smooth=f_smooth,weights=weights)

        #### IO 
        # source
        selection=[("A","all"),("Ni","all")] # selection criteria for DFT calcs
        split2groups = True # splits the selected group of dft structures into subgroups, i.e. when containing Al and Ni the groups are Al only, Ni only and Ni-Al only (no pure structures)
        save = True # EAM_setup: if True data for regression will be written to disk as *.pickl file(s), otherwise it will be returned

        # path to file which contains results from edensity regression
        load_path_rhos = reference_data_dir+"/Al-Ni.rhos"

        # target
        dump_path = reference_data_dir
        dump_fname = "setup_plain_normed" # suffix will be ".pckl", if it's not present it will be added (ending on 0 means no forcing rho->0 for r=0, ending on 1 means forcing rho->0 for r=0)

        #### settings
        # info for bonds
        num_neigh = None # number of neighoring atoms for (ghost) atom
        r_cut = 6.

        # info for basis set
        k_pair = 10 # parameter for basis size of pair energy functions
        k_emb = 10 # parameter for basis size of embedding energy functions
        smooth_emb = True
        smooth_pair = True 
        type_pair="Fourier" # type of basis for pair energy functions
        type_emb="Fourier" # type of basis for embedding energy functions
        r_smooth = 6.
        f_smooth = .01 # fudge factor for smoothing distances
        f_smooth_emb = .01 # fudge factor for smoothing embedding densities
        rho_scaling = 1.

        # info for EAM
        r_lb = 0.
        r_ub = float(r_cut)

        return_rho_bounds = True
        rho_conv_type = "psi" #"psi" #"psi" #"psi2" (shifted + smoothed to 0 in two points), "psi" (shifted + smoothed to zero in one point), None (no smoothing)
        rho_operations = ["shift"]#["absolute","normalize"]
        rho_params = [r_cut,2.]#,0,.25] # parameters for the convolution of rho functions
        N_steps = 100000

        # stuffing all variables into a dict which can be passed more easily
        setup_vars = {"dft_path":dft_path,"selection":selection,"split2groups":split2groups,"save":save,"load_path_rhos":load_path_rhos,
                    "dump_path":dump_path,"dump_fname":dump_fname,"num_neigh":num_neigh,"r_cut":r_cut,"k_pair":k_pair,
                    "k_emb":k_emb,"smooth_emb":smooth_emb,"smooth_pair":smooth_pair,"type_pair":type_pair,"type_emb":type_emb,
                    "r_smooth":r_smooth,"f_smooth":f_smooth,"f_smooth_emb":f_smooth_emb,"rho_scaling":rho_scaling,"r_lb":r_lb,
                    "r_ub":r_ub,"return_rho_bounds":return_rho_bounds,"rho_operations":rho_operations,"rho_params":rho_params,
                    "N_steps":N_steps,"rho_conv_type":rho_conv_type,}

        _ = fe.EAM_setup(show=False,**setup_vars)

    def test_ase_eam_calculator_generation(self):

        print("\nparsing DFT files...")
        gip = parsers.general.GeneralInputParser(verbose=False)
        gip.parse_all(dft_path)
        gip.sort()
        print("done")

        # loading regression results
        path_glipglobs = {"AlNi_0":reference_data_dir+"/plain_normed_glipglobs_Al-1-0_Ni-0-25_0.Es",
                          "AlNi_1":reference_data_dir+"/plain_normed_glipglobs_Al-1-0_Ni-0-25_1.Es",}
        datas_glipglobs = {k:fe.load_data(path) for k,path in path_glipglobs.items()}

        # choosing the parameterization
        idx = 25 # index for the weight to choose
        weights = datas_glipglobs["AlNi_1"]["all_weights"][idx]

        # generating the EAM calculator instance
        Es_path = reference_data_dir+"/plain_normed_glipglobs_Al-1-0_Ni-0-25_1.Es"

        Es = fe.load_data(Es_path)
        data = fe.load_data(curr_dir+Es["fit_data_load_path"][7:])
        # with open(curr_dir+Es["fit_data_load_path"][7:],"rb") as f:
        #     data = pickle.load(f)
        
        needed_vars = ["smooth_emb","smooth_pair","f_smooth_emb","f_smooth","r_smooth","rho_lb","rho_ub",
                "N_steps","rho_scaling","mapper","r_lb","r_ub","rho_dict","r_cut"]
        params = dict()
        for n in needed_vars:
            try:
                params[n] = Es[n]
            except:
                params[n] = data[n]
        
        _params = copy.deepcopy(params)
        mapper = _params.pop("mapper")
        rho_dict = _params.pop("rho_dict")
        rho_lb = _params.pop("rho_lb")
        rho_ub = _params.pop("rho_ub")

        # number of basis functions for each pair and embedding energy function
        Mr, Mrho = len(list(mapper["pair"].values())[0]), len(list(mapper["pair"].values())[0])

        # basis functions
        kappa_r = np.arange(Mr) * np.pi / params["r_cut"]
        kappa_rho = np.arange(Mrho) * np.pi / params["rho_scaling"]
        taper_fun_r = fe.taper_fun_wrapper(_type="x4ge",a=params["r_cut"],b=params["f_smooth"])
        taper_fun_emb = fe.taper_fun_wrapper(_type="x4le",a=0,b=params["f_smooth_emb"])

        basis_r = [fe.DistanceCosTapering_basis(_kappa_r,taper_fun_r) \
                for _kappa_r in kappa_r]
        basis_rho = [fe.DistanceCosTapering_basis(_kappa_rho,taper_fun_emb) \
                    for _kappa_rho in kappa_rho]
        basis_r_1stder = [fe.DistanceCosTapering_basis_1stDerivative(_kappa_r,taper_fun_r) \
                for _kappa_r in kappa_r]
        basis_rho_1stder = [fe.DistanceCosTapering_basis_1stDerivative(_kappa_rho,taper_fun_emb) \
                for _kappa_rho in kappa_rho]

        # energy functions
        energy_functions = fe.get_splined_energy_functions(params["mapper"], weights, params["rho_lb"],\
                                                        params["rho_ub"], params["rho_dict"], basis_r,\
                                                        basis_rho, basis_r_1stder=basis_r_1stder,\
                                                        basis_rho_1stder=basis_rho_1stder, show=False, figsize=(7,5), **_params)

        # creating an ase EAM calculator instance
        eam_pot = fe.generate_EAM_calculator(energy_functions, N_steps=params["N_steps"],
                                        r_cut=params["r_cut"])

        # testing energy and force calculation against ase
        ase_energies, ase_forces = fe.nonlin.get_ase_energies_forces(gip,eam_pot)
        
        self.assertEqual(len(ase_energies),len(Es["regressed_e"]))
        np.testing.assert_array_almost_equal(ase_energies,Es["regressed_e"])
        self.assertEqual(len(ase_forces),len(Es["regressed_f"]))

        np.testing.assert_array_almost_equal(ase_energies,Es["regressed_e"])
        for i in range(len(ase_forces)):
            np.testing.assert_array_almost_equal(ase_forces[i],Es["regressed_f"][i])

def get_suite():
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromTestCase(EAMRegressPipelineTest)]
    return suites

if __name__ == "__main__":
    unittest.main()