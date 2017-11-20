"""
Tests the reproducebility of Phi matrix and RVM iteration results for 
pure Python without active self contribution. Testing isotropic only.
"""

import unittest
import time
import numpy as np
import sys
import os

def potty_imports():
    # determining the current dir considering the present operating system
    if "win" in  sys.platform:
        curr_dir = os.getcwd().split("\\")
    elif "linux" in sys.platform:
        curr_dir = os.getcwd().split("/")
    else:
        raise NotImplementedError

    # potty specific
    if curr_dir[-1] == "tests":
        sys.path.insert(0,"../")

    global curr_dir

class DensityRegressionFixture(unittest.TestCase):
    def setUp(self):

        ############ setting up parameters
        import parsers
        import fitelectrondensity as fed

        pre_path = "." if curr_dir[-1] == "tests" else "./tests"

        self.dft_path = pre_path+"/unittest_files/ideal_dis_Ni3Al/" #EV_A1_h_LDA

        # info for bonds
        self.num_neigh = None # number of neighoring atoms for (ghost) atom
        self.r_cut = 6.
        self.aniso = False
        self.ultra_num = None #number of additional super cells in each sim box vector direction
        self.selection=("atom",.5,"r") # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
        self.seed = 42
        np.random.seed(seed=self.seed)

        # info for basis set
        self.k_iso=50 # parameter for iso basis size
        self.k_ani= 5 # parameter for ani basis size
        self.type_iso="Fourier" # type of basis for iso
        self.type_ani="Fourier" # type of basis for ani
        self.smooth = True # apply smoothing of basis functions towards r_smooth
        self.r_smooth = 6.
        self.self_contribution = False # whether or not an atom itself is considered contribution to the valence elenctron density
        self.f_smooth = .1 # fudge factor for smoothing

        # RVM initiate hyper parameters
        self.niter = 1
        self.tol = 1e-6
        self.fix_beta = False
        self.n_steps_beta = 1 # integer specifying every nth step to update beta
        self.sequential = False
        self.stochastic = ("all",) # ("all",), ("uniform",5000) switch for rvm.get_design_matrix, specifies how many and which observations from the ones already processed are actually used in a given run 
        self.niter_stochastic = 1 # number of iterations generating new Phis according to variable 'stochastic' if stochastic[0] != "all"

        # RAM cap
        self.max_memory = 25. # [GB] - cap on maximum size of most memory consuming variable in rvm.get_updated_hyperparameters

        # writing paths
        self.load_path_rhos = pre_path+"/unittest_files/test_density_withoutself.rhos"
        #save_path_predicted = "./predicted_Ni3Al_ideal_dis.pred"
        self.load_path_Phi = pre_path+"/unittest_files/test_density_withoutself.phi"

        self.usefortran = False

        ###### processing 
        # parsing DFT files
        self.gip = parsers.general.GeneralInputParser()
        self.gip.parse_all(self.dft_path)
        self.gip.sort()

        # get observation input and output: X, t    
        t0 = time.time()
        self.bonds = []
        print("Generating bonds for ...")
        for tmp_gip in self.gip:
            print("{}...".format(tmp_gip.get_name()))
            tmp_bonds = fed.misc.get_observations(tmp_gip,ultra_num=self.ultra_num,num_neigh=self.num_neigh,r_cut=self.r_cut,aniso=self.aniso,
                verbose=False,selection=self.selection,seed=self.seed)
            self.bonds.extend(tmp_bonds)
        print("generated bonds {}s...".format(time.time()-t0))
        
        # get basis
        self.basis,self.mapper = fed.rvm.get_basis(gips=[self.gip],k_iso=self.k_iso,k_ani=self.k_ani,\
                type_iso=self.type_iso,type_ani=self.type_ani,smooth=self.smooth,r_smooth=self.r_smooth,\
                f_smooth=self.f_smooth,verbose=True,self_contribution=self.self_contribution,aniso=self.aniso,\
                rcut=self.r_cut,num_neigh=self.num_neigh)
        
        # load things
        self.Phi_ref = fed.misc.load_Phi(self.load_path_Phi)

        self.rho_dict_ref = fed.misc.load_regressed_rho(self.load_path_rhos)
        
        try:
            from fortran.interface import isotropic_phi
            self.fortran_compiled = True
        except:
            self.fortran_compiled = False
        
        if self.stochastic[0] == "all":
            print("\nProcessing all observations during regression...")

            # get design matrix
            self.Phi, self.Phi_it = [], []
            self.t, self.t_it = [], []

            self.Phi, self.t = fed.rvm.get_design_matrix(self.bonds,self.basis,self.mapper,verbose=False,return_t=True,seed=self.seed,usefortran=False)
            
            for _s in self.gip:
                self.Phi_it,self.t_it = fed.rvm.update_design_matrix(basis=self.basis,mapper=self.mapper,\
                        structure=_s,Phi=self.Phi_it,reference_densities=self.t_it,\
                        selection=self.selection,seed=self.seed,stochastic=self.stochastic)
            
        else:
            print("\nProcessing a subset of all observations during regression as specified: {}".format(self.stochastic))
            t0s = time.time()
        
            for i in range(self.niter_stochastic):
                
                print("\nStochastic iteration {}/{}...".format(i+1,self.niter_stochastic))
                # get design matrix
                #self.Phi,self.t = fed.rvm.get_design_matrix(self.bonds,self.basis,self.mapper,verbose=True,\
                #        stochastic=self.stochastic,return_t=True,seed=self.seed,usefortran=False)

                self.Phi,self.t = [],[]
                for _s in self.gip:
                    # this may need to be handled differently
                    self.Phi,self.t = fed.rvm.update_design_matrix(basis=self.basis,mapper=self.mapper,\
                            structure=_s,Phi=self.Phi,reference_densities=self.t,num_neigh=self.num_neigh,\
                            selection=self.selection,seed=self.seed,stochastic=self.stochastic)

    def tearDown(self):
        self.Phi = None
        self.Phi_ref = None
        self.bonds = None

class Test_fitelectrondensity(DensityRegressionFixture):

    def test1_calc_design_matrix(self):
        print("\nTesting the calculated design matrix...")        
        np.testing.assert_almost_equal(self.Phi_it,self.Phi,decimal=6) # Phi generated iteratively for all bonds here vs the simultaneous version
        np.testing.assert_almost_equal(self.Phi,self.Phi_ref,decimal=4) # Phi generated simultaneously for all bonds here vs reference
        print("Difference in Phis {}".format(np.sum(self.Phi-self.Phi_ref)))

    def test2_optimizing_hyperparameters(self):
        import fitelectrondensity as fed
        print("\nTesting the calculated hyperparameters...")
        
        M = self.Phi_ref.shape[1]
        alpha_init = np.ones(M)
        alpha_init[1:] = np.inf
        beta_init = 1e2
        
        t0 = time.time()
        
        logbook = fed.rvm.iterate(self.Phi,self.t,niter=self.niter,verbose=False,
            alpha_init=alpha_init,beta_init=beta_init,tol=self.tol,fix_beta=self.fix_beta,
            sequential=self.sequential,n_steps_beta=self.n_steps_beta,max_memory=self.max_memory,seed=self.seed)
        print("completed RVM regression {}s...".format(time.time()-t0))
        keys = sorted(logbook.keys())

        k = "weights"
        
        for i in range(len(logbook[k])):
            np.testing.assert_almost_equal(np.array(logbook[k][i]),np.array(self.rho_dict_ref["logbook"][k][i]),decimal=3)
    
def TestSuite():
    potty_imports()
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_fitelectrondensity))
    return suite
    
def main():
    potty_imports()
    unittest.main()

if __name__=="__main__":
    main()
    
