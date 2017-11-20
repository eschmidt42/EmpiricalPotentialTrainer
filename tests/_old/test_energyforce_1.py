"""
test_energyforce_1.py: reproducibility of EAM_setup.py sanity checking the *.pckl file
"""

import unittest
import pickle
import sys, os
import numpy as np

class EnergyForceFixture(unittest.TestCase):

	def setUp(self):
		import parsers
		import fitenergy as fe
		import fitelectrondensity as fed
	
		############ setting up parameters
		
		pre_path = "." if curr_dir[-1] == "tests" else "./tests"
		
		# info loading from disk
		dft_path = pre_path+"/unittest_files/ideal_dis_Ni3Al/" #small, small_dis, small_dis_singleMD
		self.load_path_data = pre_path+"/unittest_files/test_energyforce.pckl"
		load_path_rhos = pre_path+"/unittest_files/test_energyforce.rhos"

		self.rho_scaling = 1.
		self.seed = 42
		np.random.seed(seed=self.seed)

		# info for bonds
		self.num_neigh = None # number of neighoring atoms for (ghost) atom
		self.r_cut = 6.
		self.aniso = False
		self.ultra_num = None #number of additional super cells in each sim box vector direction
		#selection=[("Al_4d","all",),("Al_3d","all",),("Al_dis","all"), ("Al_100","all"),("Al_300","all"),("A1_c","all"),("A2_c","all"),("A3_c","all")] # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
		self.selection=[("A","all"),("Ni","all")]#[("A1","all"),("A2","all"),("A3","all")]#,("A1_c","all"),("A2_c","all"),("A3_c","all")]

		# info for basis set
		self.k_pair = 25 # parameter for basis size of pair energy functions
		self.k_emb = 30 # parameter for basis size of embedding energy functions
		self.smooth_emb = False
		self.smooth_pair = True 
		self.type_pair="Fourier" # type of basis for pair energy functions
		self.type_emb="Fourier" # type of basis for embedding energy functions
		self.r_smooth = 6.
		self.f_smooth = .01 # fudge factor for smoothing distances
		self.f_smooth_emb = .01 # fudge factor for smoothing embedding densities

		# info for EAM
		self.r_lb=0.
		self.r_ub=float(self.r_cut)
		
		return_rho_bounds = True
		rho_conv_type = None #"psi", None
		rho_operations = []#["absolute","normalize"]
		rho_params = [self.r_cut,.01] # parameters for the convolution of rho functions

		############# doing a bunch of preparing calculation

		# parsing DFT files
		gip = parsers.general.GeneralInputParser()
		gip.parse_all(dft_path)
		gip.sort()

		# loading previously created rhos
		self.rho_dict = fed.misc.load_regressed_rho(load_path_rhos,operations=rho_operations,
		    return_bounds=False,conv=rho_conv_type,params=rho_params)

		# get observation input and output: X, t    
		bonds = fe.nonlin.get_observations(gip,self.rho_dict,ultra_num=self.ultra_num,
		    num_neigh=self.num_neigh,r_cut=self.r_cut,verbose=False,selection=self.selection)
		
		#print("pos {}".format([v.super_pos for v in bonds]))

		a0 = [bond.box[0,0] for bond in bonds]

		self.t = fe.nonlin.get_all_targets(bonds)
		
		self.X, self.rho_lb, self.rho_ub = fe.nonlin.get_all_sources(bonds,
		    return_rho_bounds=return_rho_bounds)
		
		# rescale rho(r) and embedding densities by the largest observed embedding density
		obs_emb = np.amax([v for v2 in self.X["density"] for v in v2])
		self.X["density"] = [v/obs_emb for v in self.X["density"]]
		#max_obs_emb = float(max(rho_ub.values()))
		for _s in self.rho_dict["rhos"].keys():
		    self.rho_dict["rhos"][_s] /= obs_emb
		    rho_min = min([np.amin(v) for v in self.X["density"]])
		    self.rho_lb[_s] = 0 if rho_min > 0 else rho_min
		    self.rho_ub[_s] = 2*max([np.amax(v) for v in self.X["density"]])

		# parameter mapping and pre-calculation of values for subsequent optimization of model parameters
		self.mapper = fe.nonlin.get_mapper(self.k_pair,self.k_emb,self.X)
		
		
		self.params = fe.nonlin.get_precomputed_energy_force_values(self.X,self.f_smooth,
		    self.r_smooth,self.rho_dict,type_pair=self.type_pair,type_emb=self.type_emb,
		    smooth_emb=self.smooth_emb,smooth_pair=self.smooth_pair,rho_scaling=self.rho_scaling,
		    k_emb=self.k_emb,k_pair=self.k_pair,f_smooth_emb=self.f_smooth_emb)
		        
		# initializing model parameters
		self.x0 = fe.nonlin.get_initial_weights(self.mapper)
		
		if self.f_smooth_emb is None:
		    self.f_smooth_emb = self.f_smooth
		
		# loading reference data
		self.ref_data = fe.nonlin.load_data_for_fit(self.load_path_data)

	def teardown(self):
		self.X = None
		self.params = None
		self.mapper = None

class Test_energyforce(EnergyForceFixture):
    
    def test0(self):
                
        # t
        t = self.ref_data["t"]
        for i in range(len(t["forces"])):
            np.testing.assert_almost_equal(t["forces"][i],self.t["forces"][i])
        np.testing.assert_almost_equal(t["energy"],self.t["energy"])
        
        # X
        skeys = sorted(self.ref_data["X"].keys())
        for i,k in enumerate(skeys): # looping the keys
            #print("k {}".format(k))
            if isinstance(self.X[k],list):
                for j in range(len(self.X[k])):# looping structures
                    if k in ["density","r","r_vec"]:
                        for j1 in range(len(self.X[k][j])): # looping atoms
                            np.testing.assert_allclose(self.X[k][j][j1],self.ref_data["X"][k][j][j1])
                    else:
                        self.assertEqual(self.X[k][j],self.ref_data["X"][k][j])
            else:
                self.assertEqual(self.X[k],self.ref_data["X"][k])
              
        # params
        skeys = sorted(self.params.keys())
        for i,k in enumerate(skeys): # looping the keys
            try:
                if k in ["coskr","fprho_i","ksinkr","psipr","psir","r_vec"]:
                    for j0 in range(len(self.params[k])):
                        for j1 in range(len(self.params[k][j0])):
                            np.testing.assert_allclose(self.params[k][j0][j1],self.ref_data["params"][k][j0][j1])
                elif k in ["ksinkrho","coskrho","fprho_n","psiprho","psirho"]:
                    for j0 in range(len(self.params[k])):
                        np.testing.assert_allclose(self.params[k][j0],self.ref_data["params"][k][j0])
                elif k in ["pair_species","neigh_idx_super","emb_species"]:
                    for j0 in range(len(self.params[k])):
                        for j1 in range(len(self.params[k][j0])):
                            self.assertEqual(self.params[k][j0][j1],self.ref_data["params"][k][j0][j1])
                elif k in ["species"]:
                    for j0 in range(len(self.params[k])):
                        self.assertEqual(self.params[k][j0],self.ref_data["params"][k][j0])
                else:        
                    np.testing.assert_allclose(self.params[k],self.ref_data["params"][k])
            except:
                print("actual {}\nref {}".format(self.params[k],self.ref_data["params"][k]))
                print("i {} k {}".format(i,k))
                raise
        
        # mapper
        for k in self.mapper:
            for k2 in self.mapper[k]:
                np.testing.assert_almost_equal(self.mapper[k][k2],self.ref_data["mapper"][k][k2])
        
        
        # rho_dict
        skeys = sorted(self.ref_data["rho_dict"].keys())
        for k in skeys:
            if k in ["logbook"]: continue
            try:
                if k in ["r"]:
                    np.testing.assert_allclose(self.rho_dict[k],self.ref_data["rho_dict"][k])
                elif k == "rhos":
                    for k2 in self.ref_data["rho_dict"][k]:
                        np.testing.assert_allclose(self.rho_dict[k][k2],self.ref_data["rho_dict"][k][k2])
                else:
                    self.assertEqual(self.rho_dict[k],self.ref_data["rho_dict"][k])
            except:
                print("k {}\n{}\n{}".format(self.rho_dict[k],self.ref_data["rho_dict"][k],k))
                raise
        
        # rho_lb and rho_ub
        for k in self.ref_data["rho_lb"]:
            self.assertEqual(self.rho_lb[k],self.ref_data["rho_lb"][k])
        for k in self.ref_data["rho_ub"]:
            self.assertEqual(self.rho_ub[k],self.ref_data["rho_ub"][k])

        # x0
        np.testing.assert_allclose(self.x0,self.ref_data["x0"])

        # bunch of weird things
        bowt = ["rho_scaling","seed","smooth_emb","smooth_pair","f_smooth",
                "f_smooth_emb","r_smooth","r_lb","r_ub","r_cut","num_neigh",
                "aniso","ultra_num","selection","k_pair","k_emb","type_pair",
                "type_emb"]
        for k in bowt:
            try:
                self.assertEqual(getattr(self,k),self.ref_data[k])
            except:
                print("k {}".format(k))
                raise
        

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

def TestSuite():
	potty_imports()
	suite = unittest.TestSuite()
	suite.addTest(unittest.makeSuite(Test_energyforce))
	return suite

def main():
	potty_imports()
	unittest.main()

if __name__ == "__main__":
    main()
