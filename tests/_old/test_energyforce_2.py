"""
2) test_energyforce_2.py: reproducibility of energies and forces -> weights based v spline based v eam.alloy based
"""

import unittest
import pickle
import sys, os
import numpy as np
from ase.calculators.eam import EAM

class EnergyForceFixture(unittest.TestCase):

	def setUp(self):
		import parsers
		import fitenergy as fe
		import fitelectrondensity as fed

		############ setting up parameters        
		pre_path = "." if curr_dir[-1] == "tests" else "./tests"
		
		# info loading from disk
		self.Es_path = pre_path + "/unittest_files/test_energyforce.Es"
		self.eam_path = pre_path + "/unittest_files/test_energyforce.eam.alloy"
		self.load_path_data = pre_path + "/unittest_files/test_energyforce.pckl"
		self.dft_path = pre_path + "/unittest_files/ideal_dis_Ni3Al/"

		############# doing a bunch of preparing calculations
		self.Es = fe.nonlin.load_Es(self.Es_path)
		fe.nonlin.write_EAM_setfl_file(self.eam_path,self.Es)
		self.eam_pot = EAM(potential=self.eam_path)

		self.data = fe.nonlin.load_data_for_fit(self.load_path_data)

		gip = parsers.general.GeneralInputParser()
		gip.parse_all(self.dft_path)
		gip.sort()

		self.params = self.data["params"]
		self.mapper = self.data["mapper"]
		self.weights = self.Es["weights"]

		# calculate weight based energies and forces
		if self.Es["smooth_emb"]:
		    emb_n = [np.array([[self.params["fprho_n"][s][n][j] * (self.params["psiprho"][s][n]*self.params["coskrho"][s][:,n] - self.params["psirho"][s][n] * self.params["ksinkrho"][s][:,n]) \
		            for j in range(3)] for n in range(self.params["N_atoms"][s])])\
		            for s in range(self.params["N_bonds"])] #(Natoms,Nk,dim)
		    emb_i = [[np.array([[self.params["fprho_i"][s][n][i][j] * (self.params["psiprho"][s][self.params["neigh_idx_super"][s][n][i]]*self.params["coskrho"][s][:,self.params["neigh_idx_super"][s][n][i]] - self.params["psirho"][s][self.params["neigh_idx_super"][s][n][i]]*self.params["ksinkrho"][s][:,self.params["neigh_idx_super"][s][n][i]])\
		            for j in range(3)] for i in range(self.params["fprho_i"][s][n].shape[0])]) \
		            for n in range(self.params["N_atoms"][s])] \
		            for s in range(self.params["N_bonds"])]
		else:
		    emb_n = [np.array([[self.params["fprho_n"][s][n][j] * ( - self.params["ksinkrho"][s][:,n]) \
		                for j in range(3)] for n in range(self.params["N_atoms"][s])])\
		                for s in range(self.params["N_bonds"])] #(Natoms,Nk,dim)
		    emb_i = [[np.array([[self.params["fprho_i"][s][n][i][j] * (- self.params["ksinkrho"][s][:,self.params["neigh_idx_super"][s][n][i]])\
		                for j in range(3)] for i in range(self.params["fprho_i"][s][n].shape[0])]) \
		                for n in range(self.params["N_atoms"][s])] \
		                for s in range(self.params["N_bonds"])]
		pair_n = [[np.array([[self.params["r_vec"][s][n][i][j] * (self.params["psipr"][s][n][i]*self.params["coskr"][s][n][:,i] - self.params["psir"][s][n][i]*self.params["ksinkr"][s][n][:,i])\
		                for j in range(3)] for i in range(self.params["r_vec"][s][n].shape[0])]) \
		                for n in range(self.params["N_atoms"][s])] \
		                for s in range(self.params["N_bonds"])]
		
		pair_map, emb_neigh_map, emb_map = fe.nonlin.get_mappers(self.params,self.mapper)
		self.W_emb,self.W_pair,self.W_emb_neigh,self.W_pair_neigh = fe.nonlin.setup_weights(self.weights,self.params,self.mapper,emb_map,pair_map,emb_neigh_map)    
		self.e_fun = fe.nonlin.calculate_all_energies_fast_new
		
		if self.Es["force_analytic"]:
		    self.f_fun = fe.nonlin.calculate_all_forces_fast_new
		else:
		    raise NotImplementedError
		    #self.f_fun = get_forces_from_splines_fast
		
		
		self.energies_w = self.e_fun(self.weights,self.W_emb,self.W_pair,self.params)
		self.forces_w = self.f_fun(self.weights,self.W_emb,self.W_pair_neigh,
		    self.W_emb_neigh,self.params,emb_n,emb_i,pair_n)
		self.energies_w = np.array(self.energies_w)
		self.forces_w = [np.around(np.array(v),decimals=6) for v in self.forces_w]

		# calculate energies and forces via ase
		self.energies_ase, self.forces_ase = fe.nonlin.get_ase_energies_forces(gip,self.eam_pot)
		self.energies_ase = np.array(self.energies_ase)
		self.forces_ase = [np.around(np.array(v),decimals=6) for v in self.forces_ase]

	def tearDown(self):
		self.Es = None
		self.data = None

class Test_energyforce(EnergyForceFixture):

	def test0_energy(self):
		# compare energies
		np.testing.assert_almost_equal(self.energies_w,self.energies_ase,decimal=4)

	def test1_forces(self):
		# comparing forces
		for i in range(len(self.forces_w)):
			print("\n{}".format(self.forces_w[i]-self.forces_ase[i]))
			np.testing.assert_almost_equal(self.forces_w[i],self.forces_ase[i],decimal=2)

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
