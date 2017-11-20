"""
This is a unittest/tdda script for setting up design matrices
from DFT structures and corresponding electron density fields.
The tests just make sure the construction works. No proof of 
correctness.

In order to refresh the reference data files run:
>>> python test_edens_designmatrix.py --write-all

Just test using existing reference data files:
>>> python test_edens_designmatrix.py

Notes
-----
The tests using fortran are expected to fail on Windows or 
under Linux if the 'build.sh' in the 'fortran' dir was not executed.

Eric Schmidt
2017-10-20
"""
from __future__ import print_function
import os, sys
import unittest
from tdda.referencetest import ReferenceTestCase
import pandas as pd
import numpy as np

sys.path.append("..")
import parsers
import fitelectrondensity as fed

# identify platform
if os.name == "posix":
    win = False
else:
    win = True

# current directory of this script
curr_dir = os.path.abspath(os.path.dirname(__file__))
if win:
   curr_dir = curr_dir.replace("\\","/")

dft_path = curr_dir+"/unittest_files/castep/"

# reference dir where files for comparison are stored, relative to script
reference_data_dir = curr_dir+"/unittest_files/reference/"
if win:
    reference_data_dir = reference_data_dir.replace("\\","/")

# parsing DFT files
gip = parsers.general.GeneralInputParser(verbose=False)
gip.parse_all(dft_path)
gip.sort()

# design matrix specs
## info for bonds
num_neigh = None # number of neighoring atoms for (ghost) atom
r_cut = 6. # neighborhood search and isotropic cutoff radius
ultra_num = None #number of additional super cells in each sim box vector direction
selection=("atom",.5,"r") # first tuple entry specifies selection type, following fix specifics, i.e.("random",10.) mean select 10% of all points uniformly at random 
seed = 42
np.random.seed(seed=seed)

## info for basis set
ani_type = "MEAM" # "MEAM", BOP, BOP-r or BOP-invr, "R", "BOP4atoms", "BOP4atoms-ref", "BOP4atoms-ref2", "BOP-taper1"
    # useful l values: 3, 4, 5 (bad), 6 (very bad), 7 (bad), 8 (very bad), 9, 10, 11 (bad), 12 (very bad), 42
ani_specification = {"l_range": np.array([4,6,8],dtype=int),
                     "usage":"everything","r_ani":4.} #"usage": "everything" (is "usage" is not present this is assumed as default, all" and aphabetically sorted elements), "all" (only use q obtained for "all"), "Ni" (only usq q obtained for "Ni")
k_iso=50 # parameter for iso basis size
k_ani= 5 # parameter for ani basis size
type_iso="Fourier" # type of basis for iso
type_ani="Fourier" # type of basis for ani
smooth = True # apply smoothing of basis functions towards r_smooth
r_smooth = r_cut
r_scale = r_cut
f_smooth = .1 # fudge factor for smoothing

# RAM cap
max_memory = 5. # [GB] - cap on maximum size of most memory consuming variable in rvm.get_updated_hyperparameters

ref_name = lambda s,m: "glm_{}_{}.csv".format(s,m)
tmp_name = lambda s,m: "tmp_glm_{}_{}.csv".format(s,m)

# fortran compilation status
fortran_compiled = False

def get_dataframe(data):
    df = pd.DataFrame(data=data)
    df.columns = [str(v) for v in df.columns]
    df = df.round(decimals=5)
    return df

class DesignMatrixTests(ReferenceTestCase):
    
    def test_selfcontribution_iso_nofortran(self):
        
        aniso = False
        self_contribution = True # whether or not an atom itself is considered contribution to the valence elenctron density
        usefortran = False

        # set up bonds
        bonds = []
        for tmp_gip in gip:
            tmp_bonds = fed.misc.get_observations(tmp_gip, ultra_num=ultra_num, 
                                                num_neigh=num_neigh, r_cut=r_cut, aniso=aniso,
                                                verbose=False, selection=selection, seed=seed,)
            bonds.extend(tmp_bonds)

        # setup the basis and the corresponding mapper
        basis, mapper = fed.rvm.get_basis(gips=[gip], k_iso=k_iso, k_ani=k_ani,\
                type_iso=type_iso, type_ani=type_ani, smooth=smooth, r_smooth=r_smooth,\
                f_smooth=f_smooth, verbose=False,self_contribution=self_contribution, aniso=aniso,\
                rcut=r_cut, num_neigh=num_neigh, r_scale=r_scale)

        # setup the design matrix
        Phi, t = fed.rvm.get_design_matrix(bonds, basis, mapper, verbose=False,
                                        return_t=True, seed=seed, usefortran=usefortran)

        # test design matrix
        test_against = reference_data_dir+"/"+ref_name("designmatrix","self_iso_nofortran",)
        df_Phi = get_dataframe(np.array(Phi))
        self.assertDataFrameCorrect(df_Phi,test_against)

        # test target values
        test_against = reference_data_dir+"/"+ref_name("t","self_iso_nofortran",)
        df_t = get_dataframe(np.array(t))
        self.assertDataFrameCorrect(df_t,test_against)

    def test_nocontribution_iso_nofortran(self):
        
        aniso = False
        self_contribution = False # whether or not an atom itself is considered contribution to the valence elenctron density
        usefortran = False

        # set up bonds
        bonds = []
        for tmp_gip in gip:
            tmp_bonds = fed.misc.get_observations(tmp_gip, ultra_num=ultra_num, 
                                                num_neigh=num_neigh, r_cut=r_cut, aniso=aniso,
                                                verbose=False, selection=selection, seed=seed)
            bonds.extend(tmp_bonds)

        # setup the basis and the corresponding mapper
        basis, mapper = fed.rvm.get_basis(gips=[gip], k_iso=k_iso, k_ani=k_ani,\
                type_iso=type_iso, type_ani=type_ani, smooth=smooth, r_smooth=r_smooth,\
                f_smooth=f_smooth, verbose=False,self_contribution=self_contribution, aniso=aniso,\
                rcut=r_cut, num_neigh=num_neigh, r_scale=r_scale)

        # setup the design matrix
        Phi, t = fed.rvm.get_design_matrix(bonds, basis, mapper, verbose=False,
                                        return_t=True, seed=seed, usefortran=usefortran)

        # test design matrix
        test_against = reference_data_dir+"/"+ref_name("designmatrix","noself_iso_nofortran",)
        df_Phi = get_dataframe(np.array(Phi))
        self.assertDataFrameCorrect(df_Phi,test_against)

        # test target values
        test_against = reference_data_dir+"/"+ref_name("t","noself_iso_nofortran",)
        df_t = get_dataframe(np.array(t))
        self.assertDataFrameCorrect(df_t,test_against)

    def test_selfcontribution_ani_nofortran(self):
        
        aniso = True
        self_contribution = True # whether or not an atom itself is considered contribution to the valence elenctron density
        usefortran = False

        # set up bonds
        bonds = []
        for tmp_gip in gip:
            tmp_bonds = fed.misc.get_observations(tmp_gip, ultra_num=ultra_num, 
                                                num_neigh=num_neigh, r_cut=r_cut, aniso=aniso,
                                                verbose=False, selection=selection, seed=seed,
                                                ani_type=ani_type, ani_specification=ani_specification)
            bonds.extend(tmp_bonds)

        # setup the basis and the corresponding mapper
        basis, mapper = fed.rvm.get_basis(gips=[gip], k_iso=k_iso, k_ani=k_ani,\
                type_iso=type_iso, type_ani=type_ani, smooth=smooth, r_smooth=r_smooth,\
                f_smooth=f_smooth, verbose=False,self_contribution=self_contribution, aniso=aniso,\
                rcut=r_cut, num_neigh=num_neigh, ani_type=ani_type, ani_specification=ani_specification,\
                r_scale=r_scale)

        # setup the design matrix
        Phi, t = fed.rvm.get_design_matrix(bonds, basis, mapper, verbose=False,
                                        return_t=True, seed=seed, usefortran=usefortran)

        # test design matrix
        test_against = reference_data_dir+"/"+ref_name("designmatrix","self_ani_nofortran",)
        df_Phi = get_dataframe(np.array(Phi))
        self.assertDataFrameCorrect(df_Phi,test_against)

        # test target values
        test_against = reference_data_dir+"/"+ref_name("t","self_ani_nofortran",)
        df_t = get_dataframe(np.array(t))
        self.assertDataFrameCorrect(df_t,test_against)

    def test_selfcontribution_iso_fortran(self):
        if fortran_compiled:
            aniso = False
            self_contribution = True # whether or not an atom itself is considered contribution to the valence elenctron density
            usefortran = True

            # set up bonds
            bonds = []
            for tmp_gip in gip:
                tmp_bonds = fed.misc.get_observations(tmp_gip, ultra_num=ultra_num, 
                                                    num_neigh=num_neigh, r_cut=r_cut, aniso=aniso,
                                                    verbose=False, selection=selection, seed=seed)
                bonds.extend(tmp_bonds)

            # setup the basis and the corresponding mapper
            basis, mapper = fed.rvm.get_basis(gips=[gip], k_iso=k_iso, k_ani=k_ani,\
                    type_iso=type_iso, type_ani=type_ani, smooth=smooth, r_smooth=r_smooth,\
                    f_smooth=f_smooth, verbose=False,self_contribution=self_contribution, aniso=aniso,\
                    rcut=r_cut, num_neigh=num_neigh, r_scale=r_scale)

            # setup the design matrix
            Phi, t = fed.rvm.get_design_matrix(bonds, basis, mapper, verbose=False,
                                            return_t=True, seed=seed, usefortran=usefortran)

            # test design matrix
            test_against = reference_data_dir+"/"+ref_name("designmatrix","self_iso_fortran",)
            df_Phi = get_dataframe(np.array(Phi))
            self.assertDataFrameCorrect(df_Phi,test_against)

            # test target values
            test_against = reference_data_dir+"/"+ref_name("t","self_iso_fortran",)
            df_t = get_dataframe(np.array(t))
            self.assertDataFrameCorrect(df_t,test_against)

    def test_selfcontribution_ani_fortran(self):
        if fortran_compiled:
            aniso = True
            self_contribution = True # whether or not an atom itself is considered contribution to the valence elenctron density
            usefortran = True

            # set up bonds
            bonds = []
            for tmp_gip in gip:
                tmp_bonds = fed.misc.get_observations(tmp_gip, ultra_num=ultra_num, 
                                                    num_neigh=num_neigh, r_cut=r_cut, aniso=aniso,
                                                    verbose=False, selection=selection, seed=seed,
                                                    ani_type=ani_type, ani_specification=ani_specification)
                bonds.extend(tmp_bonds)

            # setup the basis and the corresponding mapper
            basis, mapper = fed.rvm.get_basis(gips=[gip], k_iso=k_iso, k_ani=k_ani,\
                    type_iso=type_iso, type_ani=type_ani, smooth=smooth, r_smooth=r_smooth,\
                    f_smooth=f_smooth, verbose=True,self_contribution=self_contribution, aniso=aniso,\
                    rcut=r_cut, num_neigh=num_neigh, ani_type=ani_type, ani_specification=ani_specification,\
                    r_scale=r_scale)

            # setup the design matrix
            Phi, t = fed.rvm.get_design_matrix(bonds, basis, mapper, verbose=False,
                                            return_t=True, seed=seed, usefortran=usefortran)

            # test design matrix
            test_against = reference_data_dir+"/"+ref_name("designmatrix","self_ani_fortran",)
            df_Phi = get_dataframe(np.array(Phi))
            self.assertDataFrameCorrect(df_Phi,test_against)

            # test target values
            test_against = reference_data_dir+"/"+ref_name("t","self_ani_fortran",)
            df_t = get_dataframe(np.array(t))
            self.assertDataFrameCorrect(df_t,test_against)

def get_suite():
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromTestCase(DesignMatrixTests)]
    return suites

if __name__ == "__main__":
    ReferenceTestCase.main()