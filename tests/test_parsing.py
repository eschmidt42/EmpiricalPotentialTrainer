"""
This is a unittest/tdda script for the parsing of DFT files,
just to make sure their parsing works and reproduces earlier
output. No proof of correctness.

Todo: include VASP electron density reading

In order to refresh the reference data files run:
>>> python test_parsing.py --write-all

Just test using existing reference data files:
>>> python test_parsing.py

Eric Schmidt
e.schmidt@cantab.net
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

# identify platform
if os.name == "posix":
    win = False
else:
    win = True

# current directory of this script
curr_dir = os.path.abspath(os.path.dirname(__file__))
if win:
   curr_dir = curr_dir.replace("\\","/")

# location of castep files to parse, relative to script
castep_dft_path = curr_dir+"/unittest_files/castep/"
# location of castep files to parse, relative to script
profess_dft_path = curr_dir+"/unittest_files/profess/"
# location of castep files to parse, relative to script
vasp_dft_path = curr_dir+"/unittest_files/vasp/"

# reference dir where files for comparison are stored, relative to script
reference_data_dir = curr_dir+"/unittest_files/reference/"
if win:
    reference_data_dir = reference_data_dir.replace("\\","/")

def get_dataframe(data):
    df = pd.DataFrame(data=data)
    df.columns = [str(v) for v in df.columns]
    df = df.round(decimals=5)
    return df

class ParseTests(ReferenceTestCase):

    def test_parse_castep(self):
        ref_name = lambda i,prop: "castep_{}_{}.csv".format(i,prop)

        properties = ["positions","species","edensity"]
        gip = parsers.general.GeneralInputParser(verbose=False)
        
        # take care of win nasty slash business (just to be safe)
        if win:
            _reference_data_dir = reference_data_dir.replace("\\","/")
            _castep_dft_path = castep_dft_path.replace("\\","/")
        
        # parse files
        assert os.path.isdir(_castep_dft_path), "{} not a path".format(_castep_dft_path)
        gip.parse_all(_castep_dft_path)
        gip.sort()
        
        # loop through parsed structures
        for i,g in enumerate(gip):
            # loop through some of their properties
            for prop in properties:
                                
                if prop == "edensity":
                    # positions
                    _rname = ref_name(g.name,prop+"_pos")
                    test_against = reference_data_dir+"/"+_rname

                    df1 = get_dataframe(np.array(g[prop]["xyz"]))
                    self.assertDataFrameCorrect(df1,test_against)
                    
                    # electron densities
                    _rname = ref_name(g.name,prop+"_dens")
                    test_against = reference_data_dir+"/"+_rname

                    df2 = get_dataframe(np.array(g[prop]["density"]))
                    self.assertDataFrameCorrect(df2,test_against)
                else:
                    _rname = ref_name(i,prop)
                    test_against = reference_data_dir+"/"+_rname
                    
                    df1 = pd.DataFrame(data=np.array(g[prop]))
                    df1.columns = [str(v) for v in df1.columns]
                    df1 = get_dataframe(np.array(g[prop]))
                    self.assertDataFrameCorrect(df1,test_against)

    def test_parse_profess(self):
        ref_name = lambda i,prop: "profess_{}_{}.csv".format(i,prop)

        properties = ["positions","species","edensity"]
        gip = parsers.general.GeneralInputParser(verbose=False)
        
        # take care of win nasty slash business (just to be safe)
        if win:
            _reference_data_dir = reference_data_dir.replace("\\","/")
            _profess_dft_path = profess_dft_path.replace("\\","/")
        
        # parse files
        assert os.path.isdir(_profess_dft_path), "{} not a path".format(_profess_dft_path)
        gip.parse_all(_profess_dft_path)
        gip.sort()
        
        # loop through parsed structures
        for i,g in enumerate(gip):
            # loop through some of their properties
            
            for prop in properties:
                                
                if prop == "edensity":
                    # positions
                    _rname = ref_name(g.name,prop+"_pos")
                    test_against = reference_data_dir+"/"+_rname

                    df1 = get_dataframe(np.array(g[prop]["xyz"]))
                    self.assertDataFrameCorrect(df1,test_against)
                    
                    # electron densities
                    _rname = ref_name(g.name,prop+"_dens")
                    test_against = reference_data_dir+"/"+_rname

                    df2 = get_dataframe(np.array(g[prop]["density"]))
                    self.assertDataFrameCorrect(df2,test_against)
                else:
                    _rname = ref_name(i,prop)
                    test_against = reference_data_dir+"/"+_rname
                    
                    df1 = get_dataframe(np.array(g[prop]))
                    self.assertDataFrameCorrect(df1,test_against)

    def test_parse_vasp(self):
        """
        This test is for VASP. Note: Currently it's not reading the electron density.
        """
        ref_name = lambda i,prop: "vasp_{}_{}.csv".format(i,prop)

        properties = ["positions","species"]#,"edensity"]
        gip = parsers.general.GeneralInputParser(verbose=False)
        
        # take care of win nasty slash business (just to be safe)
        if win:
            _reference_data_dir = reference_data_dir.replace("\\","/")
            _vasp_dft_path = vasp_dft_path.replace("\\","/")
        
        # parse files
        print("vasp path ",_vasp_dft_path)
        assert os.path.isdir(_vasp_dft_path), "{} not a path".format(_profess_dft_path)
        gip.parse_all(_vasp_dft_path)
        gip.sort()
        
        # loop through parsed structures
        for i,g in enumerate(gip):
            # loop through some of their properties
            
            for prop in properties:
                                
                _rname = ref_name(i,prop)
                test_against = reference_data_dir+"/"+_rname
                
                df1 = get_dataframe(np.array(g[prop]))
                self.assertDataFrameCorrect(df1,test_against)

def get_suite():
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromTestCase(ParseTests)]
    return suites

if __name__ == "__main__":
    ReferenceTestCase.main()