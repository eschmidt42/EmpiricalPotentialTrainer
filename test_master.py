"""
Master script to run all unittests from. Note that some contain 
tdda's ReferenceTestCase subclasses. Reference files shipped
with this module may be outdated and possibly require manual 
execution of scripts with ReferenceTestCase subclasses. Relevant 
scripts: test_parsing.py

Eric Schmidt
e.schmidt@cantab.net
2017-10-21
"""

import tests
import unittest
import sys

if __name__ == "__main__":
        
    # (reference) test cases for parsing
    parse_suites_list = tests.test_parsing.get_suite()

    # (reference) test cases for linear_model regression
    linear_model_suites_list = tests.test_linear_model.get_suite()

    # (reference) test cses for design matrix and target vector generation
    design_suites_list = tests.test_edens_designmatrix.get_suite()

    # (reference) test cases for electron density regression
    design_suites_list = tests.test_edens_regresspredict.get_suite()

    # test cases for eam pipeline
    eam_suites_list = tests.test_eam_regression.get_suite()

    # combine all testcases and run joint suite
    suites = parse_suites_list + design_suites_list \
        + linear_model_suites_list + eam_suites_list
    suite = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner()
    results = runner.run(suite)
    sys.exit(not results)