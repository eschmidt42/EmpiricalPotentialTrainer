"""
unittests for fitelectrondensity.linear_model.py containing in scikit-learn style
implementation of Relevance Vector Machine based regression plus 
helper functions.

python test_linear_model.py --write-all # creates reference data files
python test_linear_model.py             # performs tests

Eric Schmidt
e.schmidt@cantab.net
2017-10-20
"""
from __future__ import print_function

import unittest
from sklearn import utils, preprocessing
import sklearn, time, sys, os
from tdda.referencetest import ReferenceTestCase
sys.path.append("..")

from fitelectrondensity.linear_model import fun_wrapper, dis_wrapper, cheb_wrapper, \
    FourierFeatures, GaussianFeatures, ChebyshevFeatures, RelevanceVectorMachine, \
    distribution_wrapper, repeated_regression, print_run_stats

import numpy as np
import pandas as pd
from scipy import stats

seed = 42
np.random.seed(seed)
x = np.linspace(-np.pi,np.pi,100)
x_pred = np.linspace(-np.pi,np.pi,200)
epsilon = stats.norm(loc=0,scale=0.01)
t = np.exp(-x**2) + epsilon.rvs(size=x.shape[0])
k = 10

init_beta = distribution_wrapper(stats.halfnorm(scale=1),size=1,single=True)
init_alphas = distribution_wrapper(stats.halfnorm(scale=1),single=False)
Nruns = 5
tfun = lambda x: np.sin(x) + np.cos(2.*x)

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

class DesignMatrixTestCase(ReferenceTestCase):
    """Tests the setup of various design matrices.

    Notes
    -----
    Nothing fancy. It is simply tested if different design matrices
    can be set up or not. No reference comparison is made.
    """

    def test_polynomial_design_matrix_setup(self):
        """Polynomial design matrix including interaction terms.
        """
        trafo = preprocessing.PolynomialFeatures(k)
        X = trafo.fit_transform(x.reshape((-1,1)))
        
        df = pd.DataFrame(data=X)
        df.columns = [str(v) for v in df.columns]
        self.assertDataFrameCorrect(df,reference_data_dir+'/polynomial.csv',)
    
    def test_Fourier_design_matrix_setup(self):
        """Fourier design matrix.
        """
        trafo = FourierFeatures(k=k)
        X = trafo.fit_transform(x.reshape((-1,1)))

        df = pd.DataFrame(data=X)
        df.columns = [str(v) for v in df.columns]
        self.assertDataFrameCorrect(df,reference_data_dir+'/fourier.csv')

    def test_Gauss_design_matrix_setup(self):
        """Gauss design matrix.
        """
        trafo = GaussianFeatures(k=k,mu0=0,dmu=.25,scale=1.)
        X = trafo.fit_transform(x.reshape((-1,1)))
        
        df = pd.DataFrame(data=X)
        df.columns = [str(v) for v in df.columns]
        self.assertDataFrameCorrect(df,reference_data_dir+'/gauss.csv')

    def test_Chebyshev_design_matrix_setup(self):
        """Chebyshev design matrix.
        """
        trafo = ChebyshevFeatures(k=k)
        X = trafo.fit_transform(x.reshape((-1,1)))

        df = pd.DataFrame(data=X)
        df.columns = [str(v) for v in df.columns]
        self.assertDataFrameCorrect(df,reference_data_dir+'/chebyshev.csv')

class RVMTestCase(unittest.TestCase):
    """Tests the RVM regression.

    Notes
    -----
    Just runs RVM regression, no comparison.
    """

    def setUp(self):
        trafo = FourierFeatures(k=k)
        self.X = trafo.fit_transform(x.reshape((-1,1)))

    def test_manual_hyperparameters(self):
        
        init_beta = 1./ np.var(t) # (that's the default start)
        init_alphas = np.ones(self.X.shape[1])
        init_alphas[1:] = np.inf
        
        model = RelevanceVectorMachine(n_iter=50,verbose=False,compute_score=True,init_beta=init_beta,
                               init_alphas=init_alphas)
        model.fit(self.X,t)
        y, yerr = model.predict(self.X,return_std=True)

    def test_semimanual_hyperparameters(self):
        
        np.random.seed(seed)
        init_beta = stats.halfnorm(scale=1).rvs(size=1)[0]
        init_alphas = stats.halfnorm(scale=1).rvs(size=self.X.shape[1])
        
        model = RelevanceVectorMachine(n_iter=50,verbose=False,compute_score=True,init_beta=init_beta,
                               init_alphas=init_alphas)
        model.fit(self.X,t)
        y, yerr = model.predict(self.X,return_std=True)

    def test_random_hyperparameters(self):
        np.random.seed(seed)
        init_beta = distribution_wrapper(stats.halfnorm(scale=1),single=True)
        init_alphas = distribution_wrapper(stats.halfnorm(scale=1),single=False)
        
        model = RelevanceVectorMachine(n_iter=50,verbose=False,compute_score=True,init_beta=init_beta,
                               init_alphas=init_alphas)
        model.fit(self.X,t)
        y, yerr = model.predict(self.X,return_std=True)

    def test_rerun_regressions(self):
        np.random.seed(seed)
        trafo = FourierFeatures(k=k)
        base_trafo = trafo.fit_transform
        
        model_type = RelevanceVectorMachine
        model_kwargs = dict(n_iter=50,verbose=False,compute_score=True,init_beta=init_beta,
                            init_alphas=init_alphas)
        
        runtimes, coefs = repeated_regression(x,base_trafo,model_type,t=None,tfun=tfun,epsilon=epsilon,
                                            model_kwargs=model_kwargs,Nruns=Nruns,return_coefs=True)
        print_run_stats(base_trafo,x,runtimes,coefs,Nruns,show_coefs=False)
        
    def test_multiple_basis_and_training_set_sizes(self):
        np.random.seed(seed)
        Ns = [50, 100,] # triaing set sizes
        Ms = [5, 10,] # basis set size

        epsilon = stats.norm(loc=0,scale=0.01)
        tfun = lambda x: np.sin(x) + np.cos(2.*x)

        init_beta = distribution_wrapper(stats.halfnorm(scale=1),size=1,single=True)
        init_alphas = distribution_wrapper(stats.halfnorm(scale=1),single=False)

        for N in Ns:
            t_est, t_err = [], []
            for M in Ms:
                x = np.linspace(0,1,N)
                k = M
                
                trafo = FourierFeatures(k=k)
                base_trafo = trafo.fit_transform
                
                model_type = RelevanceVectorMachine
                model_kwargs = dict(n_iter=50,verbose=False,compute_score=True,init_beta=init_beta,
                                    init_alphas=init_alphas)

                runtimes, coefs = repeated_regression(x,base_trafo,model_type,t=None,tfun=tfun,epsilon=epsilon,
                                                    model_kwargs=model_kwargs,Nruns=Nruns,return_coefs=True)
                #print_run_stats(base_trafo,x,runtimes,coefs,Nruns)
                t_est.append(runtimes.mean())
                t_err.append(runtimes.std(ddof=1)*2)
            print("\ntime for N = {}:".format(N))
            for est, err in zip(t_est,t_err):
                print("    estimate = {:.4f}s, 2*std = {:.4f}s".format(est,2*err))

def get_suite():
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromTestCase(DesignMatrixTestCase),
              loader.loadTestsFromTestCase(RVMTestCase)]
    return suites

if __name__ == "__main__":
    ReferenceTestCase.main()