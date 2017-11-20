"""
scikit-learn style implementation of Relevance Vector Machine 
based regression plus helper functions and example.

Eric Schmidt
2017-10-12
"""

from __future__ import print_function
from sklearn import linear_model, utils, preprocessing
import sklearn
import numpy as np
from scipy import stats
import time

def fun_wrapper(fun,k):
    def _fun_wrapped(x):
        return fun(x*k)
    return _fun_wrapped

def dis_wrapper(dis):
    def _dis_wrapped(x):
        return dis.pdf(x)
    return _dis_wrapped

def cheb_wrapper(i,k):
    # i = the non-zero coefficient
    # k = the number of coefficients (incl. the bias)
    vec = np.zeros(k)
    vec[i] = 1
    def _cheb_wrapped(x):
        return np.polynomial.chebyshev.chebval(x,vec)
    return _cheb_wrapped


class GaussianFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate Gaussian features.

    Generate a design matrix of k Gaussians starting at mu0, separated 
    by dmu all with the same scale.

    Parameters
    ----------
    k : int, optional, default 10
        The number of Gaussian.
    mu0 : float, optional, default 0
        The starting point for placing the first Gaussian.
    dmu : float, optional, default 1
        The increment to use separating the Gaussians.
    scale : float, optional, default 1
        The scale of all Gaussians.
    include_bias : boolean, optional, default True
        The design matrix includes a bias column if True.

    Example
    --------
    >>> x = np.linspace(-np.pi,np.pi,100)
    >>> trafo = GaussianFeatures(k=30,mu0=-3,dmu=.2)
    >>> X = trafo.fit_transform(x.reshape((-1,1)))
    
    """
    def __init__(self,k=10,mu0=0,dmu=1.,scale=1.,include_bias=True):
        self.k = k
        self.mu0 = mu0
        self.dmu = dmu
        self.scale = scale
        self.include_bias = include_bias
        
    @staticmethod
    def _basis_functions(n_features, k, include_bias=True, mu0=0., dmu=.5, scale=1.):
        """Generates a np.ndarray of Gaussian basis functions.

        Parameters
        ----------
        n_features : int
            number of features for each observation
        k : int
            number of basis functions
        include_bias : boolean, optional, default True
            whether or not to include a bias function (function that returns 1)
        mu0 : float, optional, default 0
            position of the first Gaussian
        dmu : float, optional, default .5
            increment to shift the Gaussians by
        scale : float, optional ,default 1
            scale of all Gaussians

        Returns
        -------
        basis : np.ndarray of callables of shape (k(+1),)
        """
        bias = np.array([lambda x: np.ones(x.shape[0])])
        G = np.array([dis_wrapper(stats.norm(loc=mu0+_k*dmu,scale=scale)) for _k in range(k)])
        
        if include_bias:
            basis = np.concatenate((bias,G))
        else:
            basis = G
        return basis
        
    def fit(self,X,y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = len(self._basis_functions(n_features,self.k, 
                                        self.include_bias, self.mu0, self.dmu, self.scale))
        return self
        
    
    def transform(self,X):
        """Applies the basis functions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_input_features)

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        basis = self._basis_functions(self.n_input_features_, self.k, self.include_bias,
                                     self.mu0, self.dmu, self.scale)
        for i,b in enumerate(basis):
            XP[:,i] = b(X).ravel()
        return XP
    
    def fit_transform(self,X):
        """Calls fit and transform on X.
        """
        self.fit(X)
        return self.transform(X)
    
class FourierFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Creates the design matrix X from x using the Fourier basis set.
    
    Parameters
    ----------
    k : int, optional, default 10
        number of basis functions for both sine and cosine, plus the possible bias
    include_bias : boolean, optional, default True
        whether or not to include a bias function (function that returns 1)

    Example
    -------
    >>> x = np.linspace(-np.pi,np.pi,100)
    >>> trafo = FourierFeatures(k=10)
    >>> X = trafo.fit_transform(x.reshape((-1,1)))
    """
    def __init__(self,k=10,include_bias=True):
        self.k = k
        self.include_bias = include_bias
        
    @staticmethod
    def _basis_functions(n_features, k, include_bias):
        """Generates a np.ndarray of sine and cosine basis functions.

        Parameters
        ----------
        n_features : int
            number of features for each observation
        k : int
            number of basis functions for each sine and cosine
        include_bias : boolean, optional, default True
            whether or not to include a bias function (function that returns 1)

        Returns
        -------
        basis : np.ndarray of callables of shape (2*k(+1),)
        """
        bias = np.array([lambda x: np.ones(x.shape[0])])
        sin = np.array([fun_wrapper(np.sin,_k) for _k in range(1,k)])
        cos = np.array([fun_wrapper(np.cos,_k) for _k in range(1,k)])
        if include_bias:
            basis = np.concatenate((bias,sin,cos))
        else:
            basis = np.concatenate((sin,cos))
        return basis
        
    def fit(self,X,y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = len(self._basis_functions(n_features,self.k,self.include_bias))
        return self
        
    
    def transform(self,X):
        """Applies the basis functions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_input_features)

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        basis = self._basis_functions(self.n_input_features_,self.k,self.include_bias)
        for i,b in enumerate(basis):
            XP[:,i] = b(X).ravel()
        return XP
    
    def fit_transform(self,X):
        """Calls fit and transform on X.
        """
        self.fit(X)
        return self.transform(X)
    
class ChebyshevFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Creates the design matrix X from x using Chebyshev polynomials.
    
    Parameters
    ----------
    k : int, optional, default 10
        number of basis functions , plus the possible bias
    include_bias : boolean, optional, default True
        whether or not to include a bias function (function that returns 1)

    Example
    -------
    >>> x = np.linspace(-np.pi,np.pi,100)
    >>> trafo = ChebyshevFeatures(k=10)
    >>> X = trafo.fit_transform(x.reshape((-1,1)))
    """
    def __init__(self,k=10,include_bias=True):
        self.k = k
        self.include_bias = include_bias
        
    @staticmethod
    def _basis_functions(n_features, k, include_bias):
        """Generates a np.ndarray of Chebyshev polynomials.

        Parameters
        ----------
        n_features : int
            number of features for each observation
        k : int
            number of basis functionse
        include_bias : boolean, optional, default True
            whether or not to include a bias function (function that returns 1)

        Returns
        -------
        basis : np.ndarray of callables of shape (k(+1),)
        """
        bias = np.array([lambda x: np.ones(x.shape[0])])
        T = np.array([cheb_wrapper(_k,k) for _k in range(1,k)])
        if include_bias:
            basis = np.concatenate((bias,T))
        else:
            basis = T
        return basis
        
    def fit(self,X,y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = len(self._basis_functions(n_features,self.k,self.include_bias))
        return self
        
    
    def transform(self,X):
        """Applies the basis functions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_input_features)

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        basis = self._basis_functions(self.n_input_features_,self.k,self.include_bias)
        for i,b in enumerate(basis):
            XP[:,i] = b(X).ravel()
        return XP
    
    def fit_transform(self,X):
        """Calls fit and transform on X.
        """
        self.fit(X)
        return self.transform(X)

def full_weight_vector(w, active, inactive):
    """Returns a zero-padded weights vector for RVM weights.
    
    Parameters
    ----------
    w : float np.ndarray of shape [n_active]
        Weights vector obtained with an RVM containing only non-zero values.
    active : int np.ndarray of shape [n_active]
        Index vector indicating the positions of the 'w' values in the
        full weights vector.
    inactive : int np.ndarray of shape [n_features - n_active]
        Index vector indicating the positions of 0s in the full weights 
        vector.
    
    Returns
    -------
    w_full : float np.ndarray of shape [n_features]
        Full weights vector.
    """
    w_full = np.zeros(len(active)+len(inactive))
    w_full[active] = w
    return w_full

class RelevanceVectorMachine(linear_model.base.LinearModel,sklearn.base.RegressorMixin):
    """Relevance vector machine regression.
    
    Fits the weights of a linear model. The weights of the model are assumed to 
    be normally distributed. RVMs also estimate the parameters alpha (precisions 
    of the distributions of the weights) and beta (precision of the distribution 
    of the noise) using type-II maximum likelihood or evidence maximization pruning
    weights, thus leading to sparse weights vectors.
    The algorithm is implemented as described by Faul and Tipping, 2003, AISTAT, 
    https://pdfs.semanticscholar.org/11f4/d997de8e35a1daf8b115439345d9994cfb69.pdf.

    Parameters
    ----------
    n_iter : int
        maximum number of iterations

    tol : float, optional, default 1.e-3
        weights convergence tolerance threshold

    compute_score : boolean, optional, default True
        whether or not to compute mse and estimate and standard 
        deviation of the deviation

    fit_itnercept : boolean, optional, default True
        whether or not to fit the intercept

    normalize : boolean, optional, default False

    copy_X : boolean, optional, default True

    verbose : boolean, optional, default False

    init_beta : float or callable, optional, default None
        if float needs to be bigger than 0
        elif callable then the function needs to return a single value

    init_alphas : np.ndarray list or tuple of float or callable, optional, default None
        same as for init_beta but for an vector of values    

    do_logbook : boolean
        Wether or not to keep the logbook during regression. 
        Format logbook = {"L":[],"alphas":[],"beta":[],"weights":[],"weights_full":[],"mse":[],"tse":[],"min":[],"max":[],"Sigma":[],
                "dev_est":[],"dev_std":[],"median_se":[]}
        Note that if do-logbook is True then the weights and hyperparameters with the smallest
        mse will be stored for future predictions. This is because under some circumstances
        the convergence may fluctuate strongly.
                
    beta_every : int, optional, default 1
        The noise precision is update every 'beta_every' iterations.
        
    update_pct : float, optional, default 1
        The percentage of alphas to be updated every iteration. This can be useful to prevent
        RVMs removing all weights in one iteration.

    Attributes
    ----------
    beta_ : float
        noise precision
    alphas_ : np.ndarray (n_features,) of float
        weight precisions
    active : np.ndarray (n_active,) of int
        indices to places in the full weights vector to currently active weights
    inactive : np.ndarray (n_active,) of int
        indices to places in the full weights vector to currently inactive weights
    n_iter : int
        maximum number of iterations
    tol : float
        weight covergence tolerance
    compute_score : boolean
        stores mse_, dev_est and dev_std if true
    mse_ : list of float
        mean square errors = (t-y)**2/n_samples
    dev_est : list of float
        estimate of deviation = (t-y)/n_samples
    dev_std : list of float
        one standard deviation of the deviatons = np.std(t-y,ddof=1)
    sigma_ : np.ndarray (n_features,n_features) of float
        contains the posterior covariance matrix of p(t|Xw,beta)*p(w|alphas)
    do_logbook : boolean
    logbook : dict of lists
    
    Example
    -------
    >>> from my_linear_model import RelevanceVectorMachine
    >>> from sklearn import preprocessing
    >>> import numpy as np
    >>> from scipy import stats

    >>> x = np.linspace(-np.pi,np.pi,100)
    >>> x_pred = np.linspace(-np.pi,np.pi,200)
    >>> epsilon = stats.norm(loc=0,scale=0.01)
    >>> t = np.exp(-x**2) + epsilon.rvs(size=x.shape[0])
    >>> k = 5

    >>> trafo = preprocessing.PolynomialFeatures(k)
    >>> X = trafo.fit_transform(x.reshape((-1,1)))

    >>> init_beta = 1./ np.var(t) # (that's the default start)
    >>> init_alphas = np.ones(X.shape[1])
    >>> init_alphas[1:] = np.inf

    >>> model = RelevanceVectorMachine(n_iter=50,verbose=False,compute_score=True,init_beta=init_beta,
    ...                         init_alphas=init_alphas)
    >>> model.fit(X,t)

    RelevanceVectorMachine(compute_score=True, copy_X=True, fit_intercept=True,
                init_alphas=array([  1.,  inf,  inf,  inf,  inf,  inf]),
                init_beta=8.2821399938358535, n_iter=50, normalize=False,
                tol=0.001, verbose=False)
    >>> y, yerr = model.predict(X,return_std=True)

    Notes
    -----
    The notation here is adopted from Tipping 2001, Faul and Tipping 2003 and Bishop's "Pattern 
    Recognition and Machine Learning" book. No jumping in the sewer!
    
    References
    ----------
    Mike Tipping's favorite implementation: http://www.miketipping.com/downloads.htm
    David MacKay's 1992, Bayesian Interpolation
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf
    http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf -> Ridge regression and SVD
    http://www.statisticshowto.com/wp-content/uploads/2017/07/lecture-notes.pdf -> Ridge regression and SVD and Woodbury
    """
    
    def __init__(self, n_iter=300, tol=1.e-3, compute_score=False,
                 fit_intercept=False, normalize=False, copy_X=True,
                 verbose=False,init_beta=None,init_alphas=None,do_logbook=False,
                 convergence_condition=None, beta_every=1, update_pct=1.):
        self.n_iter = n_iter
        self.tol = tol
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose
        self.init_beta = init_beta
        self.init_alphas = init_alphas
        self.beta_every = int(beta_every)
        assert 0<=update_pct<=1, "'update_pct' has to be between 0 and 1."
        self.update_pct = update_pct
        
        self.mse_ = []
        self.dev_est = [] # deviation estimate
        self.dev_std = [] # deviation standard deviation
        self.dev_mvs_95pct = [] # bayesian mean, variance and standard deviations and confidence intervals
        self.do_logbook = do_logbook
        self.logbook = {"L":[],"alphas":[],"beta":[],"weights":[],"weights_full":[],"mse":[],"tse":[],"min":[],"max":[],"Sigma":[],
              "dev_est":[],"dev_std":[],"median_se":[],"dev_mvs_95pct":[]}
        
        if not convergence_condition is None:
            assert callable(convergence_condition), "The passed 'convergence_condition' parameter needs to be callable."
        self.convergence_condition = convergence_condition

    @staticmethod
    def _initialize_beta(y, init_beta=None, verbose=False):
        beta_ = 1. / np.var(y) # default
        if not init_beta is None:
            if callable(init_beta):
                if verbose: print("Setting beta_ = init_beta()")
                beta_ = init_beta()
                assert beta_ > 0., "init_beta() produced an invalid beta_ value = {}".format(beta_)
            elif isinstance(init_beta,(int,float)):
                if verbose: print("Setting beta_ = init_beta")
                beta_ = np.copy(init_beta)
            else:
                raise ValueError("Do not understand self.init_beta = {}".format(init_beta))
        else:
            if verbose:
                print("Setting default beta_ = 1/var(t)")
        return beta_

    @staticmethod
    def _initialize_alphas(X, init_alphas=None, verbose=False):
        n_samples, n_features = X.shape
        alphas_ = np.ones(n_features) # default
        alphas_[1:] = np.inf # setting all but one basis function as inactive (see Faul and Tipping 2003 p.4)
        if not init_alphas is None:
            if callable(init_alphas):
                if verbose: print("Setting alphas_ = init_alphas()")
                alphas_ = init_alphas(X)
                assert (alphas_ > 0.).all(), "init_alphas() produced an invalid alphas_ array = {}".format(alphas_)
            elif isinstance(init_alphas,(list,tuple,np.ndarray)):
                if verbose: print("Setting alphas_ = init_alphas")
                alphas_ = np.copy(init_alphas)
            else:
                raise ValueError("Do not understand self.init_alphas = {}".format(init_alphas))
        else:
            if verbose:
                print("Setting default alphas_ = [1,inf,inf,...]")
        return alphas_
    
    def fit(self, X, y):
        """Fit the model

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """
        self.mse_ = []
        self.dev_est = []
        self.dev_std = []

        X, y = utils.check_X_y(X, y, dtype=np.float64, y_numeric=True)
        X, y, X_offset_, y_offset_, X_scale_ = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X)
        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        verbose = self.verbose
        
        # Initialization of the hyperparameters
        beta_ = self._initialize_beta(y,init_beta=self.init_beta,verbose=self.verbose)
        alphas_ = self._initialize_alphas(X,init_alphas=self.init_alphas,verbose=self.verbose)
        new_alphas_ = np.copy(alphas_)
                
        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        
        # Convergence loop of the RVM regression
        N, M = X.shape
        for iter_ in range(self.n_iter):
            
            # (in-)active basis functions
            active = np.where(np.isfinite(alphas_))[0]
            n_active = active.shape[0]
            inactive = np.where(np.isinf(alphas_))[0]
            if verbose:
                print("{}: active / inactive functions = {} / {} ".format(iter_+1, len(active),len(inactive)))
            
            # corresponding Sigma matrix (weights hyperprior covariance matrix)
            Sigma = np.diag(alphas_)
            Sigma_a = np.diag(alphas_[active]) # active part of Sigma -> numpy select?
            X_a = X[:,active] # active part of the design matrix
            
            # weights posterior mean (w_new) and covariance (A_new)
            A_new = np.linalg.inv(beta_ * X_a.T.dot(X_a) + Sigma_a)
            w_new = beta_ * A_new.dot(X_a.T.dot(y))
                        
            # mse
            dt = y - np.dot(X_a, w_new)
            #mse_ = np.linalg.norm(dt)**2
            mse_ = (dt**2).sum()
                                                
            # Compute objective function: Gaussian for p(w|X,t,alphas,beta) \propto p(t|Xw,beta)p(w|alphas)
            if self.compute_score:
                log_prefactor = n_features*(beta_ - 2.*np.pi) - alphas_[active].sum() - 2.*np.pi
                log_likelihood = -beta_ * mse_
                log_prior = - w_new.T.dot(Sigma_a.dot(w_new))
                log_posterior = .5 * (log_prefactor + log_likelihood + log_prior)
                self.scores_.append(log_posterior)
                self.mse_.append(float(mse_/n_samples))
                self.dev_est.append(dt.mean())
                self.dev_std.append(dt.std(ddof=1))
                self.dev_mvs_95pct.append(stats.bayes_mvs(dt,alpha=.95))
            if self.do_logbook:
                logbook = {"L":[],"alphas":[],"beta":[],
                "weights":[],"weights_full":[],"mse":[],"tse":[],"min":[],"max":[],"Sigma":[],
              "dev_est":[],"dev_std":[],"median_se":[]}
                if self.compute_score:
                    self.logbook["L"].append(self.scores_[-1])
                else:
                    log_prefactor = n_features*(beta_ - 2.*np.pi) - alphas_[active].sum() - 2.*np.pi
                    log_likelihood = -beta_ * mse_
                    log_prior = - w_new.T.dot(Sigma_a.dot(w_new))
                    log_posterior = .5 * (log_prefactor + log_likelihood + log_prior)
                    self.logbook["L"].append(log_posterior)
                self.logbook["alphas"].append(alphas_)
                self.logbook["beta"].append(beta_)
                self.logbook["weights"].append(w_new)
                self.logbook["weights_full"].append(full_weight_vector(w_new,active,inactive))
                self.logbook["mse"].append(mse_/n_samples)
                self.logbook["tse"].append(mse_)
                self.logbook["min"].append(np.amin(dt))
                self.logbook["max"].append(np.amax(dt))
                self.logbook["dev_est"].append(dt.mean())
                self.logbook["dev_std"].append(dt.std())
                self.logbook["dev_mvs_95pct"].append(stats.bayes_mvs(dt,alpha=.95))
                self.logbook["median_se"].append(np.median(dt))
            
            # Check for convergence
            if iter_ != 0:
                coef_new_full = full_weight_vector(np.copy(w_new),active,inactive)
                if self.convergence_condition is None:
                    if np.sum(np.abs(coef_new_full - coef_old_)) < self.tol:
                        if verbose:
                            print("Convergence after ", str(iter_), " iterations")
                        break
                else:
                    if self.convergence_condition(coef_new_full,coef_old_,self):
                        if verbose:
                            print("Convergence after ", str(iter_), " iterations")
                        break
                # end of the rope
                if iter_ >= self.n_iter-1:
                    if verbose:
                        print("Iteration terminated after n_iter = {} step(s)".format(self.n_iter))
                    break
                
            coef_old_ = full_weight_vector(np.copy(w_new),active,inactive)
            
            # Recompute beta
            beta_old_ = np.copy(beta_)
            if iter_ % self.beta_every == 0:
                beta_ = (n_samples - n_active + np.sum(alphas_[active]*np.diag(A_new)))
                beta_ /= mse_
            
            """
            # Compute S and Q (Faul and Tipping 2003 eqs. 24 & 25)
            S0_tilde = beta_old_ * np.einsum("nm,nm->m", X, X) # in R^(n_features)
            S1_tilde = - beta_old_**2 * np.einsum("mn,na->ma",X.T,np.dot(X_a,A_new)) # in R^(n_features x n_active)
            S2_tilde = np.einsum("na,nm->am",X_a, X) # in R^(n_active x n_features)
            S = S0_tilde + np.einsum("ma,am->m",S1_tilde,S2_tilde)
                        
            Q0_tilde = beta_old_ * np.einsum("nm,n->m", X, y) # in R^(n_features)
            Q2_tilde = np.einsum("na,n->a",X_a, y) # in R^(n_active)
            Q = Q0_tilde + np.einsum("ma,a->m",S1_tilde,Q2_tilde)
                        
            # Compute s and q (note the lower case)
            s = np.copy(S)
            q = np.copy(Q)
            s[active] = alphas_[active]*S[active]/(alphas_[active]-S[active])
            q[active] = alphas_[active]*Q[active]/(alphas_[active]-S[active])
            
            # Recompute alphas using pruning
            active = np.where(q**2>s)[0]
            inactive = np.where(np.logical_not(q**2>s))[0]
            new_alphas_[inactive] = np.inf
            new_alphas_[active] = s[active]**2/(q[active]**2-s[active])
            """
                        
            # alternative version
            tmp = beta_old_ * X.T - beta_old_**2 * np.dot(X.T.dot(X_a), A_new.dot(X_a.T))
            Q = np.dot(tmp,y)
            Q = np.reshape(Q,(-1,))
            S = np.einsum("ij,ji->i",tmp,X)
            
            q, s = np.zeros(M), np.zeros(M)
            q[inactive] = Q[inactive]
            q[active] = alphas_[active]*Q[active]/(alphas_[active]-S[active])
            s[inactive] = S[inactive]
            s[active] = alphas_[active]*S[active]/(alphas_[active]-S[active])
            
            q2_larger_s = np.where(q**2>s)[0]
            q2_smaller_s = np.where(q**2<=s)[0]
            new_alphas_[q2_larger_s] = s[q2_larger_s]**2/(q[q2_larger_s]**2-s[q2_larger_s])
            new_alphas_[q2_smaller_s] = np.inf
            
            if self.update_pct < 1:
                ix = np.random.choice(np.arange(M),replace=False,size=int(M*self.update_pct))
                alphas_[ix] = new_alphas_[ix]
            else:
                alphas_ = np.copy(new_alphas_)
                        
        if self.do_logbook:
            ix = np.argsort(np.array(self.logbook["mse"]))[0]
            alphas_ = np.array(self.logbook["alphas"][ix])
            beta_ = self.logbook["beta"][ix]
            
            active = np.where(np.isfinite(alphas_))[0]
            inactive = np.where(np.isinf(alphas_))[0]
            
            Sigma = np.diag(alphas_)
            Sigma_a = np.diag(alphas_[active]) # active part of Sigma -> numpy select?
            X_a = X[:,active] # active part of the design matrix
            
            # weights posterior mean (w_new) and covariance (A_new)
            A_new = np.linalg.inv(beta_ * X_a.T.dot(X_a) + Sigma_a)
            w_new = beta_ * A_new.dot(X_a.T.dot(y))
            
        self.coef_ = w_new
        self.active = active
        self.inactive = inactive
        self.sigma_ = A_new
        self.beta_ = beta_
        
        self._set_intercept(X_offset_[active], y_offset_, X_scale_[active])
        
        return self
        
    def predict(self, X, return_std=False):
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        return_std : boolean, optional
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array, shape = (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array, shape = (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        X_a = X[:,self.active]
        y_mean = self._decision_function(X_a)
        if return_std is False:
            return y_mean
        else:
            if self.normalize:
                X_a = (X_a - self.X_offset_) / self.X_scale_
            
            sigmas_squared_data = (X_a.dot(self.sigma_) * X_a).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1. / self.beta_))
            return y_mean, y_std
        
    def get_full_weights_vector(self):
        return full_weight_vector(self.coef_,self.active,self.inactive)
    
    def get_logbook(self):
        assert self.do_logbook, "Logbook empty because do_logbook = {}.".format(self.do_logbook)
        return self.logbook
    
def distribution_wrapper(dis,size=None,single=True):
    """Wraps scipy.stats distributions for RVM initialization.

    Parameters
    ----------

    dis : instance of scipy.stats.rv_continuous
        The distribution to be wrapped.

    size : int or None, optional, default None
        How many samples to draw (if given, see 'single').

    single : boolean, optional, default True
        Whether or not a single float value is to be returned or an array of values.
        If single == False then either 'size' samples are drawn or otherwise if the
        design matrix is provided as an argument of the wrapped function 'samples'
        then as M samples are drawn (N, M = X.shape).

    Returns
    -------

    samples : function
        Can be called to generate samples based on 'dis' and 'X' 
        (parameter for 'samples').
    """
    def samples(X=None):
        if single:
            return dis.rvs(size=1)[0]
        else:
            if isinstance(size,int):
                return dis.rvs(size=size)
            elif isinstance(X,np.ndarray):
                return dis.rvs(size=X.shape[1])
            else:
                raise ValueError("size is not properly specified")
    return samples

def repeated_regression(x,base_trafo,model_type,model_kwargs,t=None,tfun=None,
        epsilon=None,Nruns=100,return_coefs=False,return_models=False,):
    """Repeats regressions.

    This can be used to do multiple regressions on freshly regenerated 
    data (requires passing of a scipy.stats.rv_continuous object as epsilon,
    and a callable tfun) or simply on the same data over an over.

    Parameters
    ----------
    x : np.ndarray
        input / estimators
    tfun : callable
        t = tfun(x)
    epsilon : scipy.stats distribution object
        noise random variable
    base_trafo : callable
        for example the sklearn.preprocessing function such as PolynomialFeatures
        to transform x into X
    model : instance of regression class like RelevanceVectorMachine
    
    Example
    -------
    >>> model_type = linear_model.RelevanceVectorMachine
    >>> model_kwargs = dict(n_iter=250,verbose=False,compute_score=True,init_beta=init_beta,
                       init_alphas=init_alphas,fit_intercept=False)
    >>> runtimes, coefs, models = repeated_regression(x,base_trafo,model_type,t=t,tfun=None,epsilon=None,
                                   model_kwargs=model_kwargs,Nruns=Nruns,return_coefs=True,return_models=True)
    """
    
    X = base_trafo(x.reshape((-1,1)))
    assert not t is None or not (tfun is None and epsilon is None), "Either 't' has to be given or 'tfun' and 'epsilon'!"
    if t is None:
        t = tfun(x) + epsilon.rvs(size=x.shape[0])
    
    runtimes = np.zeros(Nruns)
    coefs, models = [], []
    for i in range(Nruns):
        t0 = time.time()
        model = model_type(**model_kwargs)
        model.fit(X,t)
        runtimes[i] = time.time() - t0
        if return_coefs:
            coefs.append(model.get_full_weights_vector())
        if return_models:
            models.append(model)
    if return_coefs and not return_models:
        return runtimes, np.array(coefs)
    elif return_coefs and return_models:
        return runtimes, np.array(coefs), models
    elif not return_coefs and return_models:
        return runtimes, models
    return runtimes

def print_run_stats(base_trafo,x,runtimes,coefs,Nruns,show_coefs=True):
    print("\n================================================")
    s = "X = {} & Nruns = {}:".format(base_trafo(x.reshape((-1,1))).shape,Nruns)
    print(s)
    print("-"*len(s))
    print("\ntime: estimate = {:.4f}s, 2*std = {:.4f}s".format(runtimes.mean(),2*np.std(runtimes,ddof=1)))
    if show_coefs:
        print("\ncoefs (estimate +- 2*std):")
        for i in range(coefs.shape[1]):
            print("    {}: {:.4f} +- {:.4f}".format(i,coefs[:,i].mean(axis=0),
                2*np.std(coefs[:,i],axis=0,ddof=1)))        

if __name__ == "__main__":
    epsilon = stats.norm(loc=0,scale=0.01)
    tfun = lambda x: np.sin(x) + np.cos(2.*x)

    init_beta = distribution_wrapper(stats.halfnorm(scale=1),size=1,single=True)
    init_alphas = distribution_wrapper(stats.halfnorm(scale=1),single=False)

    Nruns = 100

    N = 100
    Ms = [3,5,10,20,50]

    t_est, t_err = [], []
    for M in Ms:
        x = np.linspace(0,1,N)
        k = M
        
        trafo = FourierFeatures(k=k)
        base_trafo = trafo.fit_transform
        
        model_type = RelevanceVectorMachine
        model_kwargs = dict(n_iter=250,verbose=False,compute_score=True,init_beta=init_beta,
                            init_alphas=init_alphas)

        runtimes, coefs = regression_speedtest(x,base_trafo,model_type,t=None,tfun=tfun,epsilon=epsilon,
                                            model_kwargs=model_kwargs,Nruns=Nruns,return_coefs=True)
        print_run_stats(base_trafo,x,runtimes,coefs,Nruns)