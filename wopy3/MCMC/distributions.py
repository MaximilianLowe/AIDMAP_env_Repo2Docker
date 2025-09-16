"""Probability distributions
This module contains statistical distributions for use in MCMC codes.
"""

import numpy as np
import scipy.stats as stats

class SignConstrainedPrior:
    """
    Prior that returns 1 if all values in x have the same signs as self.signs, and
    -np.inf otherwise
    
    Parameters
    ----------
    signs : np.ndarray
        Signs to check
    """
    def __init__(self,signs):
        self.signs = signs
    def __call__(self,x):
        x2 = x.copy()
        x2 = x2*self.signs
        if np.all(x2>=0):
            return 1.0
        else:
            return -np.inf
            
            
class RangePrior:
    """
    Prior that returns a constant if all values in x are in a specified range
    and -np.inf otherwise.
    
    Parameters
    ----------
        ub : np.ndarray
            Upper boundaries
        lb : np.ndarray
            Lower boundaries.
    """
    def __init__(self,lb,ub):
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.log_prob = np.sum(np.log(1.0/(self.ub-self.lb)))
    
    def __call__(self,x):
        return self.logpdf(x)
    
    def pdf(self,x):
        if np.all((x>=self.lb) & (x<=self.ub)):
            return np.exp(self.logpdf(x))
        else:
            return 0.0

    def logpdf(self,x):
        if np.all((x>=self.lb) & (x<=self.ub)):
            if x.ndim==2:
                return self.log_prob * x.shape[0]
            else:
                return self.log_prob
        else:
            return -np.inf

    def rvs(self):
        return self.lb + np.random.random(self.lb.size) * (self.ub-self.lb)
            

class IndependentGaussianMisfit:
    """
    Independent Gaussian distributions with fixed or variable standard deviation
    
    Parameters
    -----------
    data: np.ndarray 
        Corresponds to mean of Gaussian distribution
    sigma : np.ndarray, optional
        std of Gaussian distribution. If not given, the resulting sigma needs to be given explcitly, when calling
        an instance of this class.
    """
    def __init__(self,data,sigma=None):
        self.data = data
        self.n = len(data)
        self.sigma = sigma
    def __call__(self,y,hyper=None):
        delta = y-self.data
        if hyper is None:
            sigma = self.sigma
        else:
            sigma = hyper[0]
        return -np.sum(self.n*np.log(sigma))-0.5*np.sum(delta**2/sigma**2)

class DiscreteIntDist:
    """Distribution of random ints between two numbers
    This is basically equivalent to scipy.stats.randint ... but they use (correctly) the name
    probability _mass_ function instead of probability _density_ function. Hence, my code
    cannot directly call the scipy.stats code.
    p(k) = 1/(high-low) if k = low, ..., high - 1

    Parameters
    ----------
    low : int
        Lowest number
    high : int
        Highest number - 1
    shape: tuple of int
        Shape of the arrays to return from rvs
    """
    def __init__(self,low,high,shape=(1,1)):
        self.v = np.arange(low,high,1,dtype=int)
        self.shape = shape
    
    def pdf(self,x):
        if np.any(x<self.v[0])  | np.any(x>self.v[-1]):
            return np.zeros(x.shape)
        else:
            return np.ones(x.shape)/len(self.v)
    
    def logpdf(self,x):
        return np.log(self.pdf(x))
    
    def rvs(self):
        return np.random.randint(self.v[0],self.v[-1]+1,size=self.shape)


class ComponentUpdateProposal:
    """
    Objects of this class can be called to generate a proposal for MCMC. 
    A component is selected randomly and altered randomly 
    using a normal distribution. 
    The scale of the change is determined by class parameter scale
    """
    def __init__(self,scales):
        self.scales = scales
        self.n = len(scales)
    def __call__(self):
        i = np.random.randint(0,self.n)
        return i,np.random.randn() * self.scales[i]
    
    def rvs(self):
        x = np.zeros(self.n)
        i = np.random.randint(0,self.n)
        x[i] = np.random.randn() * self.scales[i]
        return x

    def pdf(self,x):
        return np.exp(self.logpdf(x))

    def logpdf(self,x):
        assert np.ndim(x) == 1
        # Not implemented for 2D yet...
        nnz = np.nonzero(x)[0]
        
        if len(nnz)<=1:
            i = nnz[0]
            return -np.sum(np.log(self.scales[i]))-0.5*np.sum(x[i]**2/self.scales[i]**2)
        else:
            return -np.inf
            
class CorrelatedGaussian:
    """
    Callable object that evaluates a multivariate Gaussian distribution (log!)
    """
    def __init__(self,mu,sigma,allow_singular=False):
        self.mu = mu
        self.sigma = sigma
        self.N = len(self.mu)
        self.allow_singular = allow_singular
        if self.allow_singular:
            self.psd = stats._multivariate._PSD(sigma)
            self.U = self.psd.U
        else:
            self.sigma_inv = np.linalg.inv(sigma)
            self.slogdet = np.linalg.slogdet(self.sigma_inv)[1]

    
    def __call__(self,x,sigma_scale=1.0):
        """Evaluate correlated multivariate normal distribution
        Parameters
        ----------
        x : (N,) array
        
        sigma_scale : float
            This will be used to scale the distribution. It corresponds to the
            standard deviation (!) not the variance. 
        """
        _LOG_2PI = np.log(2 * np.pi)
        if self.allow_singular:
            # Taken from Scipy mvn code
            dev = x - self.mu
            maha = np.sum(np.square(np.dot(dev, self.U/sigma_scale)), axis=-1)
            return -0.5 * (self.psd.rank * _LOG_2PI + self.psd.log_pdet + maha + self.psd.rank * 2 * np.log(sigma_scale))
        else:
            # Note self.slogdet is positive because it is the determinant of sigma_inv!
            return 0.5*self.slogdet - self.N * np.log(sigma_scale) - 0.5 * ((x-self.mu)/sigma_scale**2).T.dot(self.sigma_inv.dot((x-self.mu))) - 0.5 * len(x)*_LOG_2PI
        