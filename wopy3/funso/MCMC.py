import numpy as np
from scipy.spatial.distance import cdist
_LOG_2PI = np.log(2 * np.pi)

class CompoundPrior:
    def __init__(self,sub_priors,ratio_lim=None):
        self.parts = len(sub_priors)
        self.sub_priors = sub_priors
        if np.isscalar(ratio_lim):
            self.ratio_lim = ratio_lim * np.ones((self.parts,self.parts))
        else:
            self.ratio_lim = ratio_lim
        
    def __call__(self,x):
        total_N = len(x)
        segments = np.array_split(x,self.parts)       
        # Check ratios
        if not self.ratio_lim is None:
            for i in range(self.parts):
                for j in range(i+1,self.parts):
                    ratio = max(segments[i],segments[j]) / min(segments[i],segments[j])
                    if ratio>self.ratio_lim[i,j]:
                        return -np.inf
                        
        L = 0
        for i in range(self.parts):
            L = L+self.sub_priors[i](segments[i])
        return L

class CompoundMisfit:
    def __init__(self,sub_misfits,hy_indices=None):
        self.parts = len(sub_misfits)
        self.sub_misfits = sub_misfits
        if hy_indices is None:
            self.hy_indices = np.zeros((self.parts))
        else:
            self.hy_indices = hy_indices
    def __call__(self,y,hyper):
        L = 0
        for i in range(self.parts):
            L=L+self.sub_misfits[i](y[i],[hyper[self.hy_indices[i]]])
        return L
class CompoundForward:
    def __init__(self,sub_forwards):
        self.parts = len(sub_forwards)
        self.sub_forwards = sub_forwards
    def __call__(self,x):
        y=[]
        for i in range(self.parts):
            y.append(self.sub_forwards[i](x))
        return np.array(y)

class DiscreteIntDist:
    """Callable discrete distribution on integers
    """
    
    def __init__(self,xv,pv):
        self.xv = xv
        self.pv = pv
    def __call__(self,x):
        """Returns log-likelihood
        """
        if x in self.xv:
            return self.pv[self.xv==x]
        else:
            return -np.inf
        
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

class LayeredPrior:
    """Independent normal distribution for model layers
    TODO: More flexible design, such that internally different priors can be used instead of normal distribution
    """
    def __init__(self,interfaces,sigma,thickness_mode=False):
        self.shape = interfaces.shape
        self.data = interfaces.copy()
        self.sigma = sigma
        self.thickness_mode = thickness_mode
        if thickness_mode:
            self.data = self.data[-1,:,:]
    
    def __call__(self,x):
        x_mat = x.reshape(self.shape)
        if np.all(np.diff(x_mat,axis=0)>=0):
            if self.thickness_mode:
                err = x_mat[-1,:,:] - self.data
                return -np.sum((err**2)/self.sigma**2)
            else:
                return -np.sum((x_mat - self.data)**2 / self.sigma**2)
        else:
            return -np.inf
            
class LogComponentUpdateProposal:
    """
    """
    
    def __init__(self,scales):
        self.scales = scales
        self.n = len(scales)
    def __call__(self):
        i = np.random.randint(0,self.n)
        return i,np.random.randn() * self.scales[i]
        
class IndependentGaussianMisfit:
    """
    Callable, returns logpdf
    Input:
    data: Corresponds to mean of Gaussian distribution
    hyper: Corresponds to std of Gaussian distribution
    """
    def __init__(self,data):
        self.data = data
        self.n = len(data)
    def __call__(self,y,hyper):
        delta = y-self.data
        return -self.n*np.log(hyper[0])-0.5*np.sum(delta**2/hyper[0]**2)


class GradientMisfit:
    """Special misfit class for gradient components
    Option 1: consistent:
        1. Correlation between gxx,gyy,gzz at the same point is taken into account
        2. A consistent sigma is used for all gradient components
    Option 2: zz
        1. Just use zz and ingore all other components
    Option 3: spatial
        1. Use a covariance function but only between the same component
           Because I haven't figured out how to account for cross-component
           spatial coupling (like in option 1). Variance is consistent between
           different components
    """
    def __init__(self,data,mode='consistent',**kwargs):
        self.data = data
        self.n = data.shape[0]
        self.mode = mode
        assert data.shape[1] == 6
        # R is a normalized (sigma=1) covariance matrix (not a correlation matrix!)
        if self.mode == 'consistent':
            self.R = make_grad_cov_matrix(1.0)
            self.Rinv = np.linalg.pinv(self.R)
        elif self.mode == 'zz':
            self.independent_misfit = IndependentGaussianMisfit(self.data[:,-1])
        elif self.mode == 'spatial':
            corr_distance = kwargs.get("corr_distance")
            # Covar_func has signature C(d,pars), where pars is tuple of three
            # (nugget,sill,range)
            covar_func = kwargs.get("covar_func")
            pd = kwargs.get("pd")
            sigma_mat = covar_func(pd,(0.0,1.0,corr_distance))
            self.base_misfit = CorrelatedGaussian(np.zeros((pd.shape[0])),sigma_mat,numerical_stabilization=True)
            self.sigma_scales = (0.375,0.125,0.5,0.375,0.5,1.0)
            
            
    def __call__(self,y,hyper):
        y = y.reshape((self.n,6))
        dy = y-self.data
        if self.mode == 'consistent':
            sigma = hyper[0]**2 * self.R 
            sigmainv = self.Rinv / hyper[0]**2
            pseudo_determinant = np.prod(np.sort(np.linalg.eigvals(sigma*2*np.pi))[1:])        
            L = - 0.5 * self.n * np.log(pseudo_determinant)
            L = L - 0.5 * np.tensordot(dy.T,sigmainv.dot(dy.T))
            return L
        elif self.mode == 'zz':
            return self.independent_misfit(y[:,-1],hyper)
        elif self.mode == 'spatial':
            L = 0.0
            for i in range(6):
                L += self.base_misfit(dy[:,i],self.sigma_scales[i]*hyper[0])
            return L
    
def make_grad_cov_matrix(sigma):
    """Compute covariance matrix for 6 gradient components in ENU system
    """
    R = np.diag((0.375,0.125,0.5,0.375,0.5,1.0))
    R[0,3] = 0.125
    R[3,0] = 0.125
    R[0,5] = -0.5
    R[5,0] = -0.5
    R[3,5] = -0.5
    R[5,3] = -0.5
    return R*sigma**2
    
        
class IndependentGaussianPrior:
    """
    Each of the parameters has its own standard deviation assigned
    to it.
    """
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    def __call__(self,x):
        delta = x - self.mu
        return -0.5*np.sum((delta/self.sigma)**2) - np.sum(np.log(self.sigma))

class CorrelatedGaussian:
    """
    Callable object that evaluates a multivariate Gaussian distribution (log!)
    """
    def __init__(self,mu,sigma,inverted=False,numerical_stabilization=False):
        self.mu = mu
        if inverted:
            self.sigma_inv = sigma
        else:
            self.sigma_inv = np.linalg.inv(sigma)
        self.slogdet = np.linalg.slogdet(self.sigma_inv)[1]
        self.N = len(self.mu)
        self.numerical_stabilization = numerical_stabilization
        if self.numerical_stabilization:
            U,S,V = np.linalg.svd(self.sigma_inv)
            self.L = np.sqrt(S) * V.T
        
    def __call__(self,x,sigma_scale=None):
        """Evaluate correlated multivariate normal distribution
        Parameters
        ----------
        x : (N,) array
        
        sigma_scale : float
            This will be used to scale the distribution. It corresponds to the
            standard deviation (!) not the variance. 
        """
        
        # Note self.slogdet is positive because it is the determinant of sigma_inv!
        if self.numerical_stabilization:
            if sigma_scale is None:
                return 0.5*self.slogdet - 0.5 * np.sum(np.square(np.dot(x-self.mu,self.L))) - 0.5 * len(x)*_LOG_2PI
            else:
                return 0.5*self.slogdet - self.N * sigma_scale - 0.5 * np.sum(np.square(np.dot(x-self.mu,self.L)))/sigma_scale**2 - 0.5 * len(x)*_LOG_2PI
        else:
            if sigma_scale is None:
                return 0.5*self.slogdet - 0.5 * (x-self.mu).T.dot(self.sigma_inv.dot((x-self.mu))) - 0.5 * len(x)*_LOG_2PI
            else:
                return 0.5*self.slogdet - self.N * sigma_scale - 0.5 * (x-self.mu).T.dot(self.sigma_inv.dot((x-self.mu)))/sigma_scale**2 - 0.5 * len(x)*_LOG_2PI

class SignConstrainedPrior:
    def __init__(self,signs):
        self.signs = signs
    def __call__(self,x):
        x2 = x.copy()
        x2 = x2*self.signs
        return np.all(x2>=0)

class RangePrior:
    def __init__(self,lb,ub):
        self.lb = lb
        self.ub = ub
    def __call__(self,x):
        if np.all((x>=self.lb) & (x<=self.ub)):
            return 1.0
        else:
            return -np.inf

class BoundedOoMPrior:
    """Bounded Order of Magnitude prior (Jeffreys)
    A uniform distribution is assigned to the log of the quantities in question.
    """
    
    def __init__(self,lb,ub):
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        assert np.all(self.lb>0) and np.all(self.ub>0)
        self.c = 1.0/(1.0/self.lb**2 - 1.0/self.ub**2)
        
    def __call__(self,x):
        if np.all((x>=self.lb) & (x<=self.ub)):
            return np.sum(np.log(self.c)) - np.sum(np.log(x))
            
        else:
            return -np.inf
    
            
class TemperedChainWithHyper:
    def __init__(
    self,x0,hyper0,LFuncDict,methodDict,temperature=1.0):
        self.x = x0.copy()
        self.hyper = hyper0.copy()
        self.forward = LFuncDict['forward']
        self.misfit = LFuncDict['misfit']
        self.prior = LFuncDict['prior']
        self.hyperPrior = LFuncDict['hyperPrior']
        
        self.proposal = methodDict['proposal']
        self.hyperProposal = methodDict['hyperProposal']
        
        if 'hyperProp' in methodDict:
            self.hyperProp = methodDict['hyperProp']
        else:
            self.hyperProp = 0.5
        
        self.temperature = temperature
        self.y = self.forward(self.x)
        self.oldL = (self.misfit(self.y,self.hyper) + 
            self.prior(self.x) + self.hyperPrior(self.hyper))
        
    def step(self):
        if np.random.rand() > self.hyperProp:
            changedParam,dx = self.proposal()
            xcan = self.x.copy()
            xcan[changedParam] = xcan[changedParam] + dx
            # ycan = forward.partial_update(
                # changedParam,xcan[changedParam])
            ycan = self.forward(xcan)
            hypercan = self.hyper.copy()
        else:
            hyperChanged,dh = self.hyperProposal()
            hypercan = self.hyper.copy()
            hypercan[hyperChanged] = hypercan[hyperChanged] + dh
            xcan = self.x.copy()
            ycan = self.y.copy()
        newL = (self.misfit(ycan,hypercan) + self.prior(xcan) + 
            self.hyperPrior(hypercan))
        acceptance = np.exp((newL - self.oldL)/self.temperature)
        if np.random.rand() < acceptance:
            self.x = xcan.copy()
            self.hyper = hypercan.copy()
            self.y = ycan.copy()
            self.oldL = newL
    
class TemperedHMCWithHyper:
    def __init__(
    self,x0,hyper0,LFuncDict,methodDict,temperature=1.0):
        self.x = x0.copy()
        self.hyper = hyper0.copy()
        
        
        self.forward = LFuncDict['forward']
        self.misfit = LFuncDict['misfit']
        self.prior = LFuncDict['prior']
        self.hyperPrior = LFuncDict['hyperPrior']
        
        self.proposal = methodDict['HMC']
        self.hyperProposal = methodDict['hyperProposal']
        self.temperature = temperature
        self.y = self.forward(self.x)
        self.oldL = (self.misfit(self.y,self.hyper) + 
            self.prior(self.x) + self.hyperPrior(self.hyper))
    
    def calc_grad(self,xp,Up,dx):
        grad_U = np.zeros((self.x.shape[0]))
        for j in range(self.x.shape[0]):
            x2 = xp.copy()
            x2[j] = x2[j] + dx
            y2 = self.forward(x2)
            U2 = -(self.misfit(y2,self.hyper) + self.prior(x2)) / self.temperature
            grad_U[j] = (U2 - Up)/dx
        return grad_U
    
    def step(self):
        if np.random.rand() > 0.5:
            # HMC step U = -1/T * log(L)
            # I'm abusing the proposal here to store the HMC parameters
            dx,eps,L = self.proposal()
            grad_U = self.calc_grad(self.x,-self.oldL/self.temperature,dx) 
            # Start HMC
            startP = np.random.randn(self.x.shape[0])
            p = startP.copy()
            p = p - 0.5 * eps * grad_U
            xp = self.x.copy()
            for i in range(L):
                xp = xp + eps * p
                yp = self.forward(xp)
                Up = -(self.misfit(yp,self.hyper) + self.prior(xp))/self.temperature
                Kp = 0.5 * np.sum(p**2) / self.temperature
                if not i==L-1:
                    p = p - eps * self.calc_grad(xp,Up,dx)
            p = p - 0.5 * eps * self.calc_grad(xp,Up,dx)
            p = -p
            
            oldK = (0.5 * np.sum(startP**2))
            newK = (0.5 * np.sum(p**2))
            
            ycan = yp
            xcan = xp
            hypercan = self.hyper.copy()
        else:
            hyperChanged,dh = self.hyperProposal()
            hypercan = self.hyper.copy()
            hypercan[hyperChanged] = hypercan[hyperChanged] + dh
            xcan = self.x.copy()
            ycan = self.y.copy()
            oldK = 0.0
            newK = 0.0
            
        newL = (self.misfit(ycan,hypercan) + self.prior(xcan) + 
            self.hyperPrior(hypercan))
        acceptance = np.exp(newL/self.temperature - self.oldL/self.temperature - newK + oldK)
        if np.random.rand() < acceptance:
            self.x = xcan.copy()
            self.hyper = hypercan.copy()
            self.y = ycan.copy()
            self.oldL = newL

def global_uniform_prior(x):
    return 1

def global_positive_prior(x):
    if x>=0:
        return 1
    else:
        return 0
    
def PT_with_hyper(
    x0,hyper0,LFuncDict,methodDict,nruns,temperatures):
    
    nT = temperatures.shape[0]
    chains = []
    xchain = np.zeros((nruns,nT,x0.shape[0]))
    hyperchain = np.zeros((nruns,nT,hyper0.shape[0]))
    Lchain = np.zeros((nruns,nT))
    for j in range(nT):
        if 'HMC' in methodDict:
            chains.append(TemperedHMCWithHyper(
                        x0,hyper0,LFuncDict,methodDict,temperatures[j]))
        else:
            chains.append(TemperedChainWithHyper(
                        x0,hyper0,LFuncDict,methodDict,temperatures[j]))
        xchain[0,j,:] = x0.copy()
        hyperchain[0,j,:] = hyper0.copy()
        Lchain[0,j] = chains[j].oldL
    
    
    for i in range(nruns-1):
        for j in range(nT):
            chains[j].step()
        half1 =  np.random.permutation(nT)
        half2 = np.random.permutation(nT)
        for j in range(len(half1)):
            p = half1[j]
            q = half2[j]
            if p==q:
                continue
            # exponent = (chains[q].oldL * temperatures[q] - 
                        # chains[p].oldL * temperatures[p])/temperatures[p] \
                        # + (chains[p].oldL * temperatures[p] -
                        # chains[q].oldL * temperatures[q])/temperatures[q]
            exponent = (chains[p].oldL - chains[q].oldL) / temperatures[q] \
                    + (chains[q].oldL - chains[p].oldL) / temperatures[p]
            acceptance = np.exp(exponent)
            u = np.random.rand()
            if u<acceptance:
                temp = chains[p].x.copy()
                chains[p].x = chains[q].x.copy()
                chains[q].x = temp.copy()
                
                temp = chains[p].y.copy()
                chains[p].y = chains[q].y.copy()
                chains[q].y = temp.copy()
                
                temp = chains[p].oldL
                chains[p].oldL = chains[q].oldL
                chains[q].oldL = temp
                
                temp = chains[p].hyper.copy()
                chains[p].hyper = chains[q].hyper.copy()
                chains[q].hyper = temp
                
        for j in range(nT):
            xchain[i+1,j,:] = chains[j].x.copy()
            hyperchain[i+1,j,:] = chains[j].hyper.copy()
            Lchain[i+1,j] = chains[j].oldL    
    return xchain,hyperchain,Lchain
        
def MCMC(x0,forward,misfit,prior,proposal,nruns,
            hyper0=None,hyperProposal=None,hyperPrior=None,
            chainSave=1):
    x = x0.copy()
    y = forward(x)
    if not hyper0 is None:
        hyper = hyper0.copy()
    else:
        hyper = None
    if not hyper0 is None:
        oldL = misfit(y,hyper) + prior(x) + hyperPrior(hyper)
    else:
        oldL = misfit(y) + prior(x)
    
    xchain = np.zeros((nruns//chainSave,x.shape[0]))
    Lchain = np.zeros((nruns//chainSave))
    xchain[0,:] = x0
    if not hyper is None:
        hyperchain = np.zeros((nruns//chainSave,hyper.shape[0]))
        hyperchain[0,:] = hyper
        
    accepted = 0
    Lchain[0] = oldL
    
    for i in range(nruns-1):
        if hyper is None or np.random.rand() > 0.5:
            changedParam,dx = proposal()
            xcan = x.copy()
            xcan[changedParam] = xcan[changedParam] + dx
            ycan = forward(xcan)
            if not hyper is None:
                hypercan = hyper.copy()
        else:
            hyperChanged,dh = hyperProposal()
            hypercan = hyper.copy()
            hypercan[hyperChanged] = hypercan[hyperChanged] + dh
            xcan = x.copy()
            ycan = y.copy()
        if not hyper is None:
            newL = misfit(ycan,hypercan) + prior(xcan) + hyperPrior(hypercan)
        else:
            newL = misfit(ycan) + prior(xcan)
        acceptance = np.exp(newL - oldL)
        if np.random.rand() < acceptance:
            accepted = accepted + 1
            x = xcan.copy()
            if not hyper is None:
                hyper = hypercan.copy()
            y = ycan.copy()
            oldL = newL
        
        if (i+1)%chainSave==0:
            xchain[(i+1)//chainSave,:] = x.copy()
            if not hyper is None:
                hyperchain[(i+1)//chainSave,:] = hyper.copy()
            Lchain[(i+1)//chainSave] = oldL
    if not hyper is None:
        return xchain,hyperchain,Lchain,accepted
    else:
        return xchain,Lchain,accepted