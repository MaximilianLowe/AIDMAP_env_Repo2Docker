"""Transdimensional MCMC framework

As it stands this contains a lot of boilerplate code, particularly with the different kinds of models and proposals.
There might be a way to do this more elegantly.
"""
import numpy as np
import scipy.interpolate
import random
import copy

class TransDimChain:
    """Stores state of a trans-dimensional chain
        
    Parameters
    ----------
        startModel : obj 
            Model parametrization
        forward : Callable 
            Takes a model and returns some forward calculated data.
        proposal : obj 
            Applies random changes to the model during the chain. Needs to define log_prior, birth, death, change_param and move
        misfit : Callable 
            If hyper is None, takes one only argument the modelled data y and returns 
            the log-likelihood. If hyper is not None, it takes the vector of hyper parameters
            as second argument.
        hyper : np.ndarray, optional
            Array of hyper-parameters. Note that if you have several hyper parameters, you
            have to make sure yourself, that it ends up in the right misfit function.
        hyperProposal : Callable, optional
            Only needed if hyper is also given. Defines how to change the hyper parameters.
        hyperPrior : Callable, optional
            Prior of the hyper parameters. Defaults to a constant prior.
        temperature : float, optional
            Used in Parallel Tempering. If the temperature is larger than 1., the likelihood function is flattened.
            In the limit temperature -> infty, the chain behaves as a random walk.
        birth_factor : float, optional
            Not recommened for general use. Increases the likelihood of proposing a birth step. As a result, the 
            acceptance factor for the birth step has to be reduced accordingly. I have not seen any improvements when
            using this. Setting it to 0 might be useful in some situations though?
    Attributes
    -------
        model : obj
            Current state of the chain
        y
            The current state of the forward calculator result. Typically an np.ndarray
        oldL : float
            The last value of the likeliood
        lastU : int
            The last type of step taken, 0=birth, 1=death, 2=change, 3=move, 4=hyper parameter
        lastAccepted : bool
            True, if the last step was accepted.
        lastProposalRatio : float
            The acceptance factor coming from the proposal. >1 for birth steps, <1 for death steps, otherwise 1.
    """
    def __init__(self,startModel,forward,proposal,misfit,hyper=None,
                hyperProposal=None,hyperPrior=None,temperature=1.0,birth_factor=1.0):
        self.model = startModel
        self.proposal = proposal
        self.misfit = misfit
        self.forward = forward
        self.birth_factor = birth_factor
        self.y = self.forward(self.model)
        
        self.hyper = hyper
        self.hyperProposal = hyperProposal
        if hyperPrior is None:
            self.hyperPrior = lambda x:1.0
        else:
            self.hyperPrior = hyperPrior
            
        if hyper is None:
            self.p_vec = np.array([0.25*birth_factor,0.25,0.25,0.25])
        else:
            self.p_vec = np.array([0.2*birth_factor,0.2,0.2,0.2,0.2])
        self.p_vec = self.p_vec / self.p_vec.sum()
        
        self.temperature = temperature
        self.oldL = self._calc_L()
        
    def _calc_L(self):
        if self.hyper is None:
            L = self.misfit(self.y) + self.proposal.log_prior(self.model) 
        else:
            L = self.misfit(self.y,self.hyper) + self.proposal.log_prior(self.model) + self.hyperPrior(self.hyper)
        return L
        
    def step(self):
        oldModel = self.model.copy()
        accepted = False
        acceptance_factor = 1.0
        if self.hyper is None:
            u = np.random.choice(4,size=1,p=self.p_vec)
            hypercan = None
        else:
            u = np.random.choice(5,size=1,p=self.p_vec)
            hypercan = self.hyper.copy()
        if u==0:
            # Birth
            candidate = self.proposal.birth(self.model,self.temperature)
            acceptance_factor = self.p_vec[1]/self.p_vec[0]
        elif u==1:
            # Death
            candidate = self.proposal.death(self.model,self.temperature)
            acceptance_factor = self.p_vec[0]/self.p_vec[1]
        elif u==2:
            # Change
            candidate = self.proposal.change_param(self.model)
        elif u==3:
            # Move
            candidate = self.proposal.move(self.model)
        elif u==4:
            hyperChanged,dh = self.hyperProposal()
            hypercan[hyperChanged] = hypercan[hyperChanged] + dh
            candidate = self.model.copy()
            ycan = self.y.copy()
            
        
        if u<4:
            proposalRatio = self.proposal.lastProposalRatio
            ycan = self.forward(candidate)
        else:
            proposalRatio = 1.0
        
        if self.hyper is None:
            newL = self.misfit(ycan) + self.proposal.log_prior(candidate)
        else:
            newL = self.misfit(ycan,hypercan) + self.proposal.log_prior(candidate) + self.hyperPrior(hypercan)
        
        alpha = np.exp((newL-self.oldL)/self.temperature) * proposalRatio * acceptance_factor
        self.last_alpha = np.minimum(alpha,1)
        self.lastU = u
        r = np.random.rand()
        if r < alpha:
            self.model = candidate
            self.oldL = newL
            self.hyper = hypercan
            self.y = ycan
            accepted = True
            
        self.lastAccepted = accepted
        
class PointModel:
    """Model with variable number of points distributed in a rectangular domain with a property attached to each point

    Parameters
    ----------
    points : np.ndarray(shape=(n,m))
        There are n points in a m-dimensional space.
    props :  np.ndarray(shape=(n,m2))       
        There are n points and each point has m2 assigned properties.
    interpolator_constructor : function
        This function must a return an interpolating function. For example most interpolators in scipy.interpolate
        work like this.
    """
       
    def __init__(self,points,props,interpolator_constructor=scipy.interpolate.NearestNDInterpolator):
        assert points.shape[0] == props.shape[0]
        self.points = points
        self.props = props
        self.interpolator_constructor = interpolator_constructor
        self.interpolator = interpolator_constructor(points,props)
        
    @property
    def k(self):
        """Number of points in model"""
        return len(self.points)

    @property
    def spatial_dim(self):
        """Spatial dimensionality"""
        return self.points.shape[1]
    
    def __call__(self,x):
        """Interpolate points and props onto the points in x.
        """
        return self.interpolator(x)
    
    def copy(self):
        return PointModel(self.points.copy(),self.props.copy(),self.interpolator_constructor)
    
    def birth(self,new_point,new_prop):
        """Return a new PointModel with additional an additional point.
        """
        new_points = np.vstack((self.points,new_point))
        new_props = np.vstack((self.props,new_prop))
        return PointModel(new_points,new_props,self.interpolator_constructor)

    def death(self,ksel):
        new_points = np.delete(self.points,ksel,axis=0)
        new_props = np.delete(self.props,ksel,axis=0)
        return PointModel(new_points,new_props,self.interpolator_constructor)

    def change_param(self,ksel,new_props):
        return PointModel(self.points,new_props,self.interpolator_constructor)
    
    def move(self,ksel,new_points):
        return PointModel(new_points,self.props,self.interpolator_constructor)

class VoronoiModel(PointModel):
    """
    This is basically not needed any more, because PointModel can do the same thing and more
    """
    def __init__(self,points,props):
        super().__init__(points,props)

class ShapeModel:
    """Abstract super class for spatial models consisting of discrete shapes
    
    This class uses a lot of memorization to speed up the evaluation in a
    MCMC context. Thus, they can end up needing a lot of memory.
    In order to define your own version, you just need to implement a 
    `calc_mask` method.

    Parameters
    ----------
    points : np.ndarray(shape=(n,m))
        The centers of the spatial shapes. Actually, it could be any point
        defining the location of the shape (i.e. upper left corner). But
        in my code I just use it as the center.
    props :  np.ndarray(shape=(n,m2))       
        The properties attached to each shape. By convention, the first
        couple of parameters describe the size of the shape and things
        like rotation. The later elements of props are then the atctual
        parameter of interest such as velocity or density.
    xi,yi : np.ndarray
        The location of the points at which to voxelize the model
    """
    @classmethod
    def create_from_vectors(cls,points,props,xi,yi):
        model = cls()
        model.points = points
        model.props = props
        model.xi = xi
        model.yi = yi
        model.masks = np.zeros((model.k,)+xi.shape,dtype=bool)
        for i in range(model.k):
            model.calc_mask(i)
        model.z = model.calc_grid()
        return model
    
    def __init__(self):
        pass 
    
    @property
    def spatial_dim(self):
        return self.points.shape[1]
    
    @property
    def k(self):
        return self.points.shape[0]
    
    def calc_grid(self):
        return np.einsum('ijk,il->jkl',self.masks,self.values)
    
    def __call__(self,x=None):
        return self.z
    
    def birth(self,new_point,new_prop):
        model = type(self)()
        model.xi = self.xi
        model.yi = self.yi
        new_points = np.vstack((self.points,new_point))
        new_props = np.vstack((self.props,new_prop))
        model.points = new_points
        model.props = new_props
        model.masks = np.zeros((model.k,)+self.xi.shape,dtype=bool)        
        model.masks[:self.k] = self.masks
        model.calc_mask(self.k)
        model.z = self.z + model.masks[self.k][:,:,None] * model.values[self.k]
        return model
    
    def death(self,ksel):
        model = type(self)()
        model.xi = self.xi
        model.yi = self.yi
        new_points = np.delete(self.points,ksel,axis=0)
        new_props = np.delete(self.props,ksel,axis=0)
        model.points = new_points
        model.props = new_props
        model.masks = np.delete(self.masks,ksel,axis=0)     
        model.z = self.z - self.masks[ksel][:,:,None] * self.values[ksel]
        return model
    
    def change_param(self,ksel,new_props):
        model = type(self)()
        model.xi = self.xi
        model.yi = self.yi
        model.points = self.points
        model.props = new_props
        model.masks = self.masks.copy()
        model.calc_mask(ksel)
        model.z = self.z - self.masks[ksel][:,:,None] * self.values[ksel] + model.masks[ksel][:,:,None] * model.values[ksel]
        return model
    
    def move(self,ksel,new_points):
        model = type(self)()
        model.xi = self.xi
        model.yi = self.yi
        model.points = new_points
        model.props = self.props
        model.masks = self.masks.copy()
        model.calc_mask(ksel)
        model.z = self.z - self.masks[ksel][:,:,None] * self.values[ksel] + model.masks[ksel][:,:,None] * model.values[ksel]
        return model
    
    def copy(self):
        model = type(self)()
        model.xi = self.xi
        model.yi = self.yi
        model.points = self.points
        model.props = self.props
        model.masks = self.masks
        model.z = self.z
        return model

class RotoRectorModel(ShapeModel):
    """Shape model consisting of rotated rectangles
    """
    @property
    def sizes(self):
        return self.props[:,:self.spatial_dim]
    
    @property
    def rotations(self):
        return self.props[:,self.spatial_dim]
        
    @property
    def values(self):
        return self.props[:,self.spatial_dim+1:]
    
    def calc_mask(self,i):
        sin,cos = np.sin(self.rotations[i]/180.0*np.pi),np.cos(self.rotations[i]/180.0*np.pi)
        rot_x = (self.xi-self.points[i,0]) * cos +(self.yi-self.points[i,1]) * sin
        rot_y = (self.yi-self.points[i,1]) * cos - (self.xi-self.points[i,0]) * sin
        mask = (np.abs(rot_x) < self.sizes[i,0]/2) & (np.abs(rot_y) < self.sizes[i,1]/2)
        self.masks[i] = mask
        
class EllipticalBlobModel(ShapeModel):
    """Shape model consisting of rotated ellipses
    """
    @property
    def sizes(self):
        return self.props[:,:self.spatial_dim]
    
    @property
    def rotations(self):
        return self.props[:,self.spatial_dim]
        
    @property
    def values(self):
        return self.props[:,self.spatial_dim+1:]
    
    def calc_mask(self,i):
        sin,cos = np.sin(self.rotations[i]/180.0*np.pi),np.cos(self.rotations[i]/180.0*np.pi)
        rot_x = (self.xi-self.points[i,0]) * cos +(self.yi-self.points[i,1]) * sin
        rot_y = (self.yi-self.points[i,1]) * cos - (self.xi-self.points[i,0]) * sin
        # rhs is 0.25 bc. diameter is given, not semi-axis
        mask = rot_x**2/self.sizes[i,0]**2 + rot_y**2/self.sizes[i,1]**2 <= 0.25 
        self.masks[i] = mask


def compress_models(models):
    """Store points and props of a list of models in array
    """
    models = np.asarray(models)
    kmax = max(m.k for m in models.flat)
    n_vars = models.flat[0].points.shape[1] + models.flat[0].props.shape[1]
    spatial_dim = models.flat[0].spatial_dim
    out_model_store = np.ones(models.shape+(kmax,n_vars)) * np.nan
    it = np.nditer(models,flags=['multi_index','refs_ok'])
    for _ in it:
        m = models[it.multi_index]
        out_model_store[it.multi_index][:m.k,:spatial_dim] = m.points
        out_model_store[it.multi_index][:m.k,spatial_dim:] = m.props
    return out_model_store

def decompress_models(compressed_models,constructor_func,spatial_dim,*args,**kwargs):
    """Create transdim model objects from compressed storage array
    """
    if compressed_models.ndim == 2:
        if np.any(np.isnan(compressed_models[:,0])):
            k = np.argmax(np.isnan(compressed_models[:,0]))
        else:
            k = compressed_models.shape[0]
        return constructor_func(compressed_models[:k,:spatial_dim],compressed_models[:k,spatial_dim:],*args,**kwargs)
    elif compressed_models.ndim == 3:
        models = []
        for i in range(compressed_models.shape[0]):
            models.append(decompress_models(compressed_models[i],constructor_func,spatial_dim,*args,**kwargs))
        return models
    else:
        raise ValueError('Cannot decompress models, because ndim > 3')
    

def create_submodel(model,sel,smart=True):
    """Create a new model by selecting only a subset of the shapes
    TODO: Should work with any Point Model, so far it only works with RotoRector
    """
    if smart:
        sub_model = RotoRectorModel.__new__(RotoRectorModel)
        sub_model.points = model.points[sel]
        sub_model.props = model.props[sel]
        sub_model.masks = model.masks[sel]
        sub_model.xi = model.xi
        sub_model.yi = model.yi
        sub_model.z = sub_model.calc_grid()
        return sub_model
    else:
        return RotoRectorModel.create_from_vectors(model.points[sel],model.props[sel],model.xi,model.yi)

def make_offspring(model1,model2,n_offspring,selection_chance=0.5):
    """Make a new model by randomly combining shapes from two 'parents'
    """
    joint_points = np.concatenate((model1.points,model2.points),axis=0)
    joint_props = np.concatenate((model1.props,model2.props),axis=0)
    joint_masks = np.concatenate((model1.masks,model2.masks),axis=0)
    joint_model = RotoRectorModel.__new__(RotoRectorModel)
    joint_model.points = joint_points
    joint_model.props = joint_props
    joint_model.masks = joint_masks
    joint_model.xi = model1.xi
    joint_model.yi = model1.yi
    joint_model.z = joint_model.calc_grid()
    
    offspring = []
    for i in range(n_offspring):
        rando_size = max(1,np.random.binomial(joint_points.shape[0],selection_chance))
        sel = np.random.choice(joint_points.shape[0],rando_size,replace=False)
        offspring.append(create_submodel(joint_model,sel,smart=True))
    return offspring

def recombination_step(ensemble,fitness,threshold=90,n_offspring=10,target_size=None):
    """Take the elite of a model ensemble and generate their offspring
    Only those models whose fitness is better than percentile `threshold`
    are used for generating offspring. Each pair of models produces `n_offspring`
    offsprings.
    """
    if target_size is None:
        target_size = len(ensemble)
    
    # Select the 100 - threshold % best models
    elite = np.where(fitness >= np.percentile(fitness,threshold))[0]
    # Generate offspring of the elite
    offspring = []
    if target_size <= len(elite):
        print('Warning: target_size too small in recombination step, only parents are returned')

    while len(offspring) < target_size - len(elite):
        i1,i2 = np.random.choice(elite,2,replace=False)
        model1,model2 = ensemble[i1],ensemble[i2]
        offspring.extend(make_offspring(model1,model2,n_offspring))
    offspring.extend([ensemble[e] for e in elite])
    new_ensemble = offspring
    return new_ensemble


class TransDimPointProposal:
    """Proposal for model with variable number of points distributed in a rectangular domain

    Parameters
    ----------
    xlim : np.array
        Defines the limits of valid points
    prop_proposal : distribution
        Proposal distributions for the properties. Needs to have a .rvs() that returns an array with changes to the parameters
    prop_prior : distribution
        Proposal prior for the properties. Needs to have .logpdf(props) that returns the log likelihood for an array of 
        properties
    move_proposal : distribution
        Move proposal distribution. Needs to have a .rvs() that returns an array with changes for each spatial dimension.
    birth_form_prior : boolean
        If True, new properties for a birth step will be drawn using prop_prior.rvs(). If False, new properties will
        be drawn from the value of the new location and perturbed by prop_proposal.rvs().
    strict_bound_axis : list of int
        By default, only the points are compared to the `xlim`. Along the axes specified by this list, the sizes (if model has sizes)
        will be taken into account. This is useful for spherical coordinates, where out-of-bounds are a problem mainly in the radial
        component.
    """
    def __init__(self,xlim,prop_proposal,prop_prior,move_proposal,birth_from_prior=False,strict_bound_axis=[]):
        self.xlim = np.asarray(xlim)
        self.ndim = self.xlim.shape[0]
        self.prop_proposal = prop_proposal
        self.prop_prior = prop_prior
        self.move_proposal = move_proposal
        self.area = np.prod(self.xlim[:,1]-self.xlim[:,0])
        self.birth_from_prior = birth_from_prior
        self.strict_bound_axis = strict_bound_axis
        
    def _sample_uniform(self,lims):
        return np.random.random(self.ndim) * (lims[:,1] - lims[:,0]) + lims[:,0]
        
    def _normal_pdf(self,val,sigma):
        # Why sigmaProposal???
        return np.exp(-0.5*(val/sigma)**2) / (np.sqrt(2*np.pi) * self.sigmaProposal)
    
    def birth(self,model,temperature=1):
        k = model.k
        points = np.copy(model.points)
        props = np.copy(model.props)
        
        new_point = self._sample_uniform(self.xlim)
        old_prop = model(new_point)
        if self.birth_from_prior:
            new_prop = self.prop_prior.rvs()
        else:
            new_prop = old_prop + self.prop_proposal.rvs()
        
        if self.birth_from_prior:
            self.lastProposalRatio = 1.0/(k+1) * (self.area/self.prop_prior.pdf(new_prop))**(1.0/temperature)        
        else:
            self.lastProposalRatio = self.area/(k+1) / self.prop_proposal.pdf(old_prop-new_prop)        
        
        return model.birth(new_point,new_prop)
    
    def death(self,model,temperature=1):
        k = model.k
        if k==1:
            self.lastProposalRatio = 0.0
            return model.copy()
        
        points = np.copy(model.points)
        props = np.copy(model.props)

        ksel = np.random.randint(0,k)
        old_point = points[ksel]
        old_prop = props[ksel]
        new_model = model.death(ksel)
        if self.birth_from_prior:
            self.lastProposalRatio = k * (self.prop_prior.pdf(old_prop)/self.area)**(1.0/temperature)
        else:
            new_prop = new_model(old_point)
            self.lastProposalRatio = k / self.area * self.prop_proposal.pdf(new_prop-old_prop)        
        return new_model
        
    def change_param(self,model):
        k = model.k
        props = np.copy(model.props)
        ksel = np.random.randint(0,k)
        props[ksel] = props[ksel] + self.prop_proposal.rvs()
        self.lastProposalRatio = 1.0
        return model.change_param(ksel,props)
        
    def move(self,model):
        k = model.k
        points = np.copy(model.points)
        ksel = np.random.randint(0,k)
        points[ksel] = points[ksel]  + self.move_proposal.rvs()
        self.lastProposalRatio = 1.0
        return model.move(ksel,points)
            
    def _lazy_compare(self,vals,lim):
        return np.all( (np.array(vals) >= lim[0]) & (np.array(vals) <= lim[1]) )
    
    def _check_bounds(self,model):
        prob_1 = ((model.points-0.5*model.sizes)<self.xlim[:,0])[:,self.strict_bound_axis]
        prob_2 = ((model.points+0.5*model.sizes)>self.xlim[:,1])[:,self.strict_bound_axis]
        return(prob_1|prob_2)

    def log_prior(self,model):
        if np.any((model.points>self.xlim[:,1]) | (model.points<self.xlim[:,0])):
            return -np.inf
        elif self.strict_bound_axis and np.any(self._check_bounds(model)):
            return -np.inf
        else:
            k = model.k
            props = model.props
            point_prior = -k * np.log(self.area)
            prop_prior = np.sum(self.prop_prior.logpdf(props))
        return point_prior + prop_prior

class VariableBirthTransDimProposal(TransDimPointProposal):
    def __init__(self,xlim,prop_proposal,prop_prior,move_proposal,prop_birth,strict_bound_axis=[]):
        super().__init__(xlim,prop_proposal,prop_prior,move_proposal,strict_bound_axis=strict_bound_axis)
        self.prop_birth = prop_birth
    
    def birth(self,model,temperature=1):
        k = model.k
        
        new_point = self._sample_uniform(self.xlim)
        new_prop = self.prop_birth.rvs()
        self.lastProposalRatio = 1.0/(k+1) * (self.area/self.prop_birth.pdf(new_prop))**(1.0/temperature)        
        
        return model.birth(new_point,new_prop)
    
    def death(self,model,temperature=1):
        k = model.k
        if k==1:
            self.lastProposalRatio = 0.0
            return model.copy()
        
        points = np.copy(model.points)
        props = np.copy(model.props)

        ksel = np.random.randint(0,k)
        old_point = points[ksel]
        old_prop = props[ksel]
        new_model = model.death(ksel)
        self.lastProposalRatio = k * (self.prop_birth.pdf(old_prop)/self.area)**(1.0/temperature)     
        return new_model
        

class LinearModel:
    """One dimensional model with linear interpolation
    
    Parameters
    ----------
        xv : np.ndarray(ndim=1)
            x values of the nodes
        yv : np.ndarray(ndim=1)
            y values of the nodes
    """
    def __init__(self,xv,yv):
        self.xv = xv
        self.yv = yv
        self.k = len(xv)
        self.interpolator = scipy.interpolate.interp1d(self.xv,self.yv)

    def __str__(self):
        s1 = (str(self.xv))
        s2 = (str(self.yv))
        return s1 + " " + s2
        
    def __call__(self,x):
        return self.interpolator(x) 

    def __copy__(self):
        return LinearModel(self.xv,self.yv)

    def copy(self):
        return LinearModel(self.xv,self.yv)

class LinearDiscreteProposal:
    """Discrete proposal for a LinearModel

    Discrete proposal means that the x values are unique and cannot be changed.

    Parameters
    ----------
    xval : np.ndarray
        Set of the possible x locations
    ylim : list of float
        Minimum and maximum of the y values
    xStep : int, optional
        How far an active node can be moved (in number of points in x)
    yStepRel : float, optional
        How much can the value be changed. The actual standard deviation of the value is relative to the ylim.
    birthSigmaRel : float, optional
        How much a newly birthed point is changed relative to the previous value. The standard deviation is calcuated
        based on ylim.
    maxK : int, optional
        Set a maximum number of nodes

    """
    def __init__(self,xval,ylim,xStep=1,yStepRel=0.05,birthSigmaRel=2.0e-1,maxK=np.inf):
        self.xval = set(xval)
        self.ylim = ylim
        self.sigmaProposal = yStepRel * (ylim[1] -ylim[0])
        self.xStep = xStep
        self.N = len(xval)
        self.xlim = [min(self.xval),max(self.xval)]
        self.birthSigma = birthSigmaRel * (ylim[1] -ylim[0])
        self.maxK = maxK

    def _sample_uniform(self,lims):
        return np.random.random() * (lims[1] - lims[0]) + lims[0]
        
    def _normal_pdf(self,val,sigma):
        return np.exp(-0.5*(val/sigma)**2) / (np.sqrt(2*np.pi) * self.sigmaProposal)
    
    def _lazy_compare(self,vals,lim):
        return np.all( (np.array(vals) >= lim[0]) & (np.array(vals) <= lim[1]) )    
    
    def createModel(self,k):
        if k<2:
            return None
        elif k==2:
            xv = np.array([(min(self.xval),max(self.xval))]).flatten()
        elif k>2:
            xrem = self.xval.difference([min(self.xval),max(self.xval)])
            xv2 = random.sample(xrem,k-2)
            xv = np.zeros((k))
            xv[0] = min(self.xval)
            xv[1] = max(self.xval)
            xv[2:] = xv2
        yv = np.random.rand(k) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        
        return LinearModel(xv,yv)
        
        
    def birth(self,model,temperature=1.0):
        k = model.k
        if k == self.N:
            self.lastProposalRatio = 1.0
            return copy.copy(model)
        xset = set(model.xv)
        xv = np.copy(model.xv)
        yv = np.copy(model.yv)
        
        xrem = self.xval.difference(xset)
        newX = random.sample(xrem,1)[0]
        oldY = model(newX)
        newY = oldY + np.random.randn() * self.birthSigma
        
        
        xv = np.append(xv,newX)
        yv = np.append(yv,newY)
        
        self.lastProposalRatio = (self.N -2.0 - k)/(k+1.0-2.0) \
           / self._normal_pdf(newY-oldY,self.birthSigma)  
        
        return LinearModel(xv,yv)
    
    def death(self,model,temperature=1.0):
        k = model.k
        xv =np.copy(model.xv)
        yv = np.copy(model.yv)
        # There need to be at least two values (at the ends), these
        # cannot be removed
        if k>2:
            ksel = np.random.randint(0,k-2)
            oldX = xv[ksel+2]
            oldY = yv[ksel+2]
            xv = np.delete(xv,ksel+2)
            yv = np.delete(yv,ksel+2)
            newModel = LinearModel(xv,yv)
            newY = newModel(oldX)
            self.lastProposalRatio = 1.0*(k - 2.0) / (self.N-2.0-k+1.0) * self._normal_pdf(newY-oldY,self.birthSigma)
            return newModel
        else:
            self.lastProposalRatio = 0.0
            return LinearModel(xv,yv)
    
    def move(self,model):
        k = model.k
        xv = np.copy(model.xv)
        yv = np.copy(model.yv)
        
        if k==2:
            self.lastProposalRatio=1.0
            return LinearModel(xv,yv)
        
        ksel = np.random.randint(0,k-2)
        step = np.random.randint(1,self.xStep+1)
        
        xvalArr = np.sort(np.array(list(self.xval)))
        index = np.where(np.abs(xvalArr - xv[ksel+2]) < 1.0e-5)[0]
        u = np.random.rand()
        if u<0.5:
            newIndex = index + step
        else:
            newIndex = index - step
        
        if newIndex < 0 or newIndex > self.N-1:
            self.lastProposalRatio = 1.0
            return LinearModel(xv,yv)
        newX = xvalArr[newIndex]
        
        if newX in xv:
            self.lastProposalRatio = 1.0
            return LinearModel(xv,yv)
        xv[ksel+2] = newX
       
        self.lastProposalRatio = 1.0
        return LinearModel(xv,yv)
    
    def change_param(self,model):
        k = model.k
        xv = np.copy(model.xv)
        yv = np.copy(model.yv)
        ksel = np.random.randint(0,k)
        yv[ksel] = yv[ksel] + np.random.randn() * self.sigmaProposal
        self.lastProposalRatio = 1.0
        return LinearModel(xv,yv)
    
    def log_prior(self,model):
        k = model.k
        xv = np.copy(model.xv)
        yv = np.copy(model.yv)
        ok1 = self._lazy_compare(xv,self.xlim)
        ok2 = self._lazy_compare(yv,self.ylim)
        ok3 = k<=self.maxK
        if ok1 and ok2 and ok3:
            if k==2:
                return -np.log(1) - k * np.log(self.ylim[1] - self.ylim[0])
            else:
                return -(self.N - 2)*np.log(self.N-2) + (k-2)*np.log(k-2) + \
                    (self.N-k)*np.log(self.N-k) - k * np.log(self.ylim[1] - self.ylim[0])
        else:
            return -np.inf    

def create_submodel(model,sel,smart=True):
    """Create a new model by selecting only a subset of the shapes
    TODO: Should work with any Point Model, so far it only works with RotoRector
    """
    if smart:
        sub_model = RotoRectorModel.__new__(RotoRectorModel)
        sub_model.points = model.points[sel]
        sub_model.props = model.props[sel]
        sub_model.masks = model.masks[sel]
        sub_model.xi = model.xi
        sub_model.yi = model.yi
        sub_model.z = sub_model.calc_grid()
        return sub_model
    else:
        return RotoRectorModel.create_from_vectors(model.points[sel],model.props[sel],model.xi,model.yi)

def make_offspring(model1,model2,n_offspring,selection_chance=0.5):
    """Make a new model by randomly combining shapes from two 'parents'
    """
    joint_points = np.concatenate((model1.points,model2.points),axis=0)
    joint_props = np.concatenate((model1.props,model2.props),axis=0)
    joint_masks = np.concatenate((model1.masks,model2.masks),axis=0)
    joint_model = RotoRectorModel.__new__(RotoRectorModel)
    joint_model.points = joint_points
    joint_model.props = joint_props
    joint_model.masks = joint_masks
    joint_model.xi = model1.xi
    joint_model.yi = model1.yi
    joint_model.z = joint_model.calc_grid()
    
    offspring = []
    for i in range(n_offspring):
        rando_size = max(1,np.random.binomial(joint_points.shape[0],selection_chance))
        sel = np.random.choice(joint_points.shape[0],rando_size,replace=False)
        offspring.append(create_submodel(joint_model,sel,smart=True))
    return offspring

def transdimensional_MCMC(startModel,forward,proposal,misfit,nruns,
    hyper=None,hyperProposal=None,hyperPrior=None,shutup=True,chainSave=1,callback = lambda i:None):
    """Run transdimensional MCMC for a given setup
    
    Parameters
    ----------
        see TransDimChain above...
        nruns : int
            How many iterations to run
        shutup : bool, optional
            If true, every 1% of the progress a message is printed to stdout.
        chainSave : int, optional
            If >1, only every 1 in chainSave models is saved and returned by the function.
    """
    models=[]
    models.append(startModel)
    perc = 0
    if not hyper is None:
        hypers = np.zeros((nruns//chainSave,hyper.shape[0]))
        accepted = np.zeros((5))
    else:
        hypers = None
        accepted = np.zeros((4))
    model = startModel
    Ls = np.zeros((nruns//chainSave))
    chain = TransDimChain(startModel,forward,proposal,misfit,hyper,hyperProposal,hyperPrior)
    for i in range(nruns):
        if (100*i)/nruns > perc:
            perc = perc + 1
            if not shutup:
                print("%d/%d runs" % (i,nruns))
    
        chain.step()
        if chain.lastAccepted:
            accepted[chain.lastU] = accepted[chain.lastU] + 1
        
        callback(i)

        if i%chainSave == 0:
            models.append(copy.copy(chain.model))
            Ls[i//chainSave] = chain.oldL
            if not hyper is None:
                hypers[i//chainSave,:] = chain.hyper
    if not hyper is None:
        return models,Ls,accepted,hypers
    else:
        return models,Ls,accepted
    
def create_model(constructor_func,proposal,model_complexity,*args,**kwargs):
    n_props = len(proposal.prop_prior.rvs())
    props = np.zeros((model_complexity,n_props))
    points = np.random.random((model_complexity,proposal.ndim)) * (proposal.xlim[:,1] - proposal.xlim[:,0]) + proposal.xlim[:,0]
    for i in range(model_complexity):
        props[i] = proposal.prop_prior.rvs()
    return constructor_func(points,props,*args,**kwargs)

def create_models(constructor_func,proposal,model_complexity,number_of_models,*args,**kwargs):
    return [create_model(constructor_func,proposal,model_complexity,*args,**kwargs) for _ in range(number_of_models)]