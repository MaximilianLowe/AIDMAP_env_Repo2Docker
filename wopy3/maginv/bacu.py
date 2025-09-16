"""BAyesian CUrie depth estimation
"""
import wopy3.maginv.SH_filter as SH_filter
import wopy3.maginv.mag_inv as mag_inv
import wopy3.func_dump as func_dump
import wopy3.utils as utils

import numpy as np
import functools

class MaternApproximator:
    """Interpolation based calculation of Matern covariance function

    Instances of this class can be called like `func_dump.C_matern`,
    but are typically faster, because interpolation is used.

    Parameters
    ----------
    nx : int
        Number of nodes for interpolation
    spacing : 'uniform' or 'root', optional
        How the nodes should be distributed. If root, the nodes are
        more concentrated near 0, because for most large x the
        matern function is zero anyway, unless the range is
        extremely large.
    """
    def __init__(self,nx,spacing='uniform'):
        self.nx = nx
        if spacing == 'uniform':
            self.generator = lambda pd_max:np.linspace(0,pd_max,self.nx)
        elif spacing == 'root':
            self.generator = lambda pd_max:np.linspace(0,np.sqrt(pd_max),self.nx)**2

    def interp1d(self,pd,pars):
        """Carry out interpolation
        
        Parameters
        ----------
        pd : np.array
            Distance values units must fit to pars
        pars : iterable
            Variance, nu and range value of the matern function to evaluate
        
        Returns
        -------
        C : np.array
            Covariance between points described by `pd`
        """
        xax = self.generator(pd.max())
        yax = func_dump.C_matern(xax,(pars[0],pars[1],pars[2]))
        return np.interp(pd,xax,yax)

def synthetic_generation(lon,lat,design_matrix,chi_0,bot_0,chi_pars=None,bot_pars=None):
    """Generate a random model of susceptibility and/or thickness

    Parameters
    ----------
    lon : np.array
        Longitudes
    lat : np.array
        Latitudes
    design_matrix : tuple of np.array
        Contains the spherical harmonic synthesis matrices for the cosine and sine
        components.
    chi_0 : float
        Reference susceptibility
    bot_0 : float
        Reference depth
    chi_pars : tuple,optional
        Contains the covariance parameters for susceptibility (order: sigma,nu,rho).
        If not given, only a constant `chi_0` is returned.
    bot_pars : tuple,optional
        Contains the covariance parameters for depth (order: sigma,nu,rho).
        If not given, only constant `bot_0` is returned.

    Returns
    -------
    syn_chi : np.array
        Grid of susceptibility values
    syn_bot : np.array
        Grid of depths values
    """
    
    lmax = int((np.sqrt(2*design_matrix[0].shape[1]/2*8+1)-3)/2)    
    n_gam = max(3*lmax,400)
    shape = (lat.size,lon.size)
    if not chi_pars is None:
        func_susc = lambda d:func_dump.C_matern(d,(1.0,chi_pars[1],chi_pars[2]))
        temp_chi,_,_ = SH_filter.generation_helper(lon,lat,design_matrix,func_susc,n_gam = n_gam)
        syn_chi = chi_0 + temp_chi*np.sqrt(chi_pars[0])
    else:
        syn_chi = np.ones(shape) * chi_0

    if not bot_pars is None:
        func_bot = lambda d:func_dump.C_matern(d,(1.0,bot_pars[1],bot_pars[2]))
        temp_bot,_,_  = SH_filter.generation_helper(lon,lat,design_matrix,func_bot,n_gam = n_gam)
        syn_bot = bot_0 + temp_bot*np.sqrt(bot_pars[0])
    else:
        syn_bot = np.ones(shape) * bot_0
    
    return syn_chi,syn_bot

def forward_calc(lon,lat,syn_chi,syn_bot,magGrid,station_fname,tess_fname='test.tess'):
    loni,lati = np.meshgrid(lon,lat)
    tesses = mag_inv.magnetize_tesseroids_from_interface(loni,lati,np.zeros(loni.shape),syn_bot,syn_chi,magGrid=magGrid)
    with open(tess_fname,'w') as f:
        for tess in tesses:
            f.write(tess)
    syn_B = mag_inv.run_magtess(tess_fname,station_fname,'z')[:,3].reshape(loni.shape)
    return syn_B

def calc_sensitivity_matrices(lon,lat,chi_0,bot_0,magGrid,lons=None,lats=None,station_height=100e3,
                verbose=False,temp_stat='stations.txt',temp_tess='temp.tess'):
    loni,lati = np.meshgrid(lon,lat)
    n = loni.size
    if lons is None:
        lons = loni.flatten()
        lats = lati.flatten()

    A = np.zeros((lons.size,n))
    B = np.zeros((lons.size,n))

    chi_0 = chi_0.reshape(loni.shape)
    bot_0 = bot_0.reshape(loni.shape)
    
    stations = np.vstack((lons,lats,np.ones(lons.size)*station_height)).T
    np.savetxt(temp_stat,stations,fmt='%.2f')
    
    if verbose:
        print('Calculating Susc sensitivity matrix ...')
    
    tesses = mag_inv.magnetize_tesseroids_from_interface(loni,lati,np.zeros(loni.shape),bot_0,np.ones(chi_0.shape),magGrid=magGrid)
    for i,tess in enumerate(tesses):
        with open(temp_tess,'w') as f:
            f.write(tess)
        A[:,i] = mag_inv.run_magtess(temp_tess,temp_stat,'z')[:,3]
    
    if verbose:
        print('Calculating Bot sensitivity matrix ...')
        
    tesses = mag_inv.magnetize_tesseroids_from_interface(loni,lati,bot_0,bot_0+1,chi_0,magGrid=magGrid)
    for i,tess in enumerate(tesses):
        with open(temp_tess,'w') as f:
            f.write(tess)
        B[:,i] = mag_inv.run_magtess(temp_tess,temp_stat,'z')[:,3]

    return A,B

class SensitivityMatrixStore:
    """Calculate approximate response of tesseroid model using lookup-table

    Objects of this class are constructed with a fixed grid of tesseroids and station locations. Then,
    the magnetic tesseroid programm by Baykiev et al. is run for each tesseroid location and for a number
    of tesseroid depths defined by `bots`. 
    
    Constructing this lookup-table takes a while, but once it is created, the response from an arbitrary
    model of tesseroid thicknesses at that location can be easily calculated. Additionally, we can
    calculate the derivative of the tesseroid response w.r.t. to susceptibility and thickness.

    As it is now, both the measured component is assumed to be purely vertical,
    but that could be easily changed in the future.

    Parameters
    ----------
    lon : np.ndarray(ndim==1)
        Vector of longitude values defining the tesseroid grid
    lat : np.ndarray(ndim==1)
        Vector of latitude values
    magGrid: np.ndarray (ndim==3)
        Shape is 3 x len(lat) x len(lon). It gives the magnetization in terms of 
        ENU at each location.
    bots : np.ndarray(ndim==1)
        The depth_steps to be used. For the best results, the number of steps in bots
        should be as high as possible, however larger values also mean higher computation
        time. The range of bots should also be large enough to capture the entire range of
        possible values, as no extrapolation will be done. (See also `bounds_error` below)
    station_height : float
        Constant height of the measurements in meters
    verbose : bool, optional
        Whether ETA messages should be output
    bounds_error : bool,optional
        Whether an exception should be raised when one of the requested thicknesses
        is out of the bounds defined by `bots`. If False, nan is returned instead.
    stat_file : str
        Name of the temporary file used to communicate the stations to tessbz.
        The file will not be deleted automatically.
    tess_file : str
        Name of the temporary tesseroid file used to communicate with tessbz.
        The file will not be deleted automatically.
    lons : np.ndarray(ndim==1),optional
        Longitude values of magnetic stations. These are not automatically
        extended to a grid! They **can** be irregular. 
        If None, the stations are assumed to coincide with the tesseroid centers
    lats : np.ndarray(ndim==1),optional
        Latitude values of magnetic stations. These are not automatically
        extended to a grid! They **can** be irregular.
    """
    def __init__(self,lon,lat,magGrid,bots,station_height=300e3,verbose=False,bounds_error=True,
                stat_file="stations.txt",tess_file="temp.tess",lons=None,lats=None):
        self.lon = lon
        self.lat = lat
        loni,lati = np.meshgrid(lon,lat)
        if lons is None:
            lons = loni.flatten()
            lats = lati.flatten()
        self.magGrid = magGrid
        self.bots = np.sort(bots)
        self.station_height = station_height
        self.bounds_error = bounds_error
        n_steps = len(bots)
        self.store = np.zeros((2,n_steps,lons.size,loni.size))

        timer = utils.ETA(n_steps)
        for k in range(n_steps):
            mat_A,mat_B_ref = calc_sensitivity_matrices(lon,lat,np.ones(loni.shape),
                np.ones(loni.shape)*self.bots[k],magGrid,station_height=station_height,
                temp_stat=stat_file,temp_tess=tess_file,lons=lons,lats=lats)
            self.store[0,k,:,:] = mat_A
            self.store[1,k,:,:] = mat_B_ref
            if verbose:
                timer()
    
    def get_sens_matrix(self,chi_0,bot_0):
        """Return sensitivity matrices assuming the same thickness and susceptibility for all tesseroids
        
        Effectively, we are linearizing around `chi_0` and `bot_0`.

        Parameters
        -----------
        chi_0 : float
            Susceptibility assumed for all tesseroids
        bot_0 : float
            Thickness of the tesseroids for linearization. Should be larger
            than min(bots) and smaller than max(bots).
        
        Returns
        -------
        mat_A : np.ndarray(ndim==2)
            `mat_A[i,j]` contains the effect of tesseroid j on station i, 
            assuming a susceptibility of 1.
        mat_B : np.ndarray(ndim==2)
            ´mat_B[i,j]` gives how much the effect of teseseroid j on station i
            would change, if the thickness of tesseroid j was increased by 1 km.
        """
        if bot_0 > self.bots.max() or bot_0 < self.bots.min():
            if self.bounds_error:
                raise ValueError('Sensitivity Matrix Store: Out of bottom range')
            else:
                shape = self.store[0,0,:,:].shape
                return np.ones(shape)*np.nan,np.ones(shape)*np.nan
        
        if bot_0 == self.bots.min():
            mat_A = self.store[0,0,:,:]
            mat_B = self.store[1,0,:,:]
        elif bot_0 == self.bots.max():
            mat_A = self.store[0,-1,:,:]
            mat_B = self.store[1,-1,:,:]
        else:
            # np.argmax gives first true occurence
            # Note that self.bots is always sorted
            next_larger_ix = np.argmax(bot_0<self.bots)
            db_up = self.bots[next_larger_ix] - bot_0
            db_down = bot_0 - self.bots[next_larger_ix-1]
            mat_A = (self.store[0,next_larger_ix] * db_down + self.store[0,next_larger_ix-1] * db_up)/(db_up+db_down)
            mat_B = (self.store[1,next_larger_ix] * db_down + self.store[1,next_larger_ix-1] * db_up)/(db_up+db_down)
        mat_B = mat_B * chi_0
        return mat_A,mat_B
    
    def get_sens_matrix_grid(self,chi_grid,bot_grid):
        """Calculate sensitivity matrices for variable susc. and thickness

        Instead of linearizing around a constant value, each tesseroid
        can have its own susceptibility and thickness

        Note
        ----
        You can use this approach also to calculate the effect of an arbitrary
        model by using `mat_A.dot(susceptibility_grid.flatten())` on the returns of this
        function. 
        This is more accurate than linearization, but it still relies on the fundamental
        approximation inherent in the approach.

        Parameters
        -----------
        chi_grid : np.ndarray
            Suscpetibility grid to linearize around. Shape must agree with lon and lat
        bot_grid : float
            Thickness grid to linearize around.
        """
        min_lim = bot_grid.flatten() == self.bots.min()
        max_lim = bot_grid.flatten() == self.bots.max()
        out_lim = (bot_grid.flatten() < self.bots.min()) | (bot_grid.flatten() > self.bots.max())
        
        in_lim = (~min_lim) & (~max_lim) & (~out_lim)
        mat_A = np.zeros(self.store[0,0,:,:].shape)
        mat_B = np.zeros(self.store[0,0,:,:].shape)
        
        if min_lim.sum()>0:
            # I have no clue why you have to transpose here..
            # Theoretically they should already have the same shape
            mat_A[:,min_lim] = self.store[0,0,:,min_lim].T
            mat_B[:,min_lim] = self.store[1,0,:,min_lim].T
        if max_lim.sum()>0:
            mat_A[:,max_lim] = self.store[0,-1,:,max_lim].T
            mat_B[:,max_lim] = self.store[1,-1,:,max_lim].T
        if out_lim.sum()>0:
            mat_A[:,out_lim] = np.nan
            mat_B[:,out_lim] = np.nan

        # np.argmax gives first true occurence
        next_larger_ix = np.argmax(bot_grid.flatten()[...,None]<self.bots,1)
        
        db_up = self.bots[next_larger_ix] - bot_grid.flatten()
        db_down = bot_grid.flatten() - self.bots[next_larger_ix-1]
        ox = np.arange(self.store.shape[3],dtype=int)
        temp = (self.store[0,next_larger_ix,:,ox].T * db_down + self.store[0,next_larger_ix-1,:,ox].T * db_up)/(db_up+db_down)
        mat_A[:,in_lim] = temp[:,in_lim]
        temp = (self.store[1,next_larger_ix,:,ox].T * db_down + self.store[1,next_larger_ix-1,:,ox].T * db_up)/(db_up+db_down)
        mat_B[:,in_lim] = temp[:,in_lim]

        mat_B = mat_B * chi_grid.flatten()[None,:]
        return mat_A,mat_B


class BayesProbabilityCalculator:
    """Organizes the evaluation of conditional probabilities of the BACU approach

    The implied grid of tesseroids and the distribution of stations are fixed 
    by the sens_matrix_store. Then, the probability_calculator can evaluate the
    probability P(b|theta), where b are magnetic data and theta are a set of
    hyperparameters describing the spatial properties of susceptibility and
    magnetic thickness (e.g. Curie depth/Depth to Bottom). They are needed
    for MCMC.

    Parameters
    ----------
    pd: np.ndarray(ndim==2)
        Pairwise distance between all tesseroid centers. This should be 
        calculated on the flattened array
    sens_matrix_store: SensitivityMatrixStore
        See above
    matern_approximator : MaternApproximator, optional
        A matern_approximator relies on interpolation to speed up the
        calculation of the Matern covariance function, which is quite
        slow due to the special functions involved. If it is not given
        the exact formula will be used.
    err_covar_matrix : np.ndarray, optional
        (An estimate of) the correlation matrix of the magnetic data error.
        The **scale** of the error can still be varied through err_b. If it
        is not present, uncorrelated errors are assumed.
    subtract_reference : bool
        Whether `b_0` (the response of the linearization model) should
        be subtracted from the data. 
    
    """
    
    def __init__(self,pd,sens_matrix_store,matern_approximator=None,err_covar_matrix=None,subtract_reference=True):
        self.pd = pd
        self.sens_matrix_store = sens_matrix_store
        self.matern_approximator = matern_approximator
        if not sens_matrix_store is None:
            self.sensitivity_calculator = functools.lru_cache(maxsize=2)(sens_matrix_store.get_sens_matrix)
        self.err_covar_matrix = err_covar_matrix
        self.subtract_reference = subtract_reference

    @functools.lru_cache(maxsize=2)
    def get_sigma_chi_mat(self,sigma,nu,rho):
        return covar_func_to_matrix(self.pd,(sigma,nu,rho),self.matern_approximator)
    
    @functools.lru_cache(maxsize=2)
    def get_sigma_bot_mat(self,sigma,nu,rho):
        return covar_func_to_matrix(self.pd,(sigma,nu,rho),self.matern_approximator)
    
    @functools.lru_cache(maxsize=2)
    def get_sigma_B_mat(self,pars,err_b):
        mat_A,mat_B = self.sensitivity_calculator(pars[0],pars[1])
        sigma_chi_mat = self.get_sigma_chi_mat(pars[6],pars[2],pars[4])
        sigma_bot_mat = self.get_sigma_bot_mat(pars[7],pars[3],pars[5])
        sigma_B_mat = mat_A.dot(sigma_chi_mat.dot(mat_A.T)) + mat_B.dot(sigma_bot_mat.dot(mat_B.T))
        if self.err_covar_matrix is None:
            sigma_B_mat[np.diag_indices_from(sigma_B_mat)] += err_b
        else:
            sigma_B_mat[np.diag_indices_from(sigma_B_mat)] += err_b * self.err_covar_matrix
        slogdet = np.linalg.slogdet(sigma_B_mat)[1]
        return sigma_B_mat,slogdet

    def P_b_given_theta(self,b,pars,err_b=0.0):       
        mat_A,mat_B = self.sensitivity_calculator(pars[0],pars[1])
        B_0 = (mat_A * pars[0]).sum(1)
        sigma_B_mat,slogdet = self.get_sigma_B_mat(pars,err_b)
        if self.subtract_reference:
            dB = (b - B_0).flatten()
        else:
            dB = b.flatten()
        try:
            misfit = dB.T.dot(np.linalg.solve(sigma_B_mat,dB))
        except np.linalg.LinAlgError:
            misfit = np.nan
        return slogdet,misfit
    
    def P_b_given_chi_bot_theta(self,b,chi,bot,pars,err_b=0.0):
        mat_A,mat_B = self.sensitivity_calculator(pars[0],pars[1])
        B_0 = (mat_A * pars[0]).sum(1)
        sigma_B_mat,slogdet = self.get_sigma_B_mat(pars,err_b)
        if self.subtract_reference:
            dB = (b - B_0 - mat_A.dot(chi.flatten()-pars[0]) - mat_B.dot(bot.flatten()-pars[1])).flatten()
        else:
            dB = (b - mat_A.dot(chi.flatten()-pars[0]) - mat_B.dot(bot.flatten()-pars[1])).flatten()
        try:
            misfit = dB.T.dot(np.linalg.solve(sigma_B_mat,dB))
        except np.linalg.LinAlgError:
            misfit = np.nan
        
        return slogdet,misfit
    
    def P_chi_bot_given_theta(self,chi,bot,pars):
        """
        Note
        ----
        Susceptibility and thickness are uncorrelated. Thus, evaluating the likelihood can basically
        be done independently for susceptibiltiy and thickness and then combined.
        """
        sigma_chi_mat = self.get_sigma_chi_mat(pars[6],pars[2],pars[4])
        sigma_bot_mat = self.get_sigma_bot_mat(pars[7],pars[3],pars[5])
        slogdet = np.linalg.slogdet(sigma_chi_mat)[1] + np.linalg.slogdet(sigma_bot_mat)[1]
        dchi = chi.flatten() - pars[0]
        dbot = bot.flatten() - pars[1]
        try:
            misfit =  dchi.T.dot(np.linalg.solve(sigma_chi_mat,dchi))
            misfit += dbot.T.dot(np.linalg.solve(sigma_bot_mat,dbot))
        except np.linalg.LinAlgError:
            misfit = np.nan
        
        return slogdet,misfit
    
    def P_b_chi_bot_given_theta(self,b,chi,bot,pars,err_b=0.0):
        slogdet_1,misfit_1 = self.P_b_given_chi_bot_theta(b,chi,bot,pars,err_b)
        slogdet_2,misfit_2 = self.P_chi_bot_given_theta(chi,bot,pars)
        return slogdet_1+slogdet_2,misfit_1+misfit_2

def covar_func_to_matrix(pd,pars,matern_approximator=None):
    if not matern_approximator is None:
        matern_func = matern_approximator.interp1d
    else:
        matern_func = func_dump.C_matern
    sigma,nu,rho = pars
    sigma_mat = matern_func(pd,(sigma,nu,rho))
    return sigma_mat
        
def gridsearch_likelihood(pd,sens_matrix_store,linspaces,syn_B,matern_approximator=None,verbose=False):
    n_steps = len(linspaces[0])
    if syn_B.ndim == 2:
        repetitions = syn_B.shape[0]
    else:
        repetitions = 1
    ref_pars = np.array([l[n_steps//2] for l in linspaces])
    misfit = np.zeros((8,8,n_steps,n_steps,repetitions))
    slogdet_B = np.zeros((8,8,n_steps,n_steps))
    timer = utils.ETA(8*8*n_steps**2)

    probability_calculator = BayesProbabilityCalculator(pd,sens_matrix_store,matern_approximator)

    for i in range(len(linspaces)):
        for j in range(len(linspaces)):
            for k1 in range(n_steps):
                for k2 in range(n_steps):
                    pars = ref_pars.copy()
                    pars[i] = linspaces[i][k1]
                    pars[j] = linspaces[j][k2]
                    for l in range(repetitions):
                        slogdet_B[i,j,k1,k2],misfit[i,j,k1,k2,l] =  probability_calculator.P_b_given_theta(syn_B[l],tuple(pars))
                    if verbose:
                        timer()
    return misfit,slogdet_B


def gibbs_solve(A,B,chi_0,bot_0,mu_chi,mu_bot,sigma_chi_mat,sigma_bot_mat,sigma_Y_mat,b,b_0):
    """Susceptibility and thickness inversion with specified hyperparameters

    Parameters
    ----------
    A : np.ndarray
        Jacobian w.r.t. susceptibility
    B : np.ndarray
        Jacobian w.r.t. thickness
    chi_0 : float or np.ndarray
        Reference susceptibility
    bot_0 : float or np.ndarray
        Reference thickness
    mu_chi : float or np.ndarray
        Mean susceptibility
    mu_bot : float or np.ndarray
        Mean thickness
    sigma_chi_mat : np.ndarray
        Prior covariance matrix of the susceptibility
    sigma_bot_mat : np.ndarray
        Prior covariance matrix of the thikness
    sigma_Y_mat : np.ndarray
        Covariance matrix of the measurements errors of the magnetic field
    b : np.ndarray
        Measured magnetic field
    b_0 : np.ndarray
        Reference magnetic field (the response of the (chi_0,bot_0)-model)
    
    Returns
    -------
    new_mu : np.ndarray
        The inverted susceptibility and thickness in a vector. The first half contains susceptibility,
        the second half thickness. Statistically it is the mean of the distribution of thickness and
        susceptibility _given_ the measured data and the hyperparameters, which affect the mean
        values and prior covariance matrices.
    sigma_gibbs : np.ndarray
        The posterior covariance matrix for susceptibility and thickness.
    """
    n = A.shape[1]
    
    schuri = np.linalg.inv(A.dot(sigma_chi_mat).dot(A.T) + B.dot(sigma_bot_mat).dot(B.T) + sigma_Y_mat)
    
    sigma_gibbs = np.block([[-sigma_chi_mat.dot(A.T).dot(schuri.dot(A.dot(sigma_chi_mat))) + sigma_chi_mat,
                            -sigma_chi_mat.dot(A.T.dot(schuri.dot(B.dot(sigma_bot_mat))))],
                             [-sigma_bot_mat.dot(B.T.dot(schuri.dot(A.dot(sigma_chi_mat)))),
                             -sigma_bot_mat.dot(B.T.dot(schuri.dot(B.dot(sigma_bot_mat))))+sigma_bot_mat]])
        
    b_reduced = b - b_0 - A.dot(mu_chi-chi_0.flatten()) - B.dot(mu_bot-bot_0.flatten())
    
    new_mu = np.zeros(2*n)
    new_mu[:n] = mu_chi + sigma_chi_mat.dot(A.T).dot(schuri.dot(b_reduced))
    new_mu[n:] = mu_bot + sigma_bot_mat.dot(B.T).dot(schuri.dot(b_reduced))
    return new_mu,sigma_gibbs