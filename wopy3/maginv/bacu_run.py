"""Helper routines to run BAyesian CUrie (BACU) depth estimation using ini files 

Examples
-------
Option 1: First, create an ini file (see `example.ini`) and call `run_with_data_hyperparameters('your_ini.ini')`.
Once this is complete (and it might well take a while), create a piggyback ini file (see `pb_example.ini`)
to run the second step of the inversion. The results will be stored in the files you have specified.

Option 2: Alternatively, you can prepare your data and settings in your own python script and then call
`hyperparameter_inversion` and `piggyback_core` manually.

Option 3: If you want to change some fundamental things and really get into the meat of things, 
you could prepare the likelihood evaluations yourself and call `MCMC_core` followed by `bacu.solve_gibbs` instead.
"""

import wopy3.maginv.SH_filter as SH_filter
import wopy3.maginv.mag_inv as mag_inv
import wopy3.func_dump as func_dump
import wopy3.utils as utils
import wopy3.maginv.bacu as bacu
import wopy3.MCMC.distributions as distributions
import wopy3.MCMC.MCMC as MCMC

import configparser
import os
import numpy as np
import yaml
import scipy.interpolate as interpolate

names = ['chi_0','bot_0','nu_chi','nu_bot','rho_chi','rho_bot','sigma_chi','sigma_bot','err_b','err_z']
names_old = ['chi_0','bot_0','nu_chi','nu_bot','rho_chi','rho_bot','sigma_chi','sigma_bot']
name_to_ix = dict()
for i,n in enumerate(names):
    name_to_ix[n] = i
"""Variable names and mapping from variable name to an index.
"""

default_options = {
    'matern_approximator_N' : 0,
    'rel_proposal_range' : 0.1,
    'use_PT':True,
    'PT_cold_fraction':0.25,
    'PT_max_temp':1000.0,
    'use_timer':True,
    'random_seed':123456789,
    'subtract_reference':True
}
"""Dict of default options for kriging_inversion and hyperparameter_inversion
"""

def make_temp_files():
    """Create temporary files with random names in subfolder /tmp
    """
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    code = utils.get_random_string(8)
    stat_file = os.path.join('tmp',code +'_station.txt')
    tess_file = os.path.join('tmp',code +'.tess')
    return code,stat_file,tess_file

def remove_temp_files(code):
    """Delete temporary files according to random code
    """
    stat_file = os.path.join('tmp',code +'_station.txt')
    tess_file = os.path.join('tmp',code +'.tess')
    try:
        os.remove(stat_file)
        os.remove(tess_file)
    except OSError:
        print('Could not remove all temporary files')

def dump_synth(lon,lat,ref_pars,lmax,syn_chi,syn_bot,syn_B,magGrid,fname):
    np.save(fname,(lon,lat,ref_pars,lmax,syn_chi,syn_bot,syn_B,magGrid))

def read_synth(fname):
    lon,lat,ref_pars,lmax,syn_chi,syn_bot,syn_B,magGrid = np.load(fname,allow_pickle=True)
    return lon,lat,ref_pars,lmax,syn_chi,syn_bot,syn_B,magGrid

def generate_and_dump(ini_file):
    """Generate random synthetic susceptibility, thickness and calculate magnetic field

    The magnetic field will be calculated with tesseroids.
    """
    config = configparser.ConfigParser()
    config.read(ini_file) 

    N = config['PARAMETERS'].getint('N')
    deg_size = config['PARAMETERS'].getfloat('deg_size')
    mag_Z = config['PARAMETERS'].getfloat('mag_Z')
    station_height = config['PARAMETERS'].getfloat('station_height')
    synth_file = config['PARAMETERS'].get('synth_file')

    repetitions = config['SYNTH'].getint('repetitions')
    synth_pars = utils.get_comma_separated_float(config,'SYNTH','synth_pars')
    lmax = config['SYNTH'].getint('lmax')

    lon = np.linspace(-deg_size,deg_size,N)
    lat = np.linspace(-deg_size,deg_size,N)
    loni,lati = np.meshgrid(lon,lat)
    dx = (lon.max()-lon.min())/lon.size * 110.0
    dx_spec = 40000.0/lmax

    code,stat_file,tess_file = make_temp_files()
    print('Resolution %.2f km, Spherical resolution %.2f km'%(dx,dx_spec))
    
    stations = np.vstack((loni.flat,lati.flat,np.ones(loni.size)*station_height)).T
    np.savetxt(stat_file,stations,fmt='%.2f')
    
    magGrid = np.zeros((3,N,N))
    magGrid[2,:,:] = mag_Z

    syn_bot = np.zeros((repetitions,N,N))
    syn_chi = np.zeros((repetitions,N,N))
    syn_B = np.zeros((repetitions,N,N))
    A_design,B_design = SH_filter.get_SHS_design_matrix_grid((lon,lat),lmax)

    for k2 in range(repetitions):
        syn_chi[k2],syn_bot[k2] = bacu.synthetic_generation(lon,lat,(A_design,B_design),synth_pars[0],synth_pars[1],
                                                                            (synth_pars[6],synth_pars[2],synth_pars[4]),
                                                                            (synth_pars[7],synth_pars[3],synth_pars[5]))
        syn_B[k2] = bacu.forward_calc(lon,lat,syn_chi[k2],syn_bot[k2],magGrid,stat_file,tess_fname=tess_file)
    
    dump_synth(lon,lat,synth_pars,lmax,syn_chi,syn_bot,syn_B,magGrid,synth_file)
    remove_temp_files(code)

def make_magnetic_misfit(syn_B,probability_calculator):
    """Creates a function that evaluates likelihood of hyperparameters given magnetic data.

    Parameters
    ----------
    syn_B : np.ndarray
        The magnetic field measurements
    probability_calculator : bacu.ProbabilityCalculator
        The locations used to construct the probability_calculator must be the same
        as the magnetic data
    
    Returns
    -------
    internal_func : func
        internal_func can be called with a vector of 9 hyperparameters (theta)
        and returns the logarithmic likelihood P(theta|b)
    """

    def internal_func(theta):
        chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot,err_b = theta
        rho_chi = rho_chi / 110.0
        rho_bot = rho_bot / 110.0
        slogdet_B,misfit = probability_calculator.P_b_given_theta(syn_B.flatten(),
                    (chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot),
                    err_b)
        if misfit<0:
            misfit = np.nan
        return -(misfit + slogdet_B).sum()
    return internal_func

def make_joint_misfit(syn_B,selected_ix,selected_vals,probability_calculator):
    """Creates a function that evaluates likelihood given magnetic data and constraints.

    Parameters
    ----------
    syn_B : np.ndarray
        The magnetic field measurements
    selected_ix : np.ndarray (int)
        Gives the indices of the locations, where pointwise constraints are provided
        in the *flattened* array of thickness values
    selected_vals : np.ndarray
        Gives the pointwise constraints at the locations defined by selected_ix, so
        len(selected_ix) == len(selected_vals)
    probability_calculator : bacu.ProbabilityCalculator
        The locations used to construct the probability_calculator must be the same
        as the magnetic data
    
    Returns
    -------
    internal_func : func
        internal_func can be called with a vector of 10 hyperparameters and
        returns the log likelihood.
    """
    def internal_func(theta):
        data = syn_B.flatten()
        chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot,err_b,err_z = theta
        rho_chi = rho_chi / 110.0
        rho_bot = rho_bot / 110.0
        mat_A,mat_B = probability_calculator.sens_matrix_store.get_sens_matrix(chi_0,bot_0)
        B_0 = (mat_A * chi_0).sum(1)

        # Use noise-free version of sigma_B_mat (errors are uncorrelated between b and z)
        # sigma_B_mat,_ = probability_calculator.get_sigma_B_mat((chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot),0.0)
        # This is the noisy version of sigma_B_mat
        sigma_B_mat_err,_ = probability_calculator.get_sigma_B_mat((chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot),err_b)

        sigma_bot_mat = probability_calculator.get_sigma_bot_mat(sigma_bot,nu_bot,rho_bot)
        sigma_z_mat = (sigma_bot_mat[np.ix_(selected_ix,selected_ix)]) + np.eye(len(selected_ix)) * err_z
        cross = (mat_B.dot(sigma_bot_mat[:,selected_ix]))
        da_big_one = np.block([[sigma_B_mat_err,cross],[cross.T,sigma_z_mat]])
        slogdet = np.linalg.slogdet(da_big_one)[1]
        
        if probability_calculator.subtract_reference:
            dB = (data - B_0)
        else:
            dB = 1.0*data
        dz = (selected_vals - bot_0)
        dx = np.hstack((dB,dz))
        try:
            misfit = dx.T.dot(np.linalg.solve(da_big_one,dx))
            # da_big_one is not positive definite
            if misfit<0:
                misfit = np.nan
        except np.linalg.LinAlgError:
            misfit = np.nan
        
        return -(misfit + slogdet).sum()
    return internal_func

def make_probability_calculator(lon,lat,lons,lats,station_height,mag_Z,depth_steps,**kwargs):
    """Convenience function to make a ProbabilityCalculator

    Parameters
    ----------
    lon : np.ndarray(ndim==1)
        Vector of longitude values defining the tesseroid grid
    lat : np.ndarray(ndim==1)
        Vector of latitude values
    lons : np.ndarray(ndim==1)
        Longitude values of magnetic stations. These are not automatically
        extended to a grid! They **can** be irregular.
    lats : np.ndarray(ndim==1) 
        Latitude values of magnetic stations. These are not automatically
        extended to a grid! They **can** be irregular.
    station_height : float
        Constant height of the measurements in meters
    mag_Z : flat
        Strength of the magnetizing field in nT (assumed vertical)
    depth_steps : np.ndarray(ndim==1)
        The depth_steps to be used by the SensitivityMatrixStore that is created
    **kwargs
        matern_approximator_N (int): 
            Number of terms to use for the MaternApproximator. If 0 no
            approximation will be applied.
        subtract_reference (bool):
            Whether b_0 should be subtracted from the data. Put false, if
            linearzing around the reference model.
    
    Returns
    -------
    probability_calculator : bacu.ProbabilityCalculator
    """
    matern_approximator_N = kwargs.get("matern_approximator_N")
    subtract_reference = kwargs.get("subtract_reference")
    #assert matern_approximator_N > 0
    loni,lati = np.meshgrid(lon,lat)
    nx,ny = len(lon),len(lat)
    pd = func_dump.get_pairwise_geo_distance(loni.flatten(),lati.flatten())
    magGrid = np.zeros((3,ny,nx))
    magGrid[2,:,:] = mag_Z
    code,stat_file,tess_file = make_temp_files()
    stations = np.vstack((lons,lats,np.ones(lons.size)*station_height)).T
    np.savetxt(stat_file,stations,fmt='%.2f')
    sens_matrix_store = bacu.SensitivityMatrixStore(lon,lat,magGrid,depth_steps,
        verbose=True,bounds_error=False,stat_file=stat_file,tess_file=tess_file,
        lons=lons,lats=lats,station_height=station_height)
    remove_temp_files(code)

    if matern_approximator_N == 0:
        probability_calculator = bacu.BayesProbabilityCalculator(pd,sens_matrix_store,matern_approximator=None,
                                                                subtract_reference=subtract_reference)
    else:
        probability_calculator = bacu.BayesProbabilityCalculator(pd,sens_matrix_store,
                                bacu.MaternApproximator(matern_approximator_N,'root'),
                                subtract_reference=subtract_reference)
    
    return probability_calculator

def make_pars_filler(fixed_mask,fixed_vals):
    """Helper function for fixing some hyperparameters
    
    This wraps the misfit function, such that only some of the parameters are actually
    varied, but the same misfit function can still be used. n_target is either 9 or 10,
    depending on whether point data are used or not.

    Parameters
    ----------
    fixed_mask : np.ndarray(dtype=bool)
        A boolean array describing whether the n-th hyperparameter should
        be fixed.
    fixed_vals : np.ndarray
        The values for the fixed parameters. 
        Thus, len(fixed_vals) == fixed_mask.sum()
    
    Returns
    -------
    internal_func : func
        This function takes a vector consisting of the *variable* hyperparameters
        that is returned by the MCMC proposal and fills in the fixed hyperparameters.
    """
    n_target = len(fixed_mask)
    def internal_func(theta):
       returnor = np.zeros((n_target))
       returnor[fixed_mask] = fixed_vals
       returnor[~fixed_mask] = theta
       return returnor 
    return internal_func

def hyperparameter_inversion(lon,lat,lons,lats,station_height,mag_data,mag_Z,ranges,n_runs,restarts,
                                n_steps,selected_ix=None,selected_vals=None,fixed_ix=[],fixed_par_vals=[],
                                **kwargs):
    """Run step 1 (hyperparameter inversion) of the BACU approach on specified data

    Parameters
    ----------
    lon : np.ndarray(ndim==1)
        Vector of longitude values defining the tesseroid grid
    lat : np.ndarray(ndim==1)
        Vector of latitude values
    lons : np.ndarray(ndim==1)
        Longitude values of magnetic stations. These are not automatically
        extended to a grid! They **can** be irregular.
    lats : np.ndarray(ndim==1) 
        Latitude values of magnetic stations. These are not automatically
        extended to a grid! They **can** be irregular.
    station_height : float
        Constant height of the measurements in meters
    mag_data : np.ndarray(ndim==1)
        Magnetic field measurement at the station locations (Z-component positive UP)
    mag_Z : float
        Strength of the magnetizing field in nT (assumed vertical)
    ranges : list of list or np.ndarray
        List of minimum and maximum values for each of the hyperparameters. The range
        must also be given for parameters, which are fixed.
    n_runs : int
        How many steps of MCMC should be run?
    restarts : int
        How many Parallel Tempering chains should be run at the same time?
    n_steps : int
        Number of depth steps used to create the bacu.SensitivityMatrixStore. Higher
        numbers lead to greater computation time but also higher accuracy. The number
        of n_steps depends also strongly on the station_height
    selected_ix : np.ndarray (int)
        Gives the indices of the locations, where pointwise constraints are provided
        in the *flattened* array of thickness values
    selected_vals : np.ndarray
        Gives the pointwise constraints at the locations defined by selected_ix, so
        len(selected_ix) == len(selected_vals)
    fixed_ix : np.ndarray(dtype==int)
        Index of the **fixed** hyperparameters
    fixed_par_vals : np.ndarray
        Values for the **fixed** hyperparameters
    **kwargs
        matern_approximator_N (int): 
            Number of terms to use for the MaternApproximator. If 0 no
            approximation will be applied.
        subtract_reference (bool):
            Whether b_0 should be subtracted from the data. Put false, if
            linearzing around the reference model.
    
    Returns
    -------

    xchain : np.ndarray
        Hyperparameter chains. The shape is `(restarts,n_runs,number_of_variable_parameters)`
    Lchain : np.ndarray
        Likelihood values of each element in the chains
    accepted : np.ndarray(int)
        Number of accepted moves for each chain with different temperatures (if PT is used)
    theta0 : np.ndarray
        Start values used by each of the chains
    """
    ranges = np.asarray(ranges)
    options = default_options
    options.update(kwargs)

    rel_proposal_range = options.get("rel_proposal_range")
    use_PT = options.get("use_PT")
    PT_cold_fraction = options.get("PT_cold_fraction")
    PT_max_temp = options.get('PT_max_temp')
    use_timer = options.get("use_timer")
    random_seed = options.get("random_seed")

    np.random.seed(random_seed)

    # Account for fixed parameter values
    if selected_ix is None:
        n_pars = 9 - len(fixed_ix)
        fixed_mask = np.zeros(9,dtype=bool)   
    else:
        n_pars = 10 - len(fixed_ix)
        fixed_mask = np.zeros(10,dtype=bool)   
    
    fixed_mask[fixed_ix] = True
    active_ix = np.where(~fixed_mask)[0]

    probability_calculator = make_probability_calculator(lon,lat,lons,lats,station_height,mag_Z,np.linspace(ranges[1][0],ranges[1][1],n_steps),**options)
    
    prior = distributions.RangePrior([r[0] for r in ranges[active_ix]],[r[1] for r in ranges[active_ix]])
    proposal = distributions.ComponentUpdateProposal(rel_proposal_range * np.array([r[1]-r[0] for r in ranges[active_ix]]))

    if use_PT:
        cold_chains = np.arange(restarts) < PT_cold_fraction * restarts    
        PT_temperatures = np.ones(restarts)
        PT_temperatures[~cold_chains] = np.power(10,np.random.random(size=(~cold_chains).sum())*np.log10(PT_max_temp))
    else:
        PT_temperatures = None
    
    if selected_ix is None:
        misfit_func = make_magnetic_misfit(mag_data,probability_calculator)
    else:
        misfit_func = make_joint_misfit(mag_data,selected_ix,selected_vals,probability_calculator)

    wrapper = make_pars_filler(fixed_mask,fixed_par_vals)
    wrapped_misfit_func = lambda theta: misfit_func(wrapper(theta))

    print('Initiating MCMC')
    if selected_ix is None:
        print('Running without point constraints')
    else:
        print('Running with point constraints')
    print('Summary of settings:')
    print(options)
    print('Active Parameters and ranges')
    for i in active_ix:
        print(names[i],ranges[i][0],ranges[i][1])
    print('Fixed Parameters and values')
    for j,i in enumerate(fixed_ix):
        print(names[i],fixed_par_vals[j])
    
    return MCMC_core(prior,wrapped_misfit_func,proposal,n_runs,restarts,PT_temperatures,use_timer)
    
def MCMC_core(prior,wrapped_misfit_func,proposal,n_runs,restarts,PT_temperatures=None,use_timer=True):
    """Run MCMC (or Parallel Tempering) with specified likelihood objects
    """
    n_pars = len(prior.lb)

    if PT_temperatures is None:
        use_PT = False
        timer = utils.ETA(n_runs*restarts,wait_time=5)
    else:
        use_PT = True
        timer = utils.ETA(n_runs,wait_time=5)
    callback = lambda i:timer() if use_timer else lambda i : None
    xchain = np.zeros((restarts,n_runs,n_pars))
    Lchain = np.zeros((restarts,n_runs))
    accepted = np.zeros((restarts))
    
    if use_PT:
        theta0 = prior.lb[:,None] + np.random.random(size=(n_pars,restarts)) * (prior.ub-prior.lb)[:,None]
        MCMC_return = MCMC.parallel_tempering(theta0,lambda theta:theta,wrapped_misfit_func,
                                prior.logpdf,proposal,PT_temperatures,n_runs,callback=callback)     
        xchain = MCMC_return[0]
        Lchain = MCMC_return[1]
        accepted = MCMC_return[2]
    else:
        for j in range(restarts):
            print('Restart %d/%d'%(j+1,restarts))
            theta0 = prior.lb + np.random.random(n_pars) * (prior.ub-prior.lb)        
            MCMC_return = MCMC.MCMC(theta0,lambda theta:theta,wrapped_misfit_func,prior.logpdf,proposal,n_runs,callback=callback)
            xchain[j,:,:] = MCMC_return[0]
            Lchain[j,:] = MCMC_return[1]
            accepted[j] = MCMC_return[2]
    return xchain,Lchain,accepted,theta0


def experiment1b(ini_file):
    """Load synthetic data and run hyperparameter inversion
    """
    config = configparser.ConfigParser()
    config.read(ini_file)

    N = config['PARAMETERS'].getint('N')
    deg_size = config['PARAMETERS'].getfloat('deg_size')
    mag_Z = config['PARAMETERS'].getfloat('mag_Z')
    station_height = config['PARAMETERS'].getfloat('station_height')

    out_file = config['PARAMETERS'].get('out_file')
    synth_file = config['PARAMETERS'].get('synth_file')
    n_constraints = config['PARAMETERS'].getint('n_constraints',0)
    if os.path.isfile(synth_file):
        # Read a previously created synth dump file and do some verification that
        # it is actually compatible with the settings
        lon,lat,_,_,syn_chi,syn_bot,syn_B,magGrid = read_synth(synth_file)
        repetitions = syn_B.shape[0]
        assert lon.size == N
        lons,lats = np.meshgrid(lon,lat)
        lons = lons.flatten()
        lats = lats.flatten()
    else:
        raise ValueError('Synthetic input file %s not found'%synth_file)

    n_runs = config['MCMC'].getint('n_runs')
    restarts = config['MCMC'].getint('restarts')
    n_steps = config['RANGES'].getint('n_steps')
    ranges = []
    for n in names:
        if config.has_option('RANGES',n):
            ranges.append(utils.get_comma_separated_float(config,"RANGES",n))
        else:
            ranges.append([0.0,1000.0])
    fixed_pars = config['MCMC'].get('fixed_pars').split(',')
    fixed_ix = []
    for p in fixed_pars:
        if p:
            fixed_ix.append(name_to_ix[p])
    fixed_vals = utils.get_comma_separated_float(config,"MCMC",'fixed_vals')
    print(fixed_ix)
    print(fixed_vals)
    if n_constraints == 0:
        n_pars = 9
    else:
        n_pars = 10
    n_pars = n_pars - len(fixed_vals)
    xchain = np.zeros((repetitions,restarts,n_runs,n_pars))
    Lchain = np.zeros((repetitions,restarts,n_runs))
    accepted = np.zeros((repetitions,restarts))
    for i in range(repetitions):
        selected_ix = np.random.choice(N**2,size=(n_constraints),replace=False)
        selected_vals = syn_bot[i].flatten()[selected_ix]
        if n_constraints == 0:
            selected_ix = None
            selected_vals  = None
        else:
            fname = '%s_constraints_%d.txt' % (out_file[:-4],i)
            np.savetxt(fname,selected_ix,fmt='%d')
        temp = hyperparameter_inversion(lon,lat,lons,lats,station_height,syn_B[i].flatten(),mag_Z,ranges,n_runs,
                                                                            restarts,n_steps,selected_ix,selected_vals,
                                                                            fixed_ix=fixed_ix,fixed_par_vals=fixed_vals,
                                                                            **config_to_kwargs(config))
        xchain[i] = temp[0]
        Lchain[i] = temp[1]
        accepted[i] = temp[2]
        theta0 = temp[3]
    np.savez(out_file,xchain,Lchain,accepted,theta0)
    return xchain,Lchain,accepted,theta0

def piggyback_core(mag_data,filled_theta_chain,probability_calculator,piggyback_burn_in,piggyback_step,selected_ix=None,sel_vals=None,
                    nonlinear_iterations=1,clip_to_bounds=False):

    """Run step 2 of the BaCu Inversion (susceptibility and DTB estimation)

    Parameters
    ----------
    mag_data : np.ndarray(ndim==1)
        Magnetic field measurement at the station locations (Z-component positive UP)
    filled_theta_chain : np.ndarray
        Chain of hyperparameter values generated by step 1 of the inversion. If some of the hyperparameters
        were fixed during the inversion, these fixed hyperparameters must have been filled in. See
        ´run_with_data_piggyback` for an example of how to do this.
    probability_calculator : bacu.ProbabilityCalculator
        The locations used to construct the probability_calculator must be the same
        as the magnetic data
    piggyback_burn_in : int
        How many of the hyperparameter chains should be discarded
    piggyback_step: int
        1 in piggyack_step of the elements from the chain is only used. So, the inversion is only
        carried out for `theta[piggyback_burn_in::piggyback_step]`
    selected_ix : np.ndarray (int), optional
        Gives the indices of the locations, where pointwise constraints are provided
        in the *flattened* array of thickness values
    selected_vals : np.ndarray, optional
        Gives the pointwise constraints at the locations defined by selected_ix, so
        len(selected_ix) == len(selected_vals)
    nonlinear_iterations : int, optional
        If larger than 1, then additional linearizations are done around the intermediate
        results.
    clip_to_bounds : bool, optional
        If true, the thickness (DTB) estimates will be clipped to the minimum and maximum
        depth step of the `probability_calculator.sensitivity_matrix_store`. Otherwise, 
        you can sometimes get only nans as result, because one of the tesseroids is getting
        too thick or thin. This is only relevant if nonlinear_iterations is larger than 1.
    """
    restarts = filled_theta_chain.shape[0]
    n_runs = filled_theta_chain.shape[1]
    sens_matrix_store = probability_calculator.sens_matrix_store
    if selected_ix is None:
        n_constraints = 0
    else:
        n_constraints = len(selected_ix)
    
    nx,ny = len(sens_matrix_store.lon),len(sens_matrix_store.lat)

    piggyback_runs = len(range(piggyback_burn_in,n_runs,piggyback_step))
    chi_chain = np.zeros((restarts,piggyback_runs,ny,nx))
    bot_chain = np.zeros((restarts,piggyback_runs,ny,nx))
    timer = utils.ETA(piggyback_runs*restarts,wait_time=5)
        
    for j in range(restarts):
        print('Restart %d/%d'%(j+1,restarts))
        if not selected_ix is None:
            b = mag_data.flatten()
            d = np.hstack((b,sel_vals))
            P = np.zeros((n_constraints,ny*nx))
            P[np.arange(n_constraints,dtype=int),selected_ix] = 1
        else:
            d = mag_data.flatten()

        for k,n_chain in enumerate(range(piggyback_burn_in,n_runs,piggyback_step)):
            chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot,err_b,err_z = filled_theta_chain[j,n_chain]
            # Update chi and bot
            sigma_chi_mat = probability_calculator.get_sigma_chi_mat(sigma_chi,nu_chi,rho_chi/110.0)
            sigma_bot_mat = probability_calculator.get_sigma_bot_mat(sigma_bot,nu_bot,rho_bot/110.0)

            mu_bot = np.ones(nx*ny) * bot_0
            mu_chi = np.ones(nx*ny) * chi_0
            last_chi_grid = mu_chi.copy()
            last_bot_grid = mu_bot.copy()

            for l in range(nonlinear_iterations):
                mat_A,mat_B = sens_matrix_store.get_sens_matrix_grid(last_chi_grid,last_bot_grid)
                if l==0:
                    edge_effect = mat_A.dot(mu_chi)

                if probability_calculator.subtract_reference:
                    d0 = mat_A.dot(last_chi_grid) 
                else:
                    d0 = mat_A.dot(last_chi_grid) - edge_effect

                if not selected_ix is None:
                    d0_aug = np.hstack((d0,P.dot(last_bot_grid)))
                    mat_A_aug = np.vstack((mat_A,np.zeros((n_constraints,nx*ny))))
                    mat_B_aug = np.vstack((mat_B,P))
                    sigma_d_mat = np.diag(np.concatenate((np.ones(mag_data.size)*err_b,np.ones(n_constraints)*err_z)))
                    new_mu,_ = bacu.gibbs_solve(mat_A_aug,mat_B_aug,last_chi_grid,last_bot_grid,mu_chi,mu_bot,
                        sigma_chi_mat,sigma_bot_mat,sigma_d_mat,d,d0_aug)
                else:
                    sigma_d_mat = np.eye(mag_data.size) * err_b
                    new_mu,_ = bacu.gibbs_solve(mat_A,mat_B,last_chi_grid,last_bot_grid,mu_chi,mu_bot,
                        sigma_chi_mat,sigma_bot_mat,sigma_d_mat,d,d0)
                
                last_chi_grid = new_mu[:nx*ny]
                last_bot_grid = new_mu[nx*ny:]
                if clip_to_bounds:
                    last_bot_grid = np.clip(last_bot_grid,sens_matrix_store.bots[0],sens_matrix_store.bots[-1])

            chi_chain[j,k] = last_chi_grid.reshape((ny,nx))
            bot_chain[j,k] = last_bot_grid.reshape((ny,nx))
            timer()
    return chi_chain,bot_chain

def experiment3a_piggyback(ini_file):
    # Use existing chain of theta values
    if isinstance(ini_file,str):
        piggy_config = configparser.ConfigParser()
        piggy_config.read(ini_file) 
    elif isinstance(ini_file,configparser.ConfigParser):
        piggy_config = ini_file

    master_file = piggy_config['PIGGYBACK'].get('master_ini')
    piggyback_burn_in = piggy_config['PIGGYBACK'].getint('burn_in')
    piggyback_step = piggy_config['PIGGYBACK'].getint('step')
    constraints_override = piggy_config['PIGGYBACK'].getint('constraints_override',-1)
    piggyback_out_file = piggy_config['PIGGYBACK'].get('out_file')
    
    config = configparser.ConfigParser()
    config.read(master_file) 

    N = config['PARAMETERS'].getint('N')
    mag_Z = config['PARAMETERS'].getfloat('mag_Z')
    station_height = config['PARAMETERS'].getfloat('station_height')
    matern_approximator_N = config['PARAMETERS'].getint('matern_approximator_N')

    synth_file = config['PARAMETERS'].get('synth_file')
    out_file = config['PARAMETERS'].get('out_file')

    restarts = config['MCMC'].getint('restarts')
    use_PT = config['MCMC'].getboolean('use_PT',False)
    if use_PT:
        PT_cold_fraction = config['MCMC'].getfloat('PT_cold_fraction')
        PT_max_temp = config['MCMC'].getfloat('PT_max_temp')
        cold_chains = np.arange(restarts) < PT_cold_fraction * restarts
        # PT_cold_fraction have T=1, the rest have log uniform temperature between 1 and `PT_max_temp`
        PT_temperatures = np.ones(restarts)
        PT_temperatures[~cold_chains] = np.power(10,np.random.random(size=(~cold_chains).sum())*np.log10(PT_max_temp))

    n_constraints_orig = config['PARAMETERS'].getint('n_constraints',0)
    n_constraints = n_constraints_orig
    # Optionally use a different number of constraints than the original chain
    if constraints_override > -1:
        n_constraints = constraints_override

    # Load existing MCMC results
    with np.load(out_file) as data:
        theta_chain = data['arr_0']
    if use_PT:
        print(theta_chain.shape,cold_chains.shape)
        theta_chain = theta_chain[:,cold_chains]
        restarts = cold_chains.sum()
    
    n_steps = config['RANGES'].getint('n_steps')
    ranges = []
    for n in names:
        if config.has_option('RANGES',n):
            ranges.append(utils.get_comma_separated_float(config,"RANGES",n))
        else:
            ranges.append([0.0,1000.0])
    fixed_pars = config['MCMC'].get('fixed_pars').split(',')
    fixed_ix = []
    for p in fixed_pars:
        if p:
            fixed_ix.append(name_to_ix[p])
    fixed_vals = utils.get_comma_separated_float(config,"MCMC",'fixed_vals')
    fixed_mask = np.zeros((10),dtype=bool)
    fixed_mask[fixed_ix] = True
    if n_constraints_orig == 0 and not fixed_mask[-1]:
        fixed_mask[-1] = True
        fixed_vals.append(0.0)
    
    if os.path.isfile(synth_file):
        # Read a previously created synth dump file and do some verification that
        # it is actually compatible with the settings
        lon,lat,_,_,syn_chi,syn_bot,syn_B,magGrid = read_synth(synth_file)
        repetitions = syn_B.shape[0]
        assert lon.size == N
        lons,lats = np.meshgrid(lon,lat)
        lons = lons.flatten()
        lats = lats.flatten()
    else:
        raise ValueError('Synthetic input file %s not found'%synth_file)
    
    options = config_to_kwargs(config)
    probability_calculator = make_probability_calculator(lon,lat,lons,lats,station_height,mag_Z,np.linspace(ranges[1][0],ranges[1][1],n_steps),**options)
    chi_chain = []
    bot_chain = []
    for i in range(repetitions):
        selected_ix = np.random.choice(N**2,size=(n_constraints),replace=False)
        sel_vals = syn_bot[i].flatten()[selected_ix]
        if n_constraints == 0:
            selected_ix = None
            sel_vals = None
        filled_theta_chain = np.zeros((theta_chain.shape[1],theta_chain.shape[2],10))
        print(filled_theta_chain.shape,theta_chain.shape)
        filled_theta_chain[:,:,fixed_mask] = fixed_vals
        filled_theta_chain[:,:,~fixed_mask] = theta_chain[i]
        print(selected_ix,sel_vals,n_constraints)
        temp = piggyback_core(syn_B[i],filled_theta_chain,probability_calculator,piggyback_burn_in,piggyback_step,selected_ix,sel_vals)
        chi_chain.append(temp[0])
        bot_chain.append(temp[1])
    chi_chain = np.array(chi_chain)
    bot_chain = np.array(bot_chain)
    np.savez(piggyback_out_file,theta_chain,chi_chain,bot_chain)
    return chi_chain,bot_chain

def make_kriging_misfit(syn_chi,syn_bot,probability_calculator,rep_sel=0):
    def internal_func(theta):
        chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot = theta
        rho_chi = rho_chi / 110.0
        rho_bot = rho_bot / 110.0
        
        sigma_bot_mat = probability_calculator.get_sigma_bot_mat(sigma_bot,nu_bot,rho_bot)
        sigma_chi_mat = probability_calculator.get_sigma_chi_mat(sigma_chi,nu_chi,rho_chi)

        slogdet_bot = np.linalg.slogdet(sigma_bot_mat)[1]
        slogdet_chi = np.linalg.slogdet(sigma_chi_mat)[1]
        
        dchi = (syn_chi[rep_sel].flatten() - chi_0)
        dz = (syn_bot[rep_sel].flatten() - bot_0)

        try:
            misfit = dchi.T.dot(np.linalg.solve(sigma_chi_mat,dchi))
            misfit += dz.T.dot(np.linalg.solve(sigma_bot_mat,dz))
            
        except np.linalg.LinAlgError:
            misfit = np.nan
        
        return -(misfit + slogdet_bot + slogdet_chi).sum()
    return internal_func

def config_to_kwargs(config):
    kwargs = dict()

    kwargs["matern_approximator_N"] = config['PARAMETERS'].getint('matern_approximator_N')
    kwargs["subtract_reference"] = config['PARAMETERS'].getboolean('subtract_reference')

    kwargs["rel_proposal_range"] = config['MCMC'].getfloat('rel_proposal_range')
    kwargs["use_PT"] = config['MCMC'].getboolean('use_PT',False)
    kwargs["PT_cold_fraction"] = config['MCMC'].getfloat('PT_cold_fraction')
    kwargs["PT_max_temp"] = config['MCMC'].getfloat('PT_max_temp')
    kwargs["use_timer"] = config['MCMC'].getboolean('use_timer',False)
    kwargs["random_seed"] = config['MCMC'].getint('random_seed')

    # Remove all keys which are None (assuming that they are missing)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    options = default_options
    options.update(kwargs)
    return options

def read_config_data(config):
    """Central place to read in data
    """
    # Define tesseroid grid
    lon_lims = utils.get_comma_separated_float(config,'PARAMETERS','lon_lims')
    lat_lims = utils.get_comma_separated_float(config,'PARAMETERS','lat_lims')
    nx,ny = utils.get_comma_separated_float(config,'PARAMETERS','block_number')
    nx,ny = int(nx),int(ny)

    lon = np.linspace(lon_lims[0],lon_lims[1],nx)
    lat = np.linspace(lat_lims[0],lat_lims[1],ny)

    # Define station locations
    mag_data_file = config['DATA'].get('mag_data')
    data_table = np.loadtxt(mag_data_file)
    lons,lats = data_table[:,0],data_table[:,1]
    mag_data = data_table[:,3]
    assert len(np.unique(data_table[:,2]))==1
    station_height = data_table[0,2]   
     
    # (Optional): Define point depth information
    point_file = config['DATA'].get('point_data',None)
    if not point_file is None:
        point_data = np.loadtxt(point_file)
        if point_data.ndim == 1:
            point_data = point_data[None,:]
        ix_mat = np.arange(nx*ny,dtype=int).reshape((ny,nx))
        selected_ix = interpolate.RegularGridInterpolator((lat,lon),ix_mat)((point_data[:,1],point_data[:,0]),"nearest")
        selected_ix = selected_ix.astype(int)
        selected_vals = point_data[:,2]
    else:
        selected_ix = None
        selected_vals = None
    
    return lon,lat,lons,lats,station_height,mag_data,selected_ix,selected_vals

def run_with_data_hyperparameters(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)

    lon,lat,lons,lats,station_height,mag_data,selected_ix,selected_vals = read_config_data(config)

    out_file = config['PARAMETERS'].get('out_file')
    n_runs = config['MCMC'].getint('n_runs')
    restarts = config['MCMC'].getint('restarts')
    
    n_steps = config['RANGES'].getint('n_steps')
    ranges = []
    for n in names:
        if config.has_option('RANGES',n):
            ranges.append(utils.get_comma_separated_float(config,"RANGES",n))
        else:
            ranges.append([0.0,1000.0])
    fixed_pars = config['MCMC'].get('fixed_pars').split(',')
    fixed_ix = []
    for p in fixed_pars:
        if p:
            fixed_ix.append(name_to_ix[p])
    fixed_vals = utils.get_comma_separated_float(config,"MCMC",'fixed_vals')
    mag_Z = config['PARAMETERS'].getfloat('mag_z')

    xchain,Lchain,accepted,theta0 = hyperparameter_inversion(lon,lat,lons,lats,station_height,mag_data,mag_Z,ranges,n_runs,
                                                                            restarts,n_steps,selected_ix,selected_vals,
                                                                            fixed_ix,fixed_vals,
                                                                            **config_to_kwargs(config))
    np.savez(out_file,xchain,Lchain,accepted,theta0)
    return xchain,Lchain,accepted,theta0

def run_with_data_piggyback(ini_file):
    # Use existing chain of theta values
    if isinstance(ini_file,str):
        piggy_config = configparser.ConfigParser()
        piggy_config.read(ini_file) 
    elif isinstance(ini_file,configparser.ConfigParser):
        piggy_config = ini_file

    master_file = piggy_config['PIGGYBACK'].get('master_ini')
    piggyback_burn_in = piggy_config['PIGGYBACK'].getint('burn_in')
    piggyback_step = piggy_config['PIGGYBACK'].getint('step')
    piggyback_out_file = piggy_config['PIGGYBACK'].get('out_file')
    nonlinear_iterations = piggy_config['PIGGYBACK'].getint('nonlinear_iterations')
    bot_0_override = piggy_config['PIGGYBACK'].get('bot_0_override',None)
    n_steps_override = piggy_config['PIGGYBACK'].getint('n_steps_override',-1)
    clip_to_bounds = piggy_config['PIGGYBACK'].getboolean('clip_to_bounds',False)
    
    config = configparser.ConfigParser()
    config.read(master_file) 

    mag_Z = config['PARAMETERS'].getfloat('mag_Z')
    out_file = config['PARAMETERS'].get('out_file')
    lon,lat,lons,lats,station_height,mag_data,selected_ix,selected_vals = read_config_data(config)

    restarts = config['MCMC'].getint('restarts')
    use_PT = config['MCMC'].getboolean('use_PT',False)
    if use_PT:
        PT_cold_fraction = config['MCMC'].getfloat('PT_cold_fraction')
        PT_max_temp = config['MCMC'].getfloat('PT_max_temp')
        cold_chains = np.arange(restarts) < PT_cold_fraction * restarts
        # PT_cold_fraction have T=1, the rest have log uniform temperature between 1 and `PT_max_temp`
        PT_temperatures = np.ones(restarts)
        PT_temperatures[~cold_chains] = np.power(10,np.random.random(size=(~cold_chains).sum())*np.log10(PT_max_temp))

    # Load existing MCMC results
    with np.load(out_file) as data:
        theta_chain = data['arr_0']
    if use_PT:
        theta_chain = theta_chain[cold_chains]
        restarts = cold_chains.sum()

    n_steps = config['RANGES'].getint('n_steps')
    if n_steps_override > 0:
        n_steps = n_steps_override

    ranges = []
    for n in names:
        if config.has_option('RANGES',n):
            ranges.append(utils.get_comma_separated_float(config,"RANGES",n))
        else:
            ranges.append([0.0,1000.0])

    if bot_0_override:
        ranges[1] = utils.get_comma_separated_float(piggy_config,"PIGGYBACK",'bot_0_override')
    
    fixed_pars = config['MCMC'].get('fixed_pars').split(',')
    fixed_ix = []
    for p in fixed_pars:
        if p:
            fixed_ix.append(name_to_ix[p])
    fixed_vals = utils.get_comma_separated_float(config,"MCMC",'fixed_vals')
    if selected_ix is None and not 'err_z' in fixed_pars:
        fixed_ix.append(9)
        fixed_vals.append(0.0)
    n_pars = 10 - len(fixed_ix)
    fixed_mask = np.zeros(10,dtype=bool)   
    fixed_mask[fixed_ix] = True
    options = config_to_kwargs(config)
   
    filled_theta_chain = np.zeros((theta_chain.shape[0],theta_chain.shape[1],10))
    print(fixed_ix,fixed_vals,fixed_mask)
    print(filled_theta_chain.shape,theta_chain.shape)
    filled_theta_chain[:,:,fixed_mask] = fixed_vals
    filled_theta_chain[:,:,~fixed_mask] = theta_chain

    probability_calculator = make_probability_calculator(lon,lat,lons,lats,station_height,mag_Z,np.linspace(ranges[1][0],ranges[1][1],n_steps),**options)
    chi_chain,bot_chain = piggyback_core(mag_data.flatten(),filled_theta_chain,probability_calculator,piggyback_burn_in,piggyback_step,
                                        selected_ix,selected_vals,nonlinear_iterations,clip_to_bounds)
    
    np.savez(piggyback_out_file,theta_chain,chi_chain,bot_chain)
    return chi_chain,bot_chain

def kriging_inversion(lon,lat,syn_bot,syn_chi,ranges,n_runs,restarts,**kwargs):
    # Run MCMC using syn_bot and syn_chi (no magnetic field)
    options = default_options
    options.update(kwargs)
    loni,lati = np.meshgrid(lon,lat)
    pd = func_dump.get_pairwise_geo_distance(loni.flatten(),lati.flatten())

    matern_approximator_N = options.get("matern_approximator_N")
    rel_proposal_range = options.get("rel_proposal_range")
    use_PT = options.get("use_PT",True)
    PT_cold_fraction = options.get("PT_cold_fraction")
    PT_max_temp = options.get('PT_max_temp')
    cold_chains = np.arange(restarts) < PT_cold_fraction * restarts
    PT_temperatures = np.ones(restarts)
    PT_temperatures[~cold_chains] = np.power(10,np.random.random(size=(~cold_chains).sum())*np.log10(PT_max_temp))
    use_timer = options.get("use_timer")
    random_seed = options.get("random_seed")

    np.random.seed(random_seed)

    # Prepare MCMC objects
    prior = distributions.RangePrior([r[0] for r in ranges],[r[1] for r in ranges])
    proposal = distributions.ComponentUpdateProposal(rel_proposal_range * np.array([r[1]-r[0] for r in ranges]))
    xchain = np.zeros((restarts,n_runs,8))
    Lchain = np.zeros((restarts,n_runs))
    accepted = np.zeros((restarts))
    timer = utils.ETA(n_runs*restarts,wait_time=5)
    
    matern_func = func_dump.C_matern
    matern_approximator = None
    if matern_approximator_N > 0:
        matern_approximator=bacu.MaternApproximator(matern_approximator_N,'root')
        matern_func = matern_approximator.interp1d

    probability_calculator = bacu.BayesProbabilityCalculator(pd,None,matern_approximator)
    if use_PT:
        timer = utils.ETA(n_runs,wait_time=5)
    else:
        timer = utils.ETA(n_runs*restarts,wait_time=5)
    callback = lambda i:timer() if use_timer else lambda i : None
    print('Starting MCMC ...')
    misfit_func = make_kriging_misfit([syn_chi],[syn_bot],probability_calculator,rep_sel=0)
    if use_PT:
        theta0 = prior.lb[:,None] + np.random.random(size=(8,restarts)) * (prior.ub-prior.lb)[:,None]
        MCMC_return = MCMC.parallel_tempering(theta0,lambda theta:theta,misfit_func,
                                prior.logpdf,proposal,PT_temperatures,n_runs,callback=callback)     
        xchain = MCMC_return[0]
        Lchain = MCMC_return[1]
        accepted = MCMC_return[2]
    else:
        for j in range(restarts):
            print('Restart %d/%d'%(j+1,restarts))
            theta0 = prior.lb + np.random.random(8) * (prior.ub-prior.lb)        
            MCMC_return = MCMC.MCMC(theta0,lambda theta:theta,misfit_func,prior.logpdf,proposal,n_runs,callback=callback)
            xchain[j,:,:] = MCMC_return[0]
            Lchain[j,:] = MCMC_return[1]
            accepted[j] = MCMC_return[2]
    return xchain,Lchain,accepted,theta0

def generate_inis(master_yaml,run_selector=['--all']):
    """Helper function to mass-generate inis from yaml file
    """
    with open(master_yaml,'r') as f:
        run_configs = yaml.load(f)
    
    if run_selector[0] =='--all':
        to_run = run_configs.keys()
    else:
        to_run = run_selector
    
    for key in to_run:
        run_config = run_configs[key]
        config = configparser.ConfigParser()
        config.read(run_config['ini_configs'])
        if 'overwrite_configs' in run_config:
            for section,par,val in run_config['overwrite_configs']:
                config[section][par] = val
        with open('%s.ini'%key,'w') as f:
            config.write(f)

def decider(func_name,ini_file):
    """Helper function to dynamically choose which function to run 
    """
    print('Decider is deciding',func_name,ini_file)
    if func_name == 'generate_and_dump':
        generate_and_dump(ini_file)
    elif func_name =='experiment1b':
        experiment1b(ini_file)
    elif func_name =='experiment3a_piggyback':
        experiment3a_piggyback(ini_file)
    elif func_name =='run_with_data_hyperparameters':
        run_with_data_hyperparameters(ini_file)
