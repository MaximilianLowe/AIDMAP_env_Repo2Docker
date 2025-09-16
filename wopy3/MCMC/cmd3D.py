#!/usr/bin/env python
# coding: utf-8

import configparser
import numpy as np
from wopy3.MCMC.blobography import TesseroidModel, create_model
from wopy3.MCMC.transdim import RotoRectorModel, compress_models
from wopy3.MCMC.transdim import TransDimPointProposal,transdimensional_MCMC
from wopy3.MCMC.MCMC import transdim_parallel_tempering,pt_with_recombination
from wopy3.MCMC.distributions import RangePrior,ComponentUpdateProposal,CorrelatedGaussian
from wopy3.utils import ETA
import scipy.stats
import wopy3.forward.mineos_link as mineos_link
import pickle
from wopy3.MCMC.blobography import CompoundForward,TessMineosForward,GravityForward3D,CompoundMisfit,generate_grav_misfit_obj,generate_model_ensemble
from wopy3.MCMC.blobography import generate_dvph_misfit_object
from wopy3.utils import get_comma_separated_float
from collections import defaultdict
import argparse
import os

class GeometryDescription:
    """Describes the boundaries and voxelization of a spherical 3D volume
    """
    def __init__(self,lon_ax,dep_ax,lat_ax=None):
        self.lon_ax = lon_ax
        self.dep_ax = dep_ax
        self.lat_ax = lat_ax
        if lat_ax is None:
            self.spatial_dim = 2
        else:
            self.spatial_dim = 3

    @property
    def lon_km_ax(self):
        return 110.0*self.lon_ax

    @property
    def lat_km_ax(self):
        return 110.0*self.lat_ax

    @property
    def bounding_box(self):
        if self.spatial_dim == 3:
            return [[self.lon_km_ax.min(),self.lon_km_ax.max()],
            [self.lat_km_ax.min(),self.lat_km_ax.max()],
            [self.dep_ax.min(),self.dep_ax.max()]]

    def get_meshgrids(self,in_km=True):
        if self.spatial_dim == 3:
            if in_km:
                return np.meshgrid(self.lon_km_ax,self.lat_km_ax,self.dep_ax)
            else:
                return np.meshgrid(self.lon_ax,self.lat_ax,self.dep_ax)
        


def read_ini(fname):
    config = configparser.ConfigParser()
    config.read(fname)

    config_dict = defaultdict(dict)
    lon_ax = np.arange(*get_comma_separated_float(config,'GEOMETRY','lon_range'))
    lat_ax = np.arange(*get_comma_separated_float(config,'GEOMETRY','lat_range'))
    dep_ax = np.arange(*get_comma_separated_float(config,'GEOMETRY','dep_range'))

    geometry = GeometryDescription(lon_ax,dep_ax,lat_ax)
    config_dict['GEOMETRY'] = geometry
    
    section = config['MINEOS']
    config_dict['MINEOS']['lmax'] = section.getint('lmax')
    config_dict['MINEOS']['ref_model_file'] = section.get('ref_model_file')
    config_dict['MINEOS']['perturbations'] = np.array(get_comma_separated_float(config,'MINEOS','perturbations'))
    config_dict['MINEOS']['n_workers'] = section.getint('n_workers')
    config_dict['MINEOS']['pkl_file'] = section.get('pkl_file')
    config_dict['MINEOS']['apply_filter'] = section.getboolean('apply_filter')
    config_dict['MINEOS']['input_data_file'] = section.get('input_data_file')
    config_dict['MINEOS']['freq_sel_override'] = np.array(get_comma_separated_float(config,'MINEOS','freq_sel_override')).astype(int)
    config_dict['MINEOS']['n_ensemble'] = section.getint('n_ensemble')
    config_dict['MINEOS']['lim_sigma'] = np.array(get_comma_separated_float(config,'MINEOS','lim_sigma'))
    config_dict['MINEOS']['prop_sigma'] = section.getfloat('prop_sigma')
    config_dict['MINEOS']['sigma_0'] = section.getfloat('sigma_0')

    section = config['GRAVITY']
    config_dict['GRAVITY']['use_grad'] = section.getboolean('use_grad')
    config_dict['GRAVITY']['use_grav'] = section.getboolean('use_grav')
    config_dict['GRAVITY']['use_pot'] = section.getboolean('use_pot')
    config_dict['GRAVITY']['input_data_file'] = section.get('input_data_file')
    config_dict['GRAVITY']['input_unit'] = section.get('input_unit')
    config_dict['GRAVITY']['n_ensemble'] = section.getint('n_ensemble')
    config_dict['GRAVITY']['lim_sigma'] = np.array(get_comma_separated_float(config,'GRAVITY','lim_sigma'))
    config_dict['GRAVITY']['prop_sigma'] = section.getfloat('prop_sigma')
    config_dict['GRAVITY']['sigma_0'] = section.getfloat('sigma_0')

    upper_bounds = []
    lower_bounds = []
    property_proposal_var = []
    move_proposal_var = []

    for key in 'lon_size,lat_size,dep_size,vs,rho'.split(','):
        lb,ub = get_comma_separated_float(config,'PROPOSAL','lim_%s'%key)
        upper_bounds.append(ub)
        lower_bounds.append(lb)
        property_proposal_var.append(config['PROPOSAL'].getfloat('prop_%s'%key)**2)

    move_proposal_var.append(config['PROPOSAL'].getfloat('move_lon_km')**2)
    move_proposal_var.append(config['PROPOSAL'].getfloat('move_lat_km')**2)
    move_proposal_var.append(config['PROPOSAL'].getfloat('move_dep_km')**2)

    config_dict['PROPOSAL']['upper_bounds'] = upper_bounds
    config_dict['PROPOSAL']['lower_bounds'] = lower_bounds
    config_dict['PROPOSAL']['property_proposal_var'] = np.array(property_proposal_var)
    config_dict['PROPOSAL']['move_proposal_var'] = np.array(move_proposal_var)

    section = config['INVERSION']
    config_dict['INVERSION']['use_variable_sigma'] = section.getboolean('use_variable_sigma')
    config_dict['INVERSION']['use_parallel_tempering'] = section.getboolean('use_parallel_tempering')
    config_dict['INVERSION']['use_recombination'] = section.getboolean('use_recombination')
    config_dict['INVERSION']['n_parallel_chains'] = section.getint('n_parallel_chains')
    config_dict['INVERSION']['n_recombinations'] = section.getint('n_recombinations')
    config_dict['INVERSION']['n_runs'] = section.getint('n_runs')

    config_dict['INVERSION']['save_step'] = section.getint('save_step')
    config_dict['INVERSION']['pt_max_temp'] = section.getfloat('pt_max_temp')
    config_dict['INVERSION']['pt_cold_fraction'] = section.getfloat('pt_cold_fraction')

    config_dict['INVERSION']['rc_initial_ensemble_size'] = section.getint('rc_initial_ensemble_size')
    config_dict['INVERSION']['rc_recombination_ensemble_size'] = section.getint('rc_recombination_ensemble_size')
    config_dict['INVERSION']['rc_n_offspring'] = section.getint('rc_n_offspring')

    config_dict['OUTPUT']['out_file'] = config['OUTPUT'].get('out_file')
    config_dict['OUTPUT']['no_overwrite'] = config['OUTPUT'].getboolean('no_overwrite')

    return config_dict

def read_rayleigh_data(config_dict):
    with open(config_dict['MINEOS']['input_data_file']) as f:
        line = f.readlines()[1]
    freq_sel_header = -2 + np.array(list(map(float,line[3:-2].split()))).astype(int)
    rayleigh_data = np.loadtxt(config_dict['MINEOS']['input_data_file'])
    #TODO: Add option for interpolating.
    #Due to rounding errors, I expect the assertion below to fail, but I'll have to see
    assert np.allclose(config_dict['GEOMETRY'].lon_ax,np.unique(rayleigh_data[:,0]))
    assert np.allclose(config_dict['GEOMETRY'].lat_ax,np.unique(rayleigh_data[:,1]))
    assert len(freq_sel_header) == rayleigh_data.shape[1]-2
    rayleigh_data = rayleigh_data[:,2:]
    return rayleigh_data,freq_sel_header

def read_gravity_data(config_dict):
    grav_data = np.loadtxt(config_dict['GRAVITY']['input_data_file']) 
    stations = grav_data[:,:3]
    grav_true = grav_data[:,3:] 
    if config_dict['GRAVITY']['input_unit'] == 'mgal':
        grav_true[:,0] = grav_true[:,0] * 1e-5
    return stations,grav_true

def apply_freq_sel_override(config_dict,rayleigh_data,freq_sel_header):
    if np.any(config_dict['MINEOS']['freq_sel_override']):
        print('Applying freq_sel_override',config_dict['MINEOS']['freq_sel_override'])
        subbi = np.zeros(len(config_dict['MINEOS']['freq_sel_override']),dtype=int)
        for i,f in enumerate(config_dict['MINEOS']['freq_sel_override']):
            assert f in freq_sel_header
            subbi[i] = np.argmax(freq_sel_header==f)
        rayleigh_data = rayleigh_data[:,subbi]
        freq_sel = config_dict['MINEOS']['freq_sel_override']
    else:
        freq_sel = freq_sel_header
    return rayleigh_data,freq_sel

def create_mineos_approx(config_dict,ref_model,recalculate):
    dep_ax = config_dict['GEOMETRY'].dep_ax
    perturbations = config_dict['MINEOS']['perturbations']
    lmax = config_dict['MINEOS']['lmax']

    if not recalculate and os.path.exists(config_dict['MINEOS']['pkl_file']):
        with open(config_dict['MINEOS']['pkl_file'],'rb') as f:
            mineos_approx = pickle.load(f)
        assert np.allclose(mineos_approx.depths,dep_ax)
        assert np.allclose(mineos_approx.amplitudes,perturbations)
        assert np.allclose(mineos_approx.reference_model,ref_model)
        assert mineos_approx.ref_out[:,1].max() == lmax
    else:    
        n_workers = config_dict['MINEOS']['n_workers']
        pkl_file = config_dict['MINEOS']['pkl_file']
        mineos_approx = mineos_link.MineosApproximator(dep_ax,ref_model,perturbations,n_workers=n_workers,
                                                       mineos_kwd=dict(l_min_max=(0,lmax)))
        with open(pkl_file,'wb') as f:
            pickle.dump(mineos_approx,f)                              
    return mineos_approx

def create_proposal(config_dict):
    section = config_dict['PROPOSAL']
    
    prop_prior = RangePrior(section['lower_bounds'],section['upper_bounds'])
    prop_proposal = scipy.stats.multivariate_normal(mean=np.zeros(5),
                                                    cov=np.eye(5)*section['property_proposal_var'])
    move_proposal = scipy.stats.multivariate_normal(mean=np.zeros(3),cov=np.eye(3)*section['move_proposal_var'])
    
    
    proposal = TransDimPointProposal(config_dict['GEOMETRY'].bounding_box,
                                 prop_proposal,
                                 prop_prior,
                                 move_proposal,
                                 birth_from_prior=True)
    return proposal

def setup(fname,recalculate,no_overwrite):
    config_dict = read_ini(fname)

    geometry = config_dict['GEOMETRY']
    
    rayleigh_data,freq_sel_header = read_rayleigh_data(config_dict)    
    rayleigh_data,freq_sel = apply_freq_sel_override(config_dict,rayleigh_data,freq_sel_header)
    config_dict['MINEOS']['freq_sel'] = freq_sel
    ref_model = np.loadtxt(config_dict['MINEOS']['ref_model_file'],skiprows=3)
    interpolated_ref_model = mineos_link.interp_prem(ref_model,1000*geometry.dep_ax)
    density_interpolator = scipy.interpolate.interp1d(6371e3-interpolated_ref_model[:,0],interpolated_ref_model[:,1],
                                                    bounds_error=False,fill_value=np.nan)
    print(rayleigh_data.shape,freq_sel)


    mineos_forward = TessMineosForward(geometry.lon_ax,geometry.lat_ax,geometry.dep_ax,
                                create_mineos_approx(config_dict,ref_model,recalculate),
                                apply_filter=config_dict['MINEOS']['apply_filter'],
                                freq_sel=config_dict['MINEOS']['freq_sel'])
    stations,grav_true = read_gravity_data(config_dict)

    grav_forward = GravityForward3D(stations,density_interpolator)
    grav_forward.use_grad = config_dict['GRAVITY']['use_grad']
    grav_forward.use_grav = config_dict['GRAVITY']['use_grav']
    grav_forward.use_pot = config_dict['GRAVITY']['use_pot']
    n_grav = grav_forward.use_grav + grav_forward.use_pot + grav_forward.use_grad
    assert n_grav <= grav_true.shape[1]

    proposal = create_proposal(config_dict)    
    start_model = create_model(TesseroidModel,proposal,1,*geometry.get_meshgrids(in_km=True))
    start_model.grid = start_model.calc_grid()
    assert ~np.isinf(proposal.log_prior(start_model))

    misfits = [CorrelatedGaussian(rayleigh_data[:,i],sigma=np.eye(rayleigh_data.shape[0])*rayleigh_data[:,i].var()) for i in range(rayleigh_data.shape[1])]
    dvph_misfit = CompoundMisfit(misfits,hy_indices=np.zeros(rayleigh_data.shape[1],dtype=int),use_cols=True)
    mineos_forward.flatten = False

    grav_misfits = [CorrelatedGaussian(grav_true[:,i],sigma=np.eye(grav_true.shape[0])*grav_true[:,i].var()) for i in range(grav_true.shape[1])]
    grav_misfit = CompoundMisfit(grav_misfits,hy_indices=np.zeros(grav_true.shape[1],dtype=int))
    grav_forward.flatten = False

    try:
        np.load(config_dict['OUTPUT']['out_file'])
    except FileNotFoundError:
        pass
    else:
        if no_overwrite:
            raise FileExistsError('Output file exists and should not be overwritten!')
    
    manager = CompoundForward([mineos_forward,grav_forward])
    misfit = CompoundMisfit([dvph_misfit,grav_misfit],hy_indices=[[0],[1]])

    print(manager.calculators[0](start_model).shape)
    print(manager.calculators[1](start_model).shape)
    return start_model,manager,misfit,proposal

def run(fname,recalculate,no_overwrite):
    config_dict = read_ini(fname)
    start_model,manager,misfit,proposal = setup(fname,recalculate,no_overwrite)
    
    geometry = config_dict['GEOMETRY']
    if config_dict['INVERSION']['use_variable_sigma']:
        hyper_0 = np.array([config_dict['MINEOS']['sigma_0'],config_dict['GRAVITY']['sigma_0']])

        hyper_prior = RangePrior([config_dict['MINEOS']['lim_sigma'][0],config_dict['GRAVITY']['lim_sigma'][0]],
                                [config_dict['MINEOS']['lim_sigma'][1],config_dict['GRAVITY']['lim_sigma'][1]])

        hyper_proposal = ComponentUpdateProposal([config_dict['MINEOS']['prop_sigma'],
                                                config_dict['GRAVITY']['prop_sigma']])

        print(hyper_prior(hyper_0))
        wrapped_misfit = lambda y,hyper:misfit(y,hyper)
        print(misfit(manager.forward(start_model),hyper_0))

    else:
        hyper_0_fixed = np.array([config_dict['MINEOS']['sigma_0'],config_dict['GRAVITY']['sigma_0']])
        hyper_0 = None
        hyper_prior = None
        hyper_proposal = None
        wrapped_misfit = lambda y:misfit(y,hyper_0_fixed)
        print(wrapped_misfit(manager.forward(start_model)))
    nruns = config_dict['INVERSION']['n_runs']
    chainSave = config_dict['INVERSION']['save_step']

    if not config_dict['INVERSION']['use_parallel_tempering']:
        run_mode = 0
        print('Starting Normal MCMC')
        timer = ETA(nruns,wait_time=10)

        results_1 = transdimensional_MCMC(start_model,manager.forward,proposal,wrapped_misfit,nruns,
                                        hyper_0,hyper_proposal,hyper_prior,
                                                chainSave=chainSave,callback=lambda n:timer())

    elif not config_dict['INVERSION']['use_recombination']:
        run_mode = 1
        print('Starting Parallel Tempering MCMC')
        timer = ETA(nruns,wait_time=10)
        
        cold_fraction = config_dict['INVERSION']['pt_cold_fraction']
        n_chains = config_dict['INVERSION']['n_parallel_chains']
        max_temp = config_dict['INVERSION']['pt_max_temp']

        cold_chains = np.arange(n_chains) < cold_fraction * n_chains
        start_models = generate_model_ensemble(geometry.lon_ax,geometry.dep_ax,n_chains,proposal,1)
        
        PT_temperatures = np.ones(n_chains)
        PT_temperatures[~cold_chains] = np.power(10,np.random.random(size=(~cold_chains).sum())*np.log10(max_temp))
        
        results_PT = transdim_parallel_tempering(start_models,manager.forward,wrapped_misfit,proposal,PT_temperatures,
                                            nruns,chain_save=chainSave,callback=lambda n:timer(),
                                        hyper0=hyper_0,hyper_proposal=hyper_proposal,hyper_prior=hyper_prior)

    elif config_dict['INVERSION']['use_recombination']:
        run_mode = 2
        if not hyper_0 is None:
            raise NotImplementedError('Recombination with variable sigma not yet implemented!')
        print('Starting Parallel Tempering MCMC with recombination')
        
        cold_fraction = config_dict['INVERSION']['pt_cold_fraction']
        n_chains = config_dict['INVERSION']['n_parallel_chains']
        max_temp = config_dict['INVERSION']['pt_max_temp']
        initial_ensemble_size = config_dict['INVERSION']['rc_initial_ensemble_size']
        recombination_size = config_dict['INVERSION']['rc_recombination_ensemble_size']
        n_offspring = config_dict['INVERSION']['rc_n_offspring']
        repetitions = config_dict['INVERSION']['n_recombinations']

        start_models = generate_model_ensemble(geometry.lon_ax,geometry.dep_ax,initial_ensemble_size,proposal,1)
        ensemble,PT_memory,PT_temperatures = pt_with_recombination(start_models,proposal,wrapped_misfit,
                                                                manager.forward,repetitions,nruns,n_chains,
                                                                cold_fraction=cold_fraction,max_temp=max_temp,
                                                                    recombination_size=recombination_size,
                                                                n_offspring = n_offspring,
                                                                chain_save=chainSave,verbose=True)
        
    else:
        raise ValueError('Should never happen')


    # ### Store output

    store = dict()

    if run_mode == 0:
        key_list = ['models','Lchain','accepted']
        store['models'] = compress_models(results_1[0])
        store['Lchain'] = results_1[1]
        store['accepted'] = results_1[2]
        if len(results_1) == 4:
            store['hyper'] = results_1[3]
            key_list.append('hyper')
    elif run_mode == 1:
        key_list = ['models','Lchain','accepted','swaps']
        store['models'] = compress_models(results_PT[0])
        store['Lchain'] = results_PT[1]
        store['accepted'] = results_PT[2]
        store['swaps'] = results_PT[3]
        if len(results_PT) == 5:
            store['hyper'] = results_PT[4]    
            key_list.append('hyper')
            
    elif run_mode == 2:
        key_list = ['models','Lchain','accepted','swaps']    
        store['models'] = compress_models(np.array([PT_memory[i][0] for i in range(repetitions)]))
        store['Lchain'] = np.array([PT_memory[i][1] for i in range(repetitions)])
        store['accepted'] = np.array([PT_memory[i][2] for i in range(repetitions)])
        store['swaps'] = np.array([PT_memory[i][3] for i in range(repetitions)])
        if len(PT_memory[0])==5:
            store['hyper'] = np.array([PT_memory[i][4] for i in range(repetitions)])
            key_list.append('hyper')

    np.savez(config_dict['OUTPUT']['out_file'],*[store[key] for key in key_list])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname",help="ini file with settings")
    parser.add_argument("--recalculate",help="always recalculate mineos",
                        action="store_true")
    parser.add_argument("--no_overwrite",help="don't overwrite old output files",
                        action="store_true")             
    args = parser.parse_args()
    run(args.fname,args.recalculate,args.no_overwrite)