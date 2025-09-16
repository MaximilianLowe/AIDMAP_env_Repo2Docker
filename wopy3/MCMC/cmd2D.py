#!/usr/bin/env python
# coding: utf-8

import configparser
import numpy as np
import sys
from wopy3.MCMC.transdim import RotoRectorModel, decompress_models
sys.path.append("Z:")
sys.path.append("D:/repositories")
from wopy3.MCMC.transdim import TransDimPointProposal,transdimensional_MCMC,compress_models,VariableBirthTransDimProposal
from wopy3.MCMC.MCMC import transdim_parallel_tempering,pt_with_recombination
from wopy3.MCMC.distributions import RangePrior,ComponentUpdateProposal
from wopy3.utils import ETA
import scipy.stats
import wopy3.forward.mineos_link as mineos_link
import pickle
from wopy3.MCMC.blobography import CompoundForward,MineosForward,GravityForward,CompoundMisfit,generate_grav_misfit_obj,generate_model_ensemble
from wopy3.MCMC.blobography import generate_dvph_misfit_object
from wopy3.utils import get_comma_separated_float
from collections import defaultdict
import argparse
import os


def read_ini(fname):
    config = configparser.ConfigParser()
    config.read(fname)

    config_dict = defaultdict(dict)

    config_dict['GEOMETRY']['lon_ax'] = np.arange(*get_comma_separated_float(config,'GEOMETRY','lon_range'))
    config_dict['GEOMETRY']['lon_km_ax'] = config_dict['GEOMETRY']['lon_ax'] * 110.0
    config_dict['GEOMETRY']['dep_ax'] = np.arange(*get_comma_separated_float(config,'GEOMETRY','dep_range'))

    section = config['MINEOS']
    config_dict['MINEOS']['lmax'] = section.getint('lmax')
    config_dict['MINEOS']['ref_model_file'] = section.get('ref_model_file')
    config_dict['MINEOS']['perturbations'] = np.array(get_comma_separated_float(config,'MINEOS','perturbations'))
    config_dict['MINEOS']['n_workers'] = section.getint('n_workers')
    config_dict['MINEOS']['pkl_file'] = section.get('pkl_file')
    config_dict['MINEOS']['apply_filter'] = section.getboolean('apply_filter')
    config_dict['MINEOS']['input_data_file'] = section.get('input_data_file')
    config_dict['MINEOS']['freq_sel_override'] = np.array(get_comma_separated_float(config,'MINEOS','freq_sel_override')).astype(int)
    config_dict['MINEOS']['error_correlation_mode'] = section.get('error_correlation_mode')
    config_dict['MINEOS']['n_ensemble'] = section.getint('n_ensemble')
    config_dict['MINEOS']['lim_sigma'] = np.array(get_comma_separated_float(config,'MINEOS','lim_sigma'))
    config_dict['MINEOS']['prop_sigma'] = section.getfloat('prop_sigma')
    config_dict['MINEOS']['sigma_0'] = section.getfloat('sigma_0')

    section = config['GRAVITY']
    config_dict['GRAVITY']['use_grad'] = section.getboolean('use_grad')
    config_dict['GRAVITY']['use_grav'] = section.getboolean('use_grav')
    config_dict['GRAVITY']['use_pot'] = section.getboolean('use_pot')
    config_dict['GRAVITY']['input_data_file'] = section.get('input_data_file')
    config_dict['GRAVITY']['error_correlation_mode'] = section.get('error_correlation_mode')
    config_dict['GRAVITY']['n_ensemble'] = section.getint('n_ensemble')
    config_dict['GRAVITY']['lim_sigma'] = np.array(get_comma_separated_float(config,'GRAVITY','lim_sigma'))
    config_dict['GRAVITY']['prop_sigma'] = section.getfloat('prop_sigma')
    config_dict['GRAVITY']['sigma_0'] = section.getfloat('sigma_0')

    upper_bounds = []
    lower_bounds = []
    property_proposal_var = []
    move_proposal_var = []

    for key in 'width,height,rot,vs,rho'.split(','):
        lb,ub = get_comma_separated_float(config,'PROPOSAL','lim_%s'%key)
        upper_bounds.append(ub)
        lower_bounds.append(lb)
        property_proposal_var.append(config['PROPOSAL'].getfloat('prop_%s'%key)**2)

    move_proposal_var.append(config['PROPOSAL'].getfloat('move_lon_km')**2)
    move_proposal_var.append(config['PROPOSAL'].getfloat('move_dep_km')**2)

    config_dict['PROPOSAL']['upper_bounds'] = upper_bounds
    config_dict['PROPOSAL']['lower_bounds'] = lower_bounds
    config_dict['PROPOSAL']['property_proposal_var'] = np.array(property_proposal_var)
    config_dict['PROPOSAL']['move_proposal_var'] = np.array(move_proposal_var)
    config_dict['PROPOSAL']['proposal_type'] = config['PROPOSAL'].get('proposal_type')
    config_dict['PROPOSAL']['birth_scale'] = config['PROPOSAL'].getfloat('birth_scale')

    section = config['INVERSION']
    config_dict['INVERSION']['start_model_file'] = section.get('start_model_file')
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
    assert np.allclose(config_dict['GEOMETRY']['lon_ax'],rayleigh_data[:,0])
    assert len(freq_sel_header) == rayleigh_data.shape[1]-1
    rayleigh_data = rayleigh_data[:,1:]
    print(freq_sel_header)
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
    dep_ax = config_dict['GEOMETRY']['dep_ax']
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

def rescale_birth_prior(prop_prior,birth_scale):
    new_lb = prop_prior.lb * birth_scale
    new_ub = prop_prior.ub * birth_scale
    # For size, the new value shouldn't be below the prior
    new_lb[0] = prop_prior.lb[0]
    new_lb[1] = prop_prior.lb[1]

    return RangePrior(new_lb,new_ub)


def create_proposal(config_dict):
    section = config_dict['PROPOSAL']
    lon_km_ax = config_dict['GEOMETRY']['lon_km_ax']
    dep_ax = config_dict['GEOMETRY']['dep_ax']    

    prop_prior = RangePrior(section['lower_bounds'],section['upper_bounds'])
    if section['proposal_type']=='mvn':
        prop_proposal = scipy.stats.multivariate_normal(mean=np.zeros(5),
                                                    cov=np.eye(5)*section['property_proposal_var'])
        move_proposal = scipy.stats.multivariate_normal(mean=np.zeros(2),cov=np.eye(2)*section['move_proposal_var'])
    elif section['proposal_type'] == 'cup':
        prop_proposal = ComponentUpdateProposal(np.sqrt(section['property_proposal_var']))
        move_proposal = ComponentUpdateProposal(np.sqrt(section['move_proposal_var']))
    
    if section['birth_scale'] is None:
        proposal = TransDimPointProposal([[lon_km_ax.min(),lon_km_ax.max()],[dep_ax.min(),dep_ax.max()]],
                                 prop_proposal,
                                 prop_prior,
                                 move_proposal,
                                 birth_from_prior=True)
    else:
        birth_proposal = rescale_birth_prior(prop_prior,section['birth_scale'])
        proposal = VariableBirthTransDimProposal([[lon_km_ax.min(),lon_km_ax.max()],[dep_ax.min(),dep_ax.max()]],
                                 prop_proposal,
                                 prop_prior,
                                 move_proposal,
                                 prop_birth=birth_proposal)
    return proposal

def setup(fname,recalculate,no_overwrite):
    config_dict = read_ini(fname)

    lon_ax = config_dict['GEOMETRY']['lon_ax']
    dep_ax = config_dict['GEOMETRY']['dep_ax']
    lon_km_ax = config_dict['GEOMETRY']['lon_km_ax']
    loni_km,depi = np.meshgrid(lon_km_ax,dep_ax)

    rayleigh_data,freq_sel = read_rayleigh_data(config_dict)    
    config_dict['MINEOS']['freq_sel'] = freq_sel
    ref_model = np.loadtxt(config_dict['MINEOS']['ref_model_file'],skiprows=3)
    interpolated_ref_model = mineos_link.interp_prem(ref_model,1000*dep_ax)

    grav_data = np.loadtxt(config_dict['GRAVITY']['input_data_file']) 
    # Probably some header information would be nice, what the file actually contains
    # Alternatively, you can always force the 4-column model but simply require
    # all-nan or non-sensical values
    lons = grav_data[:,0]
    grav_true = grav_data[:,1:] 

    config_dict['GRAVITY']['lons'] = lons

    mineos_forward = MineosForward(lon_ax,dep_ax,create_mineos_approx(config_dict,ref_model,recalculate),
                                apply_filter=config_dict['MINEOS']['apply_filter'],
                                freq_sel=config_dict['MINEOS']['freq_sel'])

    grav_forward = GravityForward(lon_ax,dep_ax,interpolated_ref_model,lons=lons)
    grav_forward.use_grad = config_dict['GRAVITY']['use_grad']
    grav_forward.use_grav = config_dict['GRAVITY']['use_grav']
    grav_forward.use_pot = config_dict['GRAVITY']['use_pot']
    n_grav = grav_forward.use_grav + grav_forward.use_pot + grav_forward.use_grad
    assert n_grav <= grav_true.shape[1]

    proposal = create_proposal(config_dict)    
    if config_dict['INVERSION']['start_model_file'] is None:
        start_model = generate_model_ensemble(proposal,1,1,loni_km,depi)[0]
    else:
        compressed_start_model = np.load(config_dict['INVERSION']['start_model_file'])
        start_model = decompress_models(compressed_start_model,RotoRectorModel.create_from_vectors,2,loni_km,depi)
    assert ~np.isinf(proposal.log_prior(start_model))

    dvph_misfit,flatten = generate_dvph_misfit_object(config_dict['MINEOS']['error_correlation_mode'],rayleigh_data,mineos_forward,proposal,
                                                    config_dict['MINEOS']['n_ensemble'])
    mineos_forward.flatten = flatten


    grav_misfit = generate_grav_misfit_obj(grav_true,config_dict['GRAVITY']['error_correlation_mode'],
                                        proposal=proposal,grav_forward=grav_forward,
                                        n_ensemble=config_dict['GRAVITY']['n_ensemble'])
    
    if config_dict['GRAVITY']['error_correlation_mode'] == 'covariance-global':
        grav_forward.flatten = True

    try:
        np.load(config_dict['OUTPUT']['out_file'])
    except FileNotFoundError:
        pass
    else:
        if no_overwrite:
            raise FileExistsError('Output file exists and should not be overwritten!')

    manager = CompoundForward([mineos_forward,grav_forward])
    misfit = CompoundMisfit([dvph_misfit,grav_misfit],hy_indices=[[0],[1]])

    return start_model,manager,misfit,proposal


def run(fname,recalculate,no_overwrite):
    config_dict = read_ini(fname)
    lon_ax = config_dict['GEOMETRY']['lon_ax']
    dep_ax = config_dict['GEOMETRY']['dep_ax']
    start_model,manager,misfit,proposal = setup(fname,recalculate,no_overwrite)
    
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



    print(proposal.log_prior(start_model))
    ## There are three "levels" of running the inversion
    # With just a single chain
    # With parallel tempering
    # With parallel tempering + recombination

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
        start_models = generate_model_ensemble(lon_ax,dep_ax,n_chains,proposal,1)
        
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

        start_models = generate_model_ensemble(lon_ax,dep_ax,initial_ensemble_size,proposal,1)
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