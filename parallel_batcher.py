import subprocess
import argparse
from filelock import FileLock
import shutil
import glob
import os
import re
import time
import h5py
import json
from mpl_toolkits.basemap import Basemap
import numpy as np
import oo_implementation
import json
import pickle

import datetime

import importlib
importlib.reload(oo_implementation)

def set_spacing(spacing):
    sources = es_problem._create_block_sources(spacing)
    print('Controller: Setting source spacing',spacing,'dipole_shape',sources.block_row.shape)
    return sources

def store_settings(phase):
    """Apply setting_overrides (global var.) to settings (global var.) and store to file
    """
    for key in setting_overrides[phase]:
        settings.__dict__[key] = setting_overrides[phase][key]
    
    with open(f'tmp/settings_{phase}.json','w') as f:
        json.dump(settings.to_json_dict(),f,indent=4)

def read_out(fname):
    eqs_dict = dict()
    pred_dict = dict()
    with h5py.File(fname) as hdf_file:
        if 'eqs' in hdf_file:
            for dset_name in hdf_file['eqs']:
                dset = hdf_file['eqs'][dset_name]
                phase = dset.attrs['phase']
                i = dset.attrs['i']
                j = dset.attrs['j']
                eqs_dict[phase,i,j] = dset[:]
        if 'pred' in hdf_file:
            for dset_name in hdf_file['pred']:
                dset = hdf_file['pred'][dset_name]
                phase = dset.attrs['phase']
                i = dset.attrs['i']
                j = dset.attrs['j']
                pred_dict[phase,i,j] = dset[:]
    return eqs_dict,pred_dict

def store_new_results(eqs_dict,pred_dict):
    with h5py.File(f'tmp/results/out_{phase}_{repetition}.hdf','a') as hdf_file:
        for _,i,j in eqs_dict:
            dset = hdf_file.create_dataset(f'out_{phase}_{i}_{j}',data=eqs_dict[phase,i,j])
            dset.attrs['phase'] = phase
            dset.attrs['i'] = i
            dset.attrs['j'] = j
    
    with h5py.File(f'tmp/results/pred_{phase}_{repetition}.hdf','a') as hdf_file:
        for _,i,j in  pred_dict:
            dset = hdf_file.create_dataset(f'pred_{phase}_{i}_{j}',data=pred_dict[phase,i,j])
            dset.attrs['phase'] = phase
            dset.attrs['i'] = i
            dset.attrs['j'] = j

def read_pred(phase,repetition):
    ## Phase is complete -> Read in the data in predicted and farfield
    local_sources = es_problem._create_block_sources(spacings[phase])
    block_stat = local_sources.block_definition.assign(*worldmap(es_input.lonlatz[:,0],es_input.lonlatz[:,1]))
    pred = np.zeros(es_input.N_data)
    farfield = np.zeros(es_input.N_data)
    fname = f'tmp/results/pred_{phase}_{repetition}.hdf'
    with h5py.File(fname) as hdf_file:
        for n,(i,j) in enumerate(local_sources.block_definition):
            temp = hdf_file[f'pred_{phase}_{i}_{j}'][:,:,0] 
            in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)
            pred[in_block_stat] = temp[:,0]
            farfield[in_block_stat] = temp[:,1]
    return pred,farfield

def get_eqs(block_list):
    """Get equivalent source results from own archive (for ff prep.)
    """
    eqs_dict = dict()
    with h5py.File(f'tmp/results/out_{phase}_{repetition}.hdf','a') as hdf_file:
        for i,j in block_list:
            eqs_dict[phase,i,j] = hdf_file[f'out_{phase}_{i}_{j}'][:]
    return eqs_dict

def update_blockstate(old_block_state,n_neigh_complete_old):
    block_state = old_block_state.copy()
    n_neigh_complete = n_neigh_complete_old.copy()
    open_blocks = list(zip(*np.where(block_state==0)))
    changed_blocks = set()
    with h5py.File(f'tmp/results/out_{phase}_{repetition}.hdf','r') as hdf_file:
        for i,j in open_blocks:
            if f'out_{phase}_{i}_{j}' in hdf_file:
                block_state[i,j] = 1
                changed_blocks.add((i,j))
    
    for i,j in changed_blocks:
        for i2,j2 in  sources.block_definition.neighbors(i,j,settings.far_field_outer):
            n_neigh_complete[i2,j2] = n_neigh_complete[i2,j2] + 1
            inv_blocks = list(zip(*np.where(block_state==1)))
        
    ## Check if all neighbors are complete
    inv_blocks = list(zip(*np.where(block_state==1)))
    for i,j in inv_blocks:
        if n_neigh_complete[i,j] == n_neighbors[i,j]:
            block_state[i,j] = 2    

    ff_queued = list(zip(*np.where(block_state==3)))
    with h5py.File(f'tmp/results/pred_{phase}_{repetition}.hdf','r') as hdf_file:
        for i,j in ff_queued:
            if f'pred_{phase}_{i}_{j}' in hdf_file:
                block_state[i,j] = 4
    
    return block_state,n_neigh_complete

def prep_inv_tasklists(phase,n_tasks,correction=None):
    """Creates input files for inversion step
    These are pretty simple because you only need to know the input data (aeromag)
    Parameters
    ----------
    phase : int
        Identifier for the phase
    n_tasks : int
        Number of tasks per task list
    correction : np.ndarray
        Correction values from (1) previous runs, (2) correcting far-field
        effects and/or (3) survey shifts
    """
    if correction is None:
        correction = np.zeros(len(aeromag))
    block_stat = sources.block_definition.assign(*worldmap(es_input.lonlatz[:,0],es_input.lonlatz[:,1]))
    
    block_groups = []
    blocks_to_invert = []
    task_counter = 0
    for n,(i,j) in enumerate(sources.block_definition):
        blocks_to_invert.append((i,j))
        task_counter = task_counter + 1
        if task_counter == n_tasks:
            task_counter = 0
            block_groups.append(blocks_to_invert)
            blocks_to_invert = []
    if blocks_to_invert:
        block_groups.append(blocks_to_invert)
    
    out_files = []
    in_files = []
    for tl_counter,blocks_to_invert in enumerate(block_groups):
        fname = f'tmp/input/tl_inv_{tl_counter}.hdf'
        all_neighbors = find_neighbors(blocks_to_invert,outer=1)
        in_files.append(fname)
        with h5py.File(fname,'w') as hdf_file:
            hdf_file.attrs['out_name'] = f'tmp/results/tl_inv_{tl_counter}.hdf'
            out_files.append(hdf_file.attrs['out_name'])
            hdf_file.create_group("aeromag")
            hdf_file.create_group("eqs")
            hdf_file.create_group("aux")
            for i,j in all_neighbors:
                in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)
                sub_aux = aeromag[in_block_stat].copy()
                sub_aux[:,4] = sub_aux[:,4] - correction[in_block_stat]
                data = np.hstack((sub_aux,aeromag_year[in_block_stat,None],err_vals[in_block_stat,None]))
                dset = hdf_file.create_dataset(f'aeromag/data_{phase}_{i}_{j}',data=data)
                dset.attrs['i'] = i
                dset.attrs['j'] = j
                dset.attrs['phase'] = phase
                if (i,j) in blocks_to_invert:
                    dset.attrs['is_inv'] = True
                else:
                    dset.attrs['is_inv'] = False
                dset.attrs['is_ff'] = False
                dset.attrs['is_aux'] = False

    
    for out_name in out_files:
        try:
            os.remove(out_name)
        except OSError as e:
            print(f'Warning: Failed to remove outputfile {out_name}')
        try:
            os.remove(out_name+'.done')
        except OSError as e:
            print(f'Warning: Failed to remove outputfile {out_name}.done')

    return in_files,out_files


def prep_ff_tasklists(phase,ff_blocks,tl_counter_ff):
    print('Prepping ff tasklist for blocks')
    block_stat = sources.block_definition.assign(*worldmap(es_input.lonlatz[:,0],es_input.lonlatz[:,1]))
    in_file = f'tmp/input/tl_ff_{tl_counter_ff}.hdf'
    all_neighbors = find_neighbors(ff_blocks,settings.far_field_outer)
    eqs_dict = get_eqs(all_neighbors)

    with h5py.File(in_file,'w') as hdf_file:
        hdf_file.attrs['out_name'] = f'tmp/results/tl_ff_{tl_counter_ff}.hdf'
        out_file = hdf_file.attrs['out_name']
        hdf_file.create_group("aeromag")
        hdf_file.create_group("aux")

        for i,j in ff_blocks:
            in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)
            sub_aeromag = aeromag[in_block_stat].copy()
            data = np.hstack((sub_aeromag,aeromag_year[in_block_stat,None],err_vals[in_block_stat,None]))
            dset = hdf_file.create_dataset(f'aeromag/data_{phase}_{i}_{j}',data=data)
            dset.attrs['i'] = i
            dset.attrs['j'] = j
            dset.attrs['phase'] = phase
            dset.attrs['is_ff'] = True
            dset.attrs['is_inv'] = False
            dset.attrs['is_aux'] = False
            
        hdf_file.create_group("eqs")
        for i,j in all_neighbors:
            dset = hdf_file.create_dataset(f'eqs/eqs_{phase}_{i}_{j}',data=eqs_dict[phase,i,j])
            dset.attrs['i'] = i
            dset.attrs['j'] = j
            dset.attrs['phase'] = phase
    try:
        os.remove(out_file)
        os.remove(out_file+'.done')

    except OSError as e:
        print(f'Warning: Failed to remove outputfile {out_file}')

    return in_file,out_file

def find_neighbors(blocks_to_invert,outer):
    all_neighbors = set()
    for i,j in blocks_to_invert:
        for i2,j2 in sources.block_definition.neighbors(i,j,outer):
            all_neighbors.add((i2,j2))
    return all_neighbors
    


def get_n_neighbors(block_definition):
    n_neighbors = np.zeros(block_definition.shape,dtype=int)

    for i,j in block_definition:
        n_neighbors[i,j] = len(list(block_definition.neighbors(i,j,settings.far_field_outer)))

    return n_neighbors

def get_strftime():
    return  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def start_worker(fname):
    return subprocess.Popen(["python","batch_worker.py",code,fname],stdout=subprocess.PIPE)


parser = argparse.ArgumentParser()
parser.add_argument('code',help='Area code for which to run the inversion')
parser.add_argument('phase',default=0,type=int,help='Run this phase')
parser.add_argument('repetition',default=0,type=int,help='Run this repetition')
parser.add_argument('-n','-n-tasks',type=int,default=8)
parser.add_argument('-w','-workers',default=0,type=int)
parser.add_argument('--no-shift',default=False,type=bool)
parsed = vars(parser.parse_args())

code = parsed['code']
phase = parsed['phase']
repetition = parsed['repetition']
n_task = parsed['n']
n_workers = parsed['w']
no_shift = parsed['no_shift']
sleep_time = 5
## Setup
with open('tmp/code.txt','w') as f:
    f.write(code)

with open(f'data/{code}.json') as f:
    config_dict = json.load(f)

used_projection = config_dict['used_projection']
EW_area_f = config_dict['EW_area_f']
NS_area_f = config_dict['NS_area_f']
center_lat_f = config_dict['center_lat_f']
center_long_f = config_dict['center_long_f']
worldmap = Basemap(projection=used_projection,width=EW_area_f,height=NS_area_f,lat_0=center_lat_f,
                   lon_0=center_long_f,resolution="l")
corners = worldmap.llcrnrx,worldmap.urcrnrx,worldmap.llcrnry,worldmap.urcrnry
corners_shifted = worldmap.llcrnrx-1,worldmap.urcrnrx+1,worldmap.llcrnry-1,worldmap.urcrnry+1

aeromag = np.load(f'data/{code}_aeromag.npy')
aeromag_year = np.load(f'data/{code}_aeromag_year.npy')
err_vals = np.load(f'data/{code}_err_vals.npy')

year_DGRF = np.array([2015, 2010, 2005, 2000, 1995, 1990, 1985, 1980, 1975, 1970]) 

## TODO: I'm not sure if this works correctly with the corners. There is a duality here, because the corners
## are both defined by the worldmap settings as well as stored in the settings object

with open(f'tmp/master_settings.json','r') as f:
    thing = json.load(f)
    if 'corner_buffers' in thing:
        del thing['corner_buffers']

settings = oo_implementation.ESSettings(**thing)
settings.corners = corners
setting_overrides = [dict(lambda_d=1e-4,lambda_s=1e-4,lambda_factor_override=5e-15),
                            dict(lambda_d=3e-4,lambda_s=3e-3,lambda_factor_override=2e-12),
                            dict(lambda_d=3e-4,lambda_s=3e-3,lambda_factor_override=2.5e-10)]## END (Things which should be read in)## END (Things which should be read in)
spacings = [10000,2500,700]
es_input = oo_implementation.ESInput(aeromag[:,0],aeromag[:,1:4],aeromag[:,4],aeromag[:,5:8],aeromag_year,err_vals)
es_problem = oo_implementation.ESProblem(settings,es_input,worldmap)

previous = np.zeros(es_input.N_data)

for p in range(phase):
    pred_old,farfield_old = read_pred(p,2)
    previous = previous + pred_old + farfield_old


if repetition > 0:
    pred,farfield = read_pred(phase,repetition-1)
else:
    farfield = np.zeros(es_input.N_data)
    pred = np.zeros(es_input.N_data)

shifts,shift_vec = oo_implementation.get_shift(es_input,pred+previous,farfield)

if no_shift or (phase==0 and repetition==0):
    shifts[:] = 0
    shift_vec[:] = 0

print(f'Shift values ({min(shift_vec)} to {max(shift_vec)})')

## Finish input

spacing = spacings[phase]
sources = set_spacing(spacing)
store_settings(phase)

with open(f'tmp/sources_{code}_{phase}.pkl','wb') as f:
    pickle.dump(sources,f)

with open(f'tmp/sources_{code}_{phase}.pkl','rb') as f:
    sources = pickle.load(f)

## Prepare my own output files
with h5py.File(f'tmp/results/out_{phase}_{repetition}.hdf','w') as hdf_file:
    pass

with h5py.File(f'tmp/results/pred_{phase}_{repetition}.hdf','w') as hdf_file:
    pass

## Gather all possible tasks
## Divide into tasklists
## Write out tasklist instructions
## Wait for completion
## Update all possible tasks
## Create new task lists


block_state = np.zeros(sources.block_definition.shape,dtype=int)
n_neigh_complete = np.zeros(sources.block_definition.shape,dtype=int)

n_neighbors = get_n_neighbors(sources.block_definition)
block_stat = sources.block_definition.assign(*worldmap(es_input.lonlatz[:,0],es_input.lonlatz[:,1]))

infiles,outfiles = prep_inv_tasklists(phase,n_task,previous+farfield+shift_vec)
tl_counter_ff = 0

workers = []

while True:
    time.sleep(sleep_time)
    found_files = []
    for fname in outfiles:
        try:
            if os.path.isfile(fname+'.done'):
                print(f'Heureka! {fname} is done',end = " ")
                eqs_dict,pred_dict = read_out(fname)
                print(f'Contains {len(eqs_dict)} eqs and {len(pred_dict)} pred')
                store_new_results(eqs_dict,pred_dict)
                found_files.append(fname)
        except OSError as e:
            print(f'Errors over errors reading {fname}')
    
    for fname in found_files:
        outfiles.remove(fname)

    block_state,n_neigh_complete = update_blockstate(block_state,n_neigh_complete)

    ## Upate ff tasklists    
    while True:
        ready_for_ff = list(zip(*np.where(block_state==2)))
        da_kommt_noch_was = np.any(block_state<2)
        if len(ready_for_ff)>=n_task or (not da_kommt_noch_was and len(ready_for_ff)>0):
            sel_blocks = ready_for_ff[:n_task]
            in_file,out_name = prep_ff_tasklists(phase,sel_blocks,tl_counter_ff)
            outfiles.append(out_name)
            infiles.append(in_file)
            tl_counter_ff = tl_counter_ff + 1
            for i,j in sel_blocks:
                block_state[i,j] = 3
        else:
            break
    
    print(get_strftime(),'Block states',np.bincount(block_state.flatten()))
    print(f'No. blocks ready for ff {len(ready_for_ff)}')

    ## Check worker status
    living_workers = []
    for w in workers:
        if w.poll() is None:
            living_workers.append(w)
    workers = living_workers
    if n_workers>0:
        print(f'No. of living workers {len(workers)}')

    while len(workers) < n_workers and infiles:
        in_file = infiles.pop()
        workers.append(start_worker(in_file))
        print(f'Started new worker')

    if np.all(block_state==4):
        break


