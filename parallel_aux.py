import subprocess
import argparse
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

def prep_aux_tasklists(phase,n_tasks,auxiliary,igrf_NED_aux,eqs_dict):
    aux_stat = sources.block_definition.assign(*worldmap(auxiliary[:,0],auxiliary[:,1]))
    
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
        fname = f'tmp/input/tl_aux_{tl_counter}.hdf'
        all_neighbors = find_neighbors(blocks_to_invert,outer=settings.far_field_outer)
        in_files.append(fname)
        with h5py.File(fname,'w') as hdf_file:
            hdf_file.attrs['out_name'] = f'tmp/results/tl_aux_{tl_counter}.hdf'
            out_files.append(hdf_file.attrs['out_name'])
            hdf_file.create_group("aux")
            hdf_file.create_group("eqs")
            hdf_file.create_group("aeromag")
            for i,j in blocks_to_invert:
                in_block_aux = (aux_stat[0] == i) & (aux_stat[1] == j)
                sub_aux = auxiliary[in_block_aux]
                sub_igrf_NED_aux = igrf_NED_aux[in_block_aux]
                data = np.hstack((sub_aux,sub_igrf_NED_aux))
                dset = hdf_file.create_dataset(f'aux/data_{phase}_{i}_{j}',data=data)
                dset.attrs['i'] = i
                dset.attrs['j'] = j
                dset.attrs['phase'] = phase
                dset.attrs['is_inv'] = False
                dset.attrs['is_ff'] = False
                dset.attrs['is_aux'] = True

            for i,j in all_neighbors:
                dset = hdf_file.create_dataset(f'eqs/eqs_{phase}_{i}_{j}',data=eqs_dict[phase,i,j])
                dset.attrs['i'] = i
                dset.attrs['j'] = j
                dset.attrs['phase'] = phase

    
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

def read_out(fname):
    aux_dict = dict()
    with h5py.File(fname) as hdf_file:
        if 'aux' in hdf_file:
            for dset_name in hdf_file['aux']:
                dset = hdf_file['aux'][dset_name]
                phase = dset.attrs['phase']
                i = dset.attrs['i']
                j = dset.attrs['j']
                aux_dict[phase,i,j] = dset[:]                
    return aux_dict

def store_new_results(aux_dict):
    with h5py.File(f'tmp/results/aux_{phase}_{repetition}.hdf','a') as hdf_file:
        for _,i,j in aux_dict:
            dset = hdf_file.create_dataset(f'aux_{phase}_{i}_{j}',data=aux_dict[phase,i,j])
            dset.attrs['phase'] = phase
            dset.attrs['i'] = i
            dset.attrs['j'] = j

def find_neighbors(blocks_to_invert,outer):
    all_neighbors = set()
    for i,j in blocks_to_invert:
        for i2,j2 in sources.block_definition.neighbors(i,j,outer):
            all_neighbors.add((i2,j2))
    return all_neighbors

def start_worker(fname):
    return subprocess.Popen(["python","-u","batch_worker.py",code,fname])

## Read command line args
parser = argparse.ArgumentParser()
parser.add_argument('code',help='Area code for which to run the inversion')
parser.add_argument('phase',default=0,type=int,help='Run this phase')
parser.add_argument('repetition',default=0,type=int,help='Run this repetition')
parser.add_argument('-n','-n-tasks',type=int,default=8)
parser.add_argument('-w','-workers',default=0,type=int)
parsed = vars(parser.parse_args())

code = parsed['code']
phase = parsed['phase']
repetition = parsed['repetition']
n_task = parsed['n']
n_workers = parsed['w']
sleep_time = 5

## Init

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

with open(f'tmp/master_settings.json','r') as f:
    thing = json.load(f)
    if 'corner_buffers' in thing:
        del thing['corner_buffers']

settings = oo_implementation.ESSettings(**thing)
settings.corners = corners
## END (Things which should be read in)
spacings = [10000,2500,700]
es_input = oo_implementation.ESInput(aeromag[:,0],aeromag[:,1:4],aeromag[:,4],aeromag[:,5:8],aeromag_year,err_vals)
es_problem = oo_implementation.ESProblem(settings,es_input,worldmap)

## Open the correct out_file (with the equivalent sources)
sources = es_problem._create_block_sources(spacings[phase])

eqs_dict = dict()

with h5py.File(f'tmp/results/out_{phase}_{repetition}.hdf') as hdf_file:
    for i,j in sources.block_definition:
        in_block_dip = (sources.block_row == i) & (sources.block_col == j)
        data = hdf_file[f'out_{phase}_{i}_{j}']
        eqs_dict[phase,i,j] = data[:]
## Create tasklist
infiles,outfiles = prep_aux_tasklists(phase,n_task,es_problem.auxiliary,es_problem.igrf_NED_aux,eqs_dict)
n_files = len(infiles)

## Clear input files

with h5py.File(f'tmp/results/aux_{phase}_{repetition}.hdf','w') as hdf_file:
    pass

## Compile everything into npy, because there seem to be some strange errors (off-by-one???)
aux_field = np.zeros(es_problem.auxiliary.shape[0])
aux_field_far = np.zeros(es_problem.auxiliary.shape[0])

block_aux = sources.block_definition.assign(*worldmap(es_problem.auxiliary[:,0],es_problem.auxiliary[:,1]))

## Start workers and wait for completion


workers = []
total_found_files = 0

while True:
    time.sleep(sleep_time)
    found_files = []
    for fname in outfiles:
        try:
            if os.path.isfile(fname+'.done'):
                print(f'Heureka! {fname} is done',end = " ")
                aux_dict = read_out(fname)
                print(f'Contains {len(aux_dict)} aux')
                store_new_results(aux_dict)
                found_files.append(fname)
                for phase,i,j in aux_dict:
                    in_block_aux = (block_aux[0] == i) & (block_aux[1] == j)
                    aux_field[in_block_aux] = aux_dict[phase,i,j][:,0,0] 
                    aux_field_far[in_block_aux] =  aux_dict[phase,i,j][:,1,0]
        except OSError as e:
            print(f'Errors over errors reading {fname}')
    
    for fname in found_files:
        outfiles.remove(fname)
    total_found_files += len(found_files)
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
    print(f'Found files: {total_found_files}, n_files: {n_files} ')
    if total_found_files == n_files:
        break

## Compile everything into one file
with h5py.File(f'tmp/results/aux_complete_{phase}_{repetition}.hdf','w') as hdf_file:
    dset = hdf_file.create_dataset(f'auxiliary',data=es_problem.auxiliary)
    dset = hdf_file.create_dataset(f'aux_field',data=aux_field)
    dset = hdf_file.create_dataset(f'aux_far',data=aux_field_far)
