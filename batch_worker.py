import numpy as np
import oo_implementation
from mpl_toolkits.basemap import Basemap
import json
import glob
import os
import sys
import time
import argparse
import random
import datetime
import socket
import pickle

import importlib
importlib.reload(oo_implementation)

import h5py


def read_all(fname):
    data_dict = dict()
    eqs_dict = dict()
    aux_dict = dict()
    to_invert = list()
    to_ff = list()
    to_aux = list()
    with h5py.File(fname) as hdf_file:
        out_name = hdf_file.attrs['out_name']
        for dset_name in hdf_file['aeromag']:
            dset = hdf_file['aeromag'][dset_name]
            phase = dset.attrs['phase']
            i = dset.attrs['i']
            j = dset.attrs['j']
            is_inv = dset.attrs['is_inv']
            data_dict[phase,i,j] = dset[:]
            if is_inv:
                to_invert.append((i,j))
            elif dset.attrs['is_ff']:
                to_ff.append((i,j))
        for dset_name in hdf_file['eqs']:
            dset = hdf_file['eqs'][dset_name]
            phase = dset.attrs['phase']
            i = dset.attrs['i']
            j = dset.attrs['j']
            eqs_dict[phase,i,j] = dset[:]
        for dset_name in hdf_file['aux']:
            dset = hdf_file['aux'][dset_name]
            phase = dset.attrs['phase']
            i = dset.attrs['i']
            j = dset.attrs['j']
            aux_dict[phase,i,j] = dset[:]
            if dset.attrs['is_aux']:
                to_aux.append((i,j))            
            
    return phase,data_dict,eqs_dict,aux_dict,to_invert,to_ff,to_aux,out_name

def run_inversion(data_dict,phase,i,j):
    # read in the necessary data
    data = []
    for i2,j2 in sources.block_definition.neighbors(i,j,1):
        data.append(data_dict[phase,i2,j2])
    data = np.vstack(data)
    sub_input = oo_implementation.ESInput(data[:,0],data[:,1:4],data[:,4],data[:,5:8],data[:,9],data[:,10])
    sub_problem = oo_implementation.ESProblem(settings,sub_input,worldmap)
    block_runner = oo_implementation.ESBlockRunner(sub_input,sources,settings,sub_problem.stations_epochs,worldmap)
    in_block_stat = (block_runner.block_stat[0] == i) & (block_runner.block_stat[1] == j)
    in_block_dip = (block_runner.block_dip[0] == i) & (block_runner.block_dip[1] == j)

    fill_ratio = in_block_stat.sum() / in_block_dip.sum()
    try:
        temp,in_center = block_runner.solve_block_extend(i,j)
    except np.linalg.LinAlgError as e:
        print(f'inversion failed {i} {j}')
        in_big_block_dip = (np.abs(block_runner.block_dip[0] - i ) <= 1) & (np.abs(block_runner.block_dip[1] - j ) <= 1)
        in_center = (block_runner.block_dip[0][in_big_block_dip] == i) & (block_runner.block_dip[1][in_big_block_dip] == j)
        temp = np.zeros(in_big_block_dip.sum())

    ## Blank out any values, if the fill_threshold is not met
    if fill_ratio < settings.fill_threshold:
        temp[:] = 0.0
    return temp[in_center]

def run_recalc(data_dict,eqs_dict,phase,i,j):
    equivalent_sources = np.zeros(sources.dipoles.shape[0])
    data = data_dict[phase,i,j]
    sub_input = oo_implementation.ESInput(data[:,0],data[:,1:4],data[:,4],data[:,5:8],data[:,9],data[:,10])
    sub_problem = oo_implementation.ESProblem(settings,sub_input,worldmap)
    block_runner = oo_implementation.ESBlockRunner(sub_input,sources,settings,sub_problem.stations_epochs,worldmap)
    print('loading input ',end=" ")
    for i2,j2 in sources.block_definition.neighbors(i,j,settings.far_field_outer):
        temp = eqs_dict[phase,i2,j2]
        in_block_dip = (np.abs(block_runner.block_dip[0] - i2 ) <= 0) & (np.abs(block_runner.block_dip[1] - j2 ) <= 0)
        equivalent_sources[in_block_dip] = temp
    print('loading done')
    inner = block_runner.calculate_far_field(i,j,equivalent_sources,outer=1,inner=0)
    farfield = block_runner.calculate_far_field(i,j,equivalent_sources,settings.far_field_outer,inner=2)
    return np.stack((inner[:,None],farfield[:,None]),axis=1)

def run_aux(aux_dict,eqs_dict,phase,i,j):
    equivalent_sources = np.zeros(sources.dipoles.shape[0])
    auxiliary = aux_dict[phase,i,j]
    n = auxiliary.shape[0]
    sub_input = oo_implementation.ESInput(np.zeros(n),auxiliary[:,:3],np.zeros(n),np.zeros((n,3)),np.ones(n)*settings.aux_year,np.zeros(n))
    sub_input.igrf_NED_stat = auxiliary[:,:3]
    sub_problem = oo_implementation.ESProblem(settings,sub_input,worldmap)
    block_runner = oo_implementation.ESBlockRunner(sub_input,sources,settings,sub_problem.stations_epochs,worldmap)
    print('loading input ',end=" ")
    for i2,j2 in sources.block_definition.neighbors(i,j,settings.far_field_outer):
        temp = eqs_dict[phase,i2,j2]
        in_block_dip = (np.abs(block_runner.block_dip[0] - i2 ) <= 0) & (np.abs(block_runner.block_dip[1] - j2 ) <= 0)
        equivalent_sources[in_block_dip] = temp
    print('loading done')
    inner = block_runner.calculate_far_field(i,j,equivalent_sources,outer=1,inner=0)
    farfield = block_runner.calculate_far_field(i,j,equivalent_sources,settings.far_field_outer,inner=2)
    return np.stack((inner[:,None],farfield[:,None]),axis=1)


def init_settings(phase):
    """Load settings from json file
    """
    with open(f'{tmp_dir}/settings_{phase}.json','r') as f:
        thing = json.load(f)
        if 'corner_buffers' in thing:
            del thing['corner_buffers']
    settings = oo_implementation.ESSettings(**thing)
    return settings

def get_strftime():
    return  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

t0_timestamp = get_strftime()

parser = argparse.ArgumentParser()
parser.add_argument('code')
parser.add_argument('tasklist')

parser.add_argument('--data-dir',default='data')
parser.add_argument('--tmp-dir',default='tmp')


parsed = vars(parser.parse_args())

code = parsed['code']
data_dir = parsed['data_dir']
tmp_dir = parsed['tmp_dir']
tasklist = parsed['tasklist']

with open(f'{data_dir}/{code}.json') as f:
    config_dict = json.load(f)

## Initialize projection

used_projection = config_dict['used_projection']
EW_area_f = config_dict['EW_area_f']
NS_area_f = config_dict['NS_area_f']
center_lat_f = config_dict['center_lat_f']
center_long_f = config_dict['center_long_f']
worldmap = Basemap(projection=used_projection,width=EW_area_f,height=NS_area_f,lat_0=center_lat_f,
                   lon_0=center_long_f,resolution="l")
corners = worldmap.llcrnrx,worldmap.urcrnrx,worldmap.llcrnry,worldmap.urcrnry
corners_shifted = worldmap.llcrnrx-1,worldmap.urcrnrx+1,worldmap.llcrnry-1,worldmap.urcrnry+1

## Read in data

aeromag = np.load(f'{data_dir}/{code}_aeromag.npy')
aeromag_year = np.load(f'{data_dir}/{code}_aeromag_year.npy')
err_vals = np.load(f'{data_dir}/{code}_err_vals.npy')

year_DGRF = np.array([2015, 2010, 2005, 2000, 1995, 1990, 1985, 1980, 1975, 1970]) 
es_input = oo_implementation.ESInput(aeromag[:,0],aeromag[:,1:4],aeromag[:,4],aeromag[:,5:8],aeromag_year,err_vals)

spacings = [10000,2500,700]

phase,data_dict,eqs_dict,aux_dict,to_inv,to_ff,to_aux,out_name = read_all(tasklist)
spacing = spacings[phase]

settings = init_settings(phase)

es_problem = oo_implementation.ESProblem(settings,es_input,worldmap)

with open(f'tmp/sources_{code}_{phase}.pkl','rb') as f:
    sources = pickle.load(f)

out_dict = dict()

for i,j in to_inv:
    print(f'Starting inversion {phase} {i} {j} ...',end=" ")
    results = run_inversion(data_dict,phase,i,j)
    out_dict[i,j] = results
    print('done.')

pred_dict = dict()

for i,j in to_ff:
    print(f'Starting Farfield {phase} {i} {j} ...',end=" ")
    results = run_recalc(data_dict,eqs_dict,phase,i,j)
    pred_dict[i,j] = results
    print('done.')

aux_dict_out = dict()

for i,j in to_aux:
    print(f'Starting aux calc {phase} {i} {j} ...',end=" ")
    results = run_aux(aux_dict,eqs_dict,phase,i,j)
    aux_dict_out[i,j] = results
    print('done.')

with h5py.File(out_name,'w') as f:
    grp = f.create_group('eqs')
    for i,j in to_inv:
        dset = grp.create_dataset(f'out_{phase}_{i}_{j}',data=out_dict[i,j])
        dset.attrs['phase'] = phase
        dset.attrs['i'] = i
        dset.attrs['j'] = j
    grp = f.create_group('pred')
    for i,j in to_ff:
        dset = grp.create_dataset(f'pred_{phase}_{i}_{j}',data=pred_dict[i,j])
        dset.attrs['phase'] = phase
        dset.attrs['i'] = i
        dset.attrs['j'] = j
    grp = f.create_group('aux')
    for i,j in to_aux:
        dset = grp.create_dataset(f'aux_{phase}_{i}_{j}',data=aux_dict_out[i,j])
        dset.attrs['phase'] = phase
        dset.attrs['i'] = i
        dset.attrs['j'] = j

## Write signal file that we are done
with open(out_name+'.tmp','w') as f:
    f.write(t0_timestamp+'\n')
    f.write(get_strftime())

os.rename(out_name+'.tmp',out_name+'.done')