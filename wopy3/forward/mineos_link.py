import numpy as np
import string
import subprocess
from multiprocessing.pool import ThreadPool
import os
import scipy.interpolate
import random

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def load_mineos(fname):
    with open(fname) as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if 'root precision' in line:
            start_line = i
            break
    return np.loadtxt(fname,skiprows=start_line+4,usecols=(0,2,3,4,5,6,7,8))

def run_mineos_on_file(fname=None,mode=3,overtone_min_max=(0,0),l_min_max=(0,500),keep_temp_files=False,eps=1.0e-7,wgrav=1000.0):
    code = get_random_string(8)
    control_file = code + '_control.txt'
    out_file = code + '_out.txt'
    bin_file = code +'.bin'
    with open(control_file,'w') as f:
        f.write('%s\n'%fname)
        f.write('%s\n'%out_file)
        f.write('%s\n'%bin_file)
        f.write('%e %f\n'%(eps,wgrav))
        f.write('%d\n'%mode)
        f.write('%d %d 0 1000.0 %d %d\n' % (*l_min_max,*overtone_min_max))
    with open(control_file) as f:
        sub = subprocess.Popen('minos_bran',stdin=f)
        sub.communicate()
    results = load_mineos(out_file)
    if not keep_temp_files:
        try:
            os.remove(control_file)
            os.remove(out_file)
            os.remove(bin_file)
        except OSError as err:
            print(err)
            print('run_mineos_on_file: Could not remove temporary files')
    return results

def write_model_to_file(fname,table,nic=33,noc=66):
    assert table.shape[0]<350
    np.savetxt(fname,table,header='bla\n1 1 1\n%d %d %d' % (table.shape[0],nic,noc),comments='',fmt='%.2f')

def perturb_prem(prem,depths,dvs=None,dvp=None,drho=None):
    radius = 6371000.0 - depths
    if dvp is None:
        dvp = np.zeros(depths.shape)
    if dvs is None:
        dvs = np.zeros(depths.shape)
    if drho is None:
        drho = np.zeros(depths.shape)
    if not (radius[1]>radius[0]):
        radius = np.flip(radius)
        dvs = np.flipud(dvs)
        drho = np.flipud(drho)
        dvp = np.flipud(dvp)
    
    prem_ix_above = (prem[:,0] > np.max(radius)) 
    prem_ix_below = (prem[:,0] < np.min(radius))
    prem_ix_intern = ~prem_ix_above & ~prem_ix_below
    N_prem_below = prem_ix_below.sum()
    if N_prem_below < 65:
        raise ValueError('Model is cutting into outer core -> Invalid!')
    elif N_prem_below == 65:
        # The lower boundary is right at the CMB, make sure the outer core still has the right parameters
        prem_ix_below[65] = True
        N_prem_below = prem_ix_below.sum()
    N_prem_above = prem_ix_above.sum()
    N_prem_outside = prem_ix_above.sum() +  prem_ix_below.sum()
    N_prem_intern = len(prem) - N_prem_outside
    # Combine internal prem nodes with model nodes
    combined_nodes = np.union1d(radius,prem[prem_ix_intern,0])
    i1 = N_prem_below
    i2 = N_prem_below + len(combined_nodes)
        
    # Interpolate prem and dvs to combined nodes
    prem_earth = scipy.interpolate.interp1d(prem[:,0],prem.T)(combined_nodes).T
    dvs_earth = scipy.interpolate.interp1d(radius,dvs)(combined_nodes)
    dvp_earth = scipy.interpolate.interp1d(radius,dvp)(combined_nodes)
    drho_earth = scipy.interpolate.interp1d(radius,drho)(combined_nodes)
    prem_earth[:,1] *= (1+drho_earth/100)
    prem_earth[:,2] *= (1+dvp_earth/100)
    prem_earth[:,3] *= (1+dvs_earth/100)
    prem_earth[:,6] *= (1+dvp_earth/100)
    prem_earth[:,7] *= (1+dvs_earth/100)
    
    final_model = np.zeros((len(combined_nodes)+N_prem_outside,prem_earth.shape[1]))
    final_model[:i1,:] = prem[:i1,:]
    final_model[i1:i2,:] = prem_earth
    final_model[i2:,:] = prem[prem_ix_above,:]
    return combined_nodes,prem_earth,final_model

def perturb_prem_nouveau(prem,depths,dvs=None,dvp=None,drho=None,core_radius=3.48e6):
    """
    notebook:: A1
    """
    radius = 6371000.0 - depths
    if dvp is None:
        dvp = np.zeros(depths.shape)
    if dvs is None:
        dvs = np.zeros(depths.shape)
    if drho is None:
        drho = np.zeros(depths.shape)
    if not (radius[1]>radius[0]):
        radius = np.flip(radius)
        dvs = np.flipud(dvs)
        drho = np.flipud(drho)
        dvp = np.flipud(dvp)
    
    N_prem = prem.shape[0]

    combined_nodes = np.zeros(len(radius)+N_prem)    
    combined_nodes[:N_prem] = prem[:,0]
    combined_nodes[N_prem:] = radius
    is_prem = np.zeros(len(radius)+N_prem,dtype=bool)
    is_prem[:N_prem] = True
    ix = np.argsort(combined_nodes)
    combined_nodes = combined_nodes[ix]
    is_prem = is_prem[ix]
    
    # Interpolate prem and dvs to combined nodes, but keep original prem values to protect the jumps
    prem_earth = scipy.interpolate.interp1d(prem[:,0],prem.T)(combined_nodes).T
    prem_earth[is_prem] = prem
    dvs_earth = scipy.interpolate.interp1d(radius,dvs,fill_value=0,bounds_error=False)(combined_nodes)
    dvp_earth = scipy.interpolate.interp1d(radius,dvp,fill_value=0,bounds_error=False)(combined_nodes)
    drho_earth = scipy.interpolate.interp1d(radius,drho,fill_value=0,bounds_error=False)(combined_nodes)
    
    in_core = combined_nodes<=core_radius
    dvs_earth[in_core] = 0
    dvp_earth[in_core] = 0
    drho_earth[in_core] = 0
    
    prem_earth[:,1] *= (1+drho_earth/100)
    prem_earth[:,2] *= (1+dvp_earth/100)
    prem_earth[:,3] *= (1+dvs_earth/100)
    prem_earth[:,6] *= (1+dvp_earth/100)
    prem_earth[:,7] *= (1+dvs_earth/100)
    
    return combined_nodes,prem_earth,prem_earth

def interp_prem(prem,depths):
    radius = 6371000.0 - depths
    prem_earth = scipy.interpolate.interp1d(prem[:,0],prem.T)(radius).T
    return prem_earth

def work(fname,**kwargs):
    return run_mineos_on_file(fname,**kwargs)

def run_mineos_on_cols(dep_ax,grids,reference_model,n_workers=8,mineos_kwd={}):
    pool = ThreadPool(n_workers)
    async_obj = dict()
    for col in range(grids.shape[1]):
        fname = 'test_%d.txt'%col
        schnork,oerk,pollock = perturb_prem(reference_model,1000*dep_ax,dvs=grids[:,:,0][:,col],drho=grids[:,:,1][:,col])
        write_model_to_file(fname,pollock)
        async_obj[col] = pool.apply_async(work,(fname,),mineos_kwd)
    pool.close()
    pool.join()
    results = np.array([async_obj[col].get()[:,2] for col in range(grids.shape[1])])
    
    # Clean up
    try:
        for col in range(grids.shape[1]):
            os.remove('test_%d.txt'%col)
    except OSError:
        print('Couldnt remove input files')
    return results

class MineosApproximator:
    def __init__(self,depths,reference_model,amplitudes,n_workers=1,mineos_kwd={},perturb_mode='old'):
        self.depths = depths
        self.reference_model = reference_model
        self.amplitudes = amplitudes
        self.perturb_mode = perturb_mode
        if perturb_mode is 'old':
            self._perturb_prem = perturb_prem
        elif perturb_mode is 'nouveau':
            self._perturb_prem = perturb_prem_nouveau
        else:
            raise ValueError('Incorrect Perturb PREM mode in MineosApproximator')
        # Reference model calculation
        schnork,oerk,pollock = self._perturb_prem(reference_model,1000*depths)
        write_model_to_file('test.txt',pollock)
        self.ref_out = run_mineos_on_file('test.txt',**mineos_kwd)
        n_freq = self.ref_out.shape[0]
        # Parallel perturbation matrix calculation (Phase velocity)
        self.dvs_perturb = np.zeros((len(depths),len(amplitudes),n_freq))
        self.dvp_perturb = np.zeros((len(depths),len(amplitudes),n_freq))
        self.rho_perturb = np.zeros((len(depths),len(amplitudes),n_freq))
        counter = 0
        pool = ThreadPool(n_workers)
        async_obj = dict()        
        for i in range(len(depths)):
            for j in range(len(amplitudes)):
                column = np.zeros(len(depths))
                column[i] = amplitudes[j]
                schnork,oerk,pollock = self._perturb_prem(reference_model,1000*depths,dvs=column)
                fname = 'lin_%d.txt'%counter
                write_model_to_file(fname,pollock)
                async_obj[i,j] = pool.apply_async(work,(fname,),mineos_kwd)
                counter = counter + 1
        
        pool.close()
        pool.join()
        
        counter = 0
        for i in range(len(depths)):
            for j in range(len(amplitudes)):
                temp = async_obj[i,j].get()
                if temp.shape[0] == n_freq:
                    self.dvs_perturb[i,j,:] = (temp[:,2] - self.ref_out[:,2])/self.ref_out[:,2]
                else:
                    ix = temp[:,1].astype(int)-2
                    sel = ix<n_freq
                    ix = ix[sel]
                    self.dvs_perturb[i,j,ix] = (temp[sel,2] - self.ref_out[ix,2])/self.ref_out[ix,2]
                try:
                    os.remove('lin_%d.txt'%counter)
                except OSError:
                    print('MineosApproximator: Could not remove input files')
                counter = counter + 1
        
        # dvp_perturbation
        counter = 0
        pool = ThreadPool(n_workers)
        async_obj = dict()        
        for i in range(len(depths)):
            for j in range(len(amplitudes)):
                column = np.zeros(len(depths))
                column[i] = amplitudes[j]
                schnork,oerk,pollock = self._perturb_prem(reference_model,1000*depths,dvp=column)
                fname = 'lin_%d.txt'%counter
                write_model_to_file(fname,pollock)
                async_obj[i,j] = pool.apply_async(work,(fname,),mineos_kwd)
                counter = counter + 1
        
        pool.close()
        pool.join()
        
        counter = 0
        for i in range(len(depths)):
            for j in range(len(amplitudes)):
                temp = async_obj[i,j].get()
                if temp.shape[0] == n_freq:
                    self.dvp_perturb[i,j,:] = (temp[:,2] - self.ref_out[:,2])/self.ref_out[:,2]
                else:
                    ix = temp[:,1].astype(int)-2
                    sel = ix<n_freq
                    ix = ix[sel]
                    self.dvp_perturb[i,j,ix] = (temp[sel,2] - self.ref_out[ix,2])/self.ref_out[ix,2]
                try:
                    os.remove('lin_%d.txt'%counter)
                except OSError:
                    print('MineosApproximator: Could not remove input files')
                counter = counter + 1
        
        # rho_perturb
        counter = 0
        pool = ThreadPool(n_workers)
        async_obj = dict()        
        for i in range(len(depths)):
            for j in range(len(amplitudes)):
                column = np.zeros(len(depths))
                column[i] = amplitudes[j]
                schnork,oerk,pollock = self._perturb_prem(reference_model,1000*depths,drho=column)
                fname = 'lin_%d.txt'%counter
                write_model_to_file(fname,pollock)
                async_obj[i,j] = pool.apply_async(work,(fname,),mineos_kwd)
                counter = counter + 1
        
        pool.close()
        pool.join()
        counter = 0
        for i in range(len(depths)):
            for j in range(len(amplitudes)):
                temp = async_obj[i,j].get()
                if temp.shape[0] == n_freq:
                    self.rho_perturb[i,j,:] = (temp[:,2] - self.ref_out[:,2])/self.ref_out[:,2]
                else:
                    ix = temp[:,1].astype(int)-2
                    sel = ix<n_freq
                    ix = ix[sel]
                    self.rho_perturb[i,j,ix] = (temp[sel,2] - self.ref_out[ix,2])/self.ref_out[ix,2]
                try:
                    os.remove('lin_%d.txt'%counter)
                except OSError:
                    print('MineosApproximator: Could not remove input files')
                counter = counter + 1
        
        self.calc_polys()
    
    def calc_polys(self):
        n_freq = self.dvs_perturb.shape[2]
        # Calculate approximation polynomials
        self.polynomials_dvs = np.zeros((len(self.depths),n_freq,3))
        self.polynomials_dvp = np.zeros((len(self.depths),n_freq,3))
        self.polynomials_rho = np.zeros((len(self.depths),n_freq,3))
        for i in range(self.dvs_perturb.shape[0]):
            for j in range(self.dvs_perturb.shape[2]):
                x = self.amplitudes
                y1 = self.dvs_perturb[i,:,j]
                y2 = self.rho_perturb[i,:,j]
                y3 = self.dvp_perturb[i,:,j]
                self.polynomials_dvs[i,j] = np.polyfit(x,y1,2)
                self.polynomials_dvp[i,j] = np.polyfit(x,y3,2)
                self.polynomials_rho[i,j] = np.polyfit(x,y2,2)

    def get(self,grids):
        dvs = grids[:,:,0]
        rho = grids[:,:,1]
        rho_contrib = self.polynomials_rho[:,:,0].T.dot(rho**2) +  self.polynomials_rho[:,:,1].T.dot(rho) + self.polynomials_rho[:,:,2].T.sum(1)[:,None]
        vs_contrib = self.polynomials_dvs[:,:,0].T.dot(dvs**2) +  self.polynomials_dvs[:,:,1].T.dot(dvs) + self.polynomials_dvs[:,:,2].T.sum(1)[:,None]
        return rho_contrib + vs_contrib
        