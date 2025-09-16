import subprocess
import numpy as np
from wopy3.MCMC.transdim import create_models
from wopy3.forward import forward_calculators
from wopy3.forward.forward_calculators import tesses_to_pointmass, tesses_to_pointmass_max_size
from wopy3.MCMC.distributions import IndependentGaussianMisfit
from wopy3.utils import estimate_spatial_correlation, estimate_total_correlation
from wopy3.MCMC.distributions import CorrelatedGaussian
from wopy3.utils import LazyTuple
from wopy3.forward.forward_calculators import masspoint_calc_FTG_2
import numpy.polynomial.legendre as legendre
from wopy3.func_dump import get_pairwise_geo_distance
from wopy3.MCMC.transdim import RotoRectorModel
import collections
import scipy.interpolate as interpolate

class MineosForward:
    def __init__(self,lon_ax,dep_ax,mineos_approx,freq_sel=None,apply_filter=False,flatten=True):
        self.lon_ax = lon_ax
        self.dep_ax = dep_ax
        self.loni,self.depi = np.meshgrid(lon_ax,dep_ax)
        self.mineos_approx = mineos_approx
        if freq_sel is None:
            self.freq_sel = np.arange(mineos_approx.ref_out.shape[0],dtype=int)
        else:
            self.freq_sel = freq_sel
        self.apply_filter=apply_filter
        self.flatten = flatten
    
    def _apply_filter(self,dvph_true,freq_sel):
        F_vph = np.fft.fft(dvph_true,axis=0)
        F_vph_filt = F_vph.copy()
        l_vec = self.mineos_approx.ref_out[:,1].astype(int)
        fftfreq = np.fft.fftfreq(dvph_true.shape[0]) * dvph_true.shape[0]

        for i in range(len(freq_sel)):
            l = l_vec[freq_sel[i]]
            F_vph_filt[np.abs(fftfreq)>l,i] = 0
        return np.fft.ifft(F_vph_filt,axis=0).real

    def __call__(self,model,flatten=None,freq_sel=None,apply_filter=None):
        if freq_sel is None:
            freq_sel = self.freq_sel
        if apply_filter is None:
            apply_filter = self.apply_filter
        if flatten is None:
            flatten = self.flatten
        
        grids = model((self.loni,self.depi))
        dvph = self.mineos_approx.get(grids).T
        dvph = dvph[:,freq_sel]

        if apply_filter:
            dvph = self._apply_filter(dvph,freq_sel)

        if flatten:
            return dvph.flatten()
        else:
            return dvph

class TraveltimeForward:
    def __init__(self,wrapped,stations):
        """Turn phase velocities into phase traveltime delays at select stations
        Parameters
        ----------
        wrapped : MineosForward
        stations : np.array
        """
        self.wrapped = wrapped
        self.stations = stations
    
    def __call__(self,model,flatten=True):
        dvph = self.wrapped(model,flatten=False)
        traveltime = dvph.cumsum(0)[self.stations,:]
        
        dtt = np.zeros(traveltime.shape)
        dtt[0] = -traveltime[0]
        dtt[1:] = -np.diff(traveltime,axis=0)
        if flatten:
            return dtt.flatten()   
        else:
            return dtt

class GravityForward:
    def __init__(self,lon_ax,dep_ax,interpolated_reference_model,use_grav=True,use_pot=False,use_grad=False,
                lons=None,height=225e3,use_col=False,flatten=False,dy=None):
        self.lon_ax = lon_ax
        self.dep_ax = dep_ax
        self.loni,self.depi = np.meshgrid(lon_ax,dep_ax)
        self.interpolated_reference_model = interpolated_reference_model
        if lons is None:
            lons = lon_ax
        self.lons = lons
        self.height = height
        self.use_col = use_col
        self.flatten = flatten
        # Calculate gravity sensitivity matrix
        n_lon = lon_ax.size
        ones = np.ones(n_lon)
        zeros = np.zeros(n_lon)
        self.use_grav = use_grav
        self.use_pot = use_pot
        self.use_grad = use_grad
        self.A = np.zeros((len(lons),len(lon_ax),len(dep_ax)))
        self.B = np.zeros((len(lons),len(lon_ax),len(dep_ax)))
        self.C = np.zeros((len(lons),len(lon_ax),len(dep_ax)))
        dx = lon_ax[1] - lon_ax[0]
        if dy is None:
            dy = dx
        dr = dep_ax[1] - dep_ax[0]
        
        for i in range(len(dep_ax)):
            r_top = 6371 - dep_ax[i] + 0.5 * dr
            r_bot = 6371 - dep_ax[i] - 0.5 * dr
            surface_Area = dx*np.pi/(180.0) * np.cos(0.0) * 2 * np.sin(dy/360.0*np.pi)
            mass = surface_Area * (r_top**3-r_bot**3)/3.0 * 1e9
            self.A[:,:,i] = masspoint_calc_FTG_2(lon_ax,zeros,dep_ax[i]*ones,mass*ones,lons,zeros,ones*height,calc_mode="grav")[:,:,0]
            self.B[:,:,i] = masspoint_calc_FTG_2(lon_ax,zeros,dep_ax[i]*ones,mass*ones,lons,zeros,ones*height,calc_mode="potential")[:,:,0]
            self.C[:,:,i] = masspoint_calc_FTG_2(lon_ax,zeros,dep_ax[i]*ones,mass*ones,lons,zeros,ones*height,calc_mode="gradient")[:,:,5]
            
    
    def __call__(self,model,use_col=None,flatten=None):
        if use_col is None:
            use_col = self.use_col
        if flatten is None:
            flatten = self.flatten
        returnor = []
        grids = model((self.loni,self.depi))
        rho_model = self.interpolated_reference_model[:,1][:,None] * grids[:,:,1] / 100.0
        if self.use_grav:
            returnor.append(np.einsum('ijk,kj->i',self.A,rho_model))
        if self.use_pot:
            returnor.append(np.einsum('ijk,kj->i',self.B,rho_model))
        if self.use_grad:
            returnor.append(np.einsum('ijk,kj->i',self.C,rho_model))
        
        if flatten:
            return np.array(returnor).T.flatten()
        elif use_col:
            return np.array(returnor).T
        else:   
            return returnor


class CompoundForward:
    def __init__(self,calculators):
        self.calculators = calculators
    
    def forward(self,model):
        return LazyTuple([c(model) for c in self.calculators])

    def __call__(self,model):
        return self.forward(model)

class CompoundMisfit:
    def __init__(self,calculators,sigma=None,hy_indices=None,use_cols=False):
        self.internal_misfit = calculators
        self.sigma = sigma
        self.use_cols = use_cols
        if hy_indices is None:
            self.hy_indices = np.arange(len(calculators),dtype=int)
        else:
            self.hy_indices = hy_indices
        
    def __call__(self,y,sigma=None):
        if sigma is None:
            sigma = self.sigma
        if self.use_cols:
            return np.sum([self.internal_misfit[i](y[:,i],sigma[self.hy_indices[i]]) for i in range(len(self.internal_misfit))])
        else:
            return np.sum([self.internal_misfit[i](y[i],sigma[self.hy_indices[i]]) for i in range(len(self.internal_misfit))])

class GravityMisfit:
    """Misfit function for pointwise error correlation
    """
    def __init__(self,data,sigma,correlation_matrix):
        self.data = np.array(data)
        self.sigma = sigma
        self.correlation_matrix = correlation_matrix
        self.correlation_matrix_inv = np.linalg.inv(correlation_matrix)
    
    def __call__(self,y,sigma=None):
        if sigma is None:
            sigma = self.sigma
        sigma_mat_inv = np.linalg.inv(self.correlation_matrix[:len(y),:len(y)] * (np.outer(sigma,sigma)))
        y = np.array(y)
        
        likelihood = -0.5 * np.einsum('ik,ij,jk',y-self.data,sigma_mat_inv,y-self.data)
        likelihood = likelihood + 0.5 * np.linalg.slogdet(sigma_mat_inv)[1] * self.data.shape[1]
        return likelihood

def generate_model_ensemble(proposal,model_complexity,number_of_models,loni_km,depi):
    """Use a Transdim linear proposal to generate an ensemble of models
    """
    return create_models(RotoRectorModel.create_from_vectors,proposal,model_complexity,number_of_models,loni_km,depi)

def generate_grav_misfit_obj(grav_true,gravity_error_mode,lmax=137,height=250e3,r0=6371e3,grav_forward=None,
                        numerical_depth_ix=0,n_ensemble=1000,proposal=None):
    
    lon_ax = grav_forward.lon_ax
    dep_ax = grav_forward.dep_ax   

    if gravity_error_mode == 'componentwise-global-relative':
        # The misfits are actually uncorrelated, it's just that CorrelatedGaussian can rescale a fixed value
        # Whereas you would need to give the actual value for IndependentGaussianMisfit
        misfits = [CorrelatedGaussian(grav_true[:,i],sigma=np.eye(len(lon_ax))*grav_true[:,i].var()) for i in range(grav_true.shape[1])]
        grav_misfit = CompoundMisfit(misfits,hy_indices=np.zeros(grav_true.shape[1],dtype=int))
    
    elif gravity_error_mode == "white-noise":
        misfit_list = []
        for i in range(grav_true.shape[1]):
            misfit_list.append(CorrelatedGaussian(grav_true[:,i],np.eye(len(lon_ax)),allow_singular=True))
            # Shouldnt this be independent gaussian misfit?
        grav_misfit = CompoundMisfit(misfit_list)

    else:
        raise ValueError('missing gravity_error_mode')
    return grav_misfit

def generate_dvph_misfit_object(error_mode,dvph_true,mineos_forward=None,proposal=None,n_models=1000):
    lon_ax = mineos_forward.lon_ax
    dep_ax = mineos_forward.dep_ax
    freq_sel = mineos_forward.freq_sel
    flatten = True # Whether the mineos_forward should return a flaattened Array
    
    if error_mode == 'uncorrelated-individual':
        dvph_misfit = CompoundMisfit([IndependentGaussianMisfit(dvph_true[:,i]) for i in range(dvph_true.shape[1])],use_cols=True,
                                   hy_indices=[[i] for i in range(dvph_true.shape[1])])
        flatten = False
    
    elif error_mode == 'uncorrelated-global':
        dvph_misfit = IndependentGaussianMisfit(dvph_true.flatten())
    
    elif error_mode == 'uncorrelated-global-relative':
        flatten = False
        # The misfits are actually uncorrelated, it's just that CorrelatedGaussian can rescale a fixed value
        # Whereas you would need to give the actual value for IndependentGaussianMisfit
        misfits = [CorrelatedGaussian(dvph_true[:,i],sigma=np.eye(len(lon_ax))*dvph_true[:,i].var()) for i in range(dvph_true.shape[1])]
        dvph_misfit = CompoundMisfit(misfits,hy_indices=np.zeros(dvph_true.shape[1],dtype=int),use_cols=True)    
    else:
        raise ValueError('Incorrect error mode in generate_dvph_misfit_object')
    
    return dvph_misfit,flatten

