import collections
import subprocess
import disba
import numpy as np
import scipy.spatial as spatial
import scipy.interpolate as interpolate
import wopy3.legendrestuff as legendrestuff
import wopy3.forward.cartesian as cartesian
from wopy3.forward.forward_calculators import masspoint_calc_FTG_2, memory_save_masspoint_calc_FTG_2, tesses_to_pointmass, tesses_to_pointmass_max_size
import wopy3.MCMC.blobography as blobography
import wopy3.MCMC.transdim as transdim

def NS_to_theta(N,S):
    return (90-N)/180.0*np.pi,(90-S)/180.0*np.pi

class TesseroidModel:
    def __init__(self,points,props,loni=None,lati=None,ri=None,transformer=None):
        self.points = points
        self.props = props
        self.loni = loni
        self.lati = lati
        self.ri = ri
        if transformer is None:
            self.transformer = lambda x:x
        else:
            self.transformer = transformer
    
    @property
    def k(self):
        return self.points.shape[0]
    
    @property
    def sizes(self):
        return self.props[:,:3]
    
    @property
    def values(self):
        return self.transformer(self.props[:,3:])
    
    @property
    def n_values(self):
        return self.values.shape[1]
    
    @property
    def spatial_dim(self):
        return 3
    
    def get_coordinates(self,x):
        if x is None:
            loni = self.loni
            lati = self.lati
            ri = self.ri
        else:
            loni,lati,ri = x
        return loni,lati,ri
    
    def calc_mask(self,i,x=None):
        loni,lati,ri = self.get_coordinates(x)
        if loni.ndim == 3:
            mask_lon = np.abs(loni[0,:,0] - self.points[i,0]) <= self.sizes[i,0]/2
            mask_lon = mask_lon |(np.abs(loni[0,:,0] - self.points[i,0] + 360) <= self.sizes[i,0]/2)
            mask_lon = mask_lon |(np.abs(loni[0,:,0] - self.points[i,0] - 360) <= self.sizes[i,0]/2)

            mask_lat = np.abs(lati[:,0,0] - self.points[i,1]) <= self.sizes[i,1]/2
            mask_r = np.abs(ri[0,0,:] - self.points[i,2]) <= self.sizes[i,2]/2
            return mask_lon[None,:,None] & mask_lat[:,None,None] & mask_r[None,None,:]
        else:
            mask_lon = np.abs(loni - self.points[i,0]) <= self.sizes[i,0]/2
            mask_lat = np.abs(lati - self.points[i,1]) <= self.sizes[i,1]/2
            mask_r = np.abs(ri - self.points[i,2]) <= self.sizes[i,2]/2
            return mask_lon & mask_lat & mask_r            
           
    def calc_grid(self,x=None):
        loni,_,_ = self.get_coordinates(x)
        grid = np.zeros(loni.shape+(self.n_values,))
        for i in range(self.k):
            mask = self.calc_mask(i,x)
            grid[mask] += self.values[i]
        return grid
    
    def birth(self,new_point,new_prop):
        new_points = np.vstack((self.points,new_point))
        new_props = np.vstack((self.props,new_prop))
        offspring = TesseroidModel(new_points,new_props,self.loni,self.lati,self.ri,transformer=self.transformer)
        offspring.grid = self.grid.copy()
        offspring.grid += offspring.calc_mask(-1)[...,None] * offspring.values[-1]
        return offspring
    
    def death(self,ksel):
        new_points = np.delete(self.points,ksel,axis=0)
        new_props = np.delete(self.props,ksel,axis=0)
        offspring = TesseroidModel(new_points,new_props,self.loni,self.lati,self.ri,transformer=self.transformer)
        offspring.grid = self.grid.copy()
        offspring.grid -= self.calc_mask(ksel)[...,None] * self.values[ksel]
        return offspring
    
    def change_param(self,ksel,new_props):
        offspring = TesseroidModel(self.points,new_props,self.loni,self.lati,self.ri,transformer=self.transformer)
        offspring.grid = self.grid.copy()
        offspring.grid += offspring.calc_mask(ksel)[...,None] * offspring.values[ksel] - self.calc_mask(ksel)[...,None] * self.values[ksel] 
        return offspring
    
    def move(self,ksel,new_points):
        offspring = TesseroidModel(new_points,self.props,self.loni,self.lati,self.ri,transformer=self.transformer)
        offspring.grid = self.grid.copy()
        offspring.grid += offspring.calc_mask(ksel)[...,None] * offspring.values[ksel] - self.calc_mask(ksel)[...,None] * self.values[ksel] 
        return offspring        
    
    def calc_2D_mask(self,i,x=None):
        loni,lati,_ = self.get_coordinates(x)
        mask_lon = np.abs(loni[...,0] - self.points[i,0]) <= self.sizes[i,0]/2
        mask_lat = np.abs(lati[...,0] - self.points[i,1]) <= self.sizes[i,1]/2
        mask = (mask_lon & mask_lat)
        return mask
    
    def get_2D_nonzero_mask(self,x=None):
        loni,lati,_ = self.get_coordinates(x)
        mask = np.zeros(loni.shape[:2],dtype=bool)
        for i in range(self.k):
            mask_lon = np.abs(loni[...,0] - self.points[i,0]) <= self.sizes[i,0]/2
            mask_lat = np.abs(lati[...,0] - self.points[i,1]) <= self.sizes[i,1]/2
            mask += (mask_lon & mask_lat)
        return mask
    
    def __call__(self,x=None):
        # This looks very sketchy and is probably wrong
        if x is None:
            try:
                return self.grid
            except AttributeError:
                self.grid = self.calc_grid(x)
                return self.grid
        else:
            grid = self.calc_grid(x)
            return grid
        
    def to_pointmass(self,ix_dens=1,max_horz=1,max_vert=1):
        density = self.values[:,ix_dens]

        top = 6371 - self.points[:,2] - 0.5 * self.sizes[:,2]
        bottom = 6371 - self.points[:,2] + 0.5 * self.sizes[:,2]
        return tesses_to_pointmass_max_size(self.points[:,0],self.points[:,1],
                                                       self.sizes[:,0],self.sizes[:,1],
                                                       top,bottom,density,max_horz=max_horz,max_vert=max_vert)
    
    def to_pointmass_old(self,ix_dens=1,hsplit=1,vsplit=1):
        density = self.values[:,ix_dens]

        top = 6371 - self.points[:,2] - 0.5 * self.sizes[:,2]
        bottom = 6371 - self.points[:,2] + 0.5 * self.sizes[:,2]
        return tesses_to_pointmass(self.points[:,0],self.points[:,1],
                                                       self.sizes[:,0],self.sizes[:,1],
                                                       top,bottom,density,hsplit=hsplit,vsplit=vsplit)
    
    def to_EWSNTBD(self,dens_ix=1,reference_model_interpolator=None):
        out = np.zeros((self.k,7))
        for d in range(3):
            out[:,2*d] = self.points[:,d] - 0.5 * self.sizes[:,d]
            out[:,2*d+1] = self.points[:,d] + 0.5 * self.sizes[:,d]
        # Convert radius to depth and to meters and positive up
        out[:,[4,5]] = -1000*(6371 - out[:,[5,4]])
        out[:,-1] = self.values[:,dens_ix]
        if not reference_model_interpolator is None:
            out[:,-1] *= reference_model_interpolator(1000*self.points[:,2]) * 0.01
        return out
    
    def to_tess_string(self,fmt='%.3f',dens_ix=1,reference_model_interpolator=None):
        EWSNTBD = self.to_EWSNTBD(dens_ix,reference_model_interpolator)
        out_str = ''
        fmt = fmt.strip() + ' '
        for i in range(self.k):
            out_str = out_str + (fmt*7) % tuple((EWSNTBD[i])) + '\n'
        return out_str
    
    def copy(self):
        the_copy = TesseroidModel(self.points.copy(),self.props.copy(),self.loni,self.lati,self.ri)
        the_copy.grid = self.grid.copy()
        return the_copy

class GeneticTesseroidModel(TesseroidModel):
    """Members of this class have a "memory" of how they derived from their parent
    notebook:: A
    Parameters
    ----------
    origin : tuple
        Describes how the model was created. parent_id,code,ksel,points,props = origin
        Meaning of code: 0 (birth), 1 (death), 2 (change), 3 (move).
        In all cases except code == 0, info contains the points and props associated
        with the *parent*. Technically this is redundant for 2 and 3, but who cares.
    """
    @staticmethod
    def single_tess_to_pointmass(points,props,ix_dens=1,max_horz=1,max_vert=1):
        density = props[3+ix_dens]
        props[:3] = np.clip(props[:3],0,None)
        top = 6371 - points[2] - 0.5 * props[2]
        bottom =  6371 - points[2] + 0.5 * props[2]
        return tesses_to_pointmass_max_size(np.array([points[0]]),np.array([points[1]]),np.array([props[0]]),np.array([props[1]]),np.array([top]),np.array([bottom]),np.array([density]),max_horz=max_horz,max_vert=max_vert)
    
    def __init__(self,points,props,loni=None,lati=None,ri=None,origin=None,tess_id=None,transformer=None):
        super().__init__(points,props,loni=loni,lati=lati,ri=ri,transformer=transformer)
        self.origin = origin
        self.tess_id = tess_id
        self.n_children = 0
    
    def birth(self,new_point,new_prop):
        new_points = np.vstack((self.points,new_point))
        new_props = np.vstack((self.props,new_prop))
        child_id = self.tess_id+self.n_children+1
        offspring = GeneticTesseroidModel(new_points,new_props,self.loni,self.lati,self.ri,tess_id=child_id,transformer=self.transformer)
        offspring.origin = (0,self.tess_id,offspring.k-1,None,None)
        self.n_children = self.n_children+1
        return offspring
    
    def death(self,ksel):
        new_points = np.delete(self.points,ksel,axis=0)
        new_props = np.delete(self.props,ksel,axis=0)
        child_id = self.tess_id+self.n_children+1
        
        offspring = GeneticTesseroidModel(new_points,new_props,self.loni,self.lati,self.ri,tess_id=child_id,transformer=self.transformer)
        offspring.origin = (1,self.tess_id,None,self.points[ksel],self.props[ksel])
        self.n_children = self.n_children+1        
        return offspring
    
    def change_param(self,ksel,new_props):
        child_id = self.tess_id+self.n_children+1
       
        offspring = GeneticTesseroidModel(self.points,new_props,self.loni,self.lati,self.ri,tess_id=child_id,transformer=self.transformer)
        offspring.origin = (2,self.tess_id,ksel,self.points[ksel],self.props[ksel])
        self.n_children = self.n_children+1        
        return offspring
    
    def move(self,ksel,new_points):
        child_id = self.tess_id+self.n_children+1
        
        offspring = GeneticTesseroidModel(new_points,self.props,self.loni,self.lati,self.ri,tess_id=child_id,transformer=self.transformer)
        offspring.origin = (3,self.tess_id,ksel,self.points[ksel],self.props[ksel])
        self.n_children = self.n_children+1
        
        return offspring        
    
    def to_pointmass(self,ix_dens=1,max_horz=1,max_vert=1,selection=None):
        if selection is None:
            selection = np.ones(self.k,dtype=bool)
        elif isinstance(selection,int):
            selection = [selection]
        density = self.values[selection,ix_dens]

        top = 6371 - self.points[selection,2] - 0.5 * np.clip(self.sizes[selection,2],0,None)
        bottom = 6371 - self.points[selection,2] + 0.5 * np.clip(self.sizes[selection,2],0,None)
        return tesses_to_pointmass_max_size(self.points[selection,0],self.points[selection,1],
                                                       self.sizes[selection,0],self.sizes[selection,1],
                                                       top,bottom,density,max_horz=max_horz,max_vert=max_vert)
    
    def copy(self):
        the_copy = GeneticTesseroidModel(self.points.copy(),self.props.copy(),self.loni,self.lati,self.ri)
        the_copy.origin = self.origin
        the_copy.tess_id = self.tess_id
        return the_copy


class GravityForward3D:
    def __init__(self,stations,reference_model_interpolator=None,RAM_save=False):
        self.stations = stations
        self.reference_model_interpolator = reference_model_interpolator
        if self.reference_model_interpolator is None:
            self.use_relative_dens = False
        else:
            self.use_relative_dens = True
        self.RAM_save = RAM_save
    
    def precalc_voxels(self,lon,lat,r):
        self.lon = lon
        self.lat = lat
        self.r = r
        loni,lati = np.meshgrid(lon,lat)
        self.sensitivity_matrix = np.zeros((len(self.stations),len(lat),len(lon),len(r)))
        spacings = lon[1]-lon[0],lat[1]-lat[0],r[1]-r[0]
        for i in range(len(r)):
            tops = 6371 - (np.ones(loni.size) * r[i]  + 0.5 * spacings[2])
            bots = 6371 - (np.ones(loni.size) * r[i]  - 0.5 * spacings[2])
            
            pointmasses = tesses_to_pointmass(loni.flatten(),lati.flatten(),spacings[0],spacings[1],tops,bots,np.ones(loni.size))
            temp = masspoint_calc_FTG_2(*pointmasses,self.stations[:,0],self.stations[:,1],self.stations[:,2],
                                                                                       calc_mode="grav")
            temp = np.squeeze(temp)
            self.sensitivity_matrix[:,:,:,i] = temp.reshape((-1,)+loni.shape)
            
    def calc_grav_tesseroid_to_pointmass(self,model,max_horz=1,max_vert=1):
        masspoints = model.to_pointmass(max_horz=max_horz,max_vert=max_vert)
        return self.calc_grav_pointmass(masspoints)
        
    def calc_grav_pointmass(self,masspoints):
        if self.use_relative_dens:
            masses = masspoints[-1]/100 * self.reference_model_interpolator(6371e3-1000*masspoints[2])
            masspoints = [masspoints[0],masspoints[1],masspoints[2],masses]
        
        if len(masses) == 0:
            print('empty masses')
            return np.zeros(len(self.stations))
        
        if self.RAM_save:
            temp = memory_save_masspoint_calc_FTG_2(*masspoints,lons=self.stations[:,0],lats=self.stations[:,1],heights=self.stations[:,2],calc_mode="grav")
            return np.squeeze(temp)
        else:
            temp = masspoint_calc_FTG_2(*masspoints,lons=self.stations[:,0],lats=self.stations[:,1],heights=self.stations[:,2],calc_mode="grav")
            return temp.sum((0))[...,0]
    
    def calc_grav_tesseroid_voxel(self,model):
        density_voxel = model()[...,1]
        r = self.r
        assert np.allclose(r,model.ri[0,0,:])
                
        if self.use_relative_dens:
            ref_dens = self.reference_model_interpolator(1000*r)
            density_voxel = ref_dens[None,None,:] * density_voxel * 0.01
        
        return np.einsum('ijkl,jkl->i',self.sensitivity_matrix,density_voxel)
    
    def calc_grav_tesseroid_cmd(self,model):
        if self.use_relative_dens:
            ### TODO: Shouldn't this be x100 because the properties are in percent?            
            EWSNTBD = model.to_EWSNTBD(reference_model_interpolator=self.reference_model_interpolator)
        else:
            EWSNTBD = model.to_EWSNTBD()
        np.savetxt('temp.tess',EWSNTBD,fmt='%.3f')
        with open('stations.txt') as f:
            proc = subprocess.Popen('tessgz temp.tess',stdin=f,stdout=subprocess.PIPE,universal_newlines=True)
            stdoutdata = proc.communicate()
            out = np.genfromtxt(stdoutdata[0].split('\n'))[:,-1]
        return out    

class UpdatingGravityForward3D(GravityForward3D):
    def __init__(self,stations,reference_model_interpolator=None,max_horz=None,max_vert=None,RAM_save=False,maxsize=4):
        self.stations = stations
        self.reference_model_interpolator = reference_model_interpolator
        if self.reference_model_interpolator is None:
            self.use_relative_dens = False
        else:
            self.use_relative_dens = True
        self.cache = collections.OrderedDict()
        self.maxsize = 4
        self.max_horz = max_horz
        self.max_vert = max_vert
        self.RAM_save = RAM_save
        
    
    def retrieve_cache(self,tess_id):
        if tess_id in self.cache:
            self.cache.move_to_end(tess_id)
            return self.cache[tess_id]
        else:
            return None
        
    def store_cache(self,tess_id,result):
        self.cache[tess_id] = result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(False)
    
    def calc_grav_update(self,model,max_horz=None,max_vert=None,force_recalc=False,additional_returns=False):
        
        if max_horz is None:
            max_horz = self.max_horz
        if max_vert is None:
            max_vert = self.max_vert
        recalc = force_recalc or (not hasattr(model,'origin')) or (model.origin is None) or (self.retrieve_cache(model.origin[1]) is None)
        
        if recalc:
            result = self.calc_grav_tesseroid_to_pointmass(model,max_horz,max_vert)
            current,previous = 0,0
        elif np.any(self.retrieve_cache(model.tess_id)):
            result = self.retrieve_cache(model.tess_id)
            current,previous = 0,0
        else:
            # Calc gravity effect of previous element (which has been changed/removed). In the case of birth step this is not necessary
            if model.origin[0] == 0:
                previous = 0
            else:
                pm = GeneticTesseroidModel.single_tess_to_pointmass(model.origin[3],model.origin[4],max_horz=max_horz,max_vert=max_vert)
                previous = np.squeeze(self.calc_grav_pointmass(pm))
            
            # Calc gravity effect of current element. Not necessary for death step
            if model.origin[0] == 1:
                current = 0
            else:
                ksel = model.origin[2]
                pm = GeneticTesseroidModel.single_tess_to_pointmass(model.points[ksel],model.props[ksel],max_horz=max_horz,max_vert=max_vert)
                current = np.squeeze(self.calc_grav_pointmass(pm))
            
            result = self.retrieve_cache(model.origin[1]) - previous + current
        
        self.store_cache(model.tess_id,result)

        if additional_returns:
            return result,previous,current
        else:
            return result

class TessMineosForward:
    def __init__(self,lon_ax,lat_ax,dep_ax,mineos_approx,freq_sel,flatten=True,apply_filter=False):
        self.lon_ax = lon_ax
        self.lat_ax = lat_ax
        self.dep_ax = dep_ax
        self.mineos_approx = mineos_approx
        self.n_freq = mineos_approx.ref_out.shape[0]
        self.flatten = flatten
        self.freq_sel = freq_sel
        self.n_freq = len(freq_sel)
        self.apply_filter = apply_filter
        if apply_filter:
            raise NotImplementedError('Filtering in 3D not implemented yet')
    
    def alternate_get(self,model):
        mineos_approx = self.mineos_approx
        A = mineos_approx.polynomials_rho[:,self.freq_sel]
        B = mineos_approx.polynomials_dvs[:,self.freq_sel]
        grids = model()
        any_nonzero = model.get_2D_nonzero_mask()
        grids = grids[any_nonzero,:]
        grids = np.transpose(grids,[1,0,2])
        
        rho_contrib = A[:,:,0].T.dot(grids[...,1]**2) +  A[:,:,1].T.dot(grids[...,1]) + A[:,:,2].T.sum(1)[:,None]
        vs_contrib =  B[:,:,0].T.dot(grids[...,0]**2) +  B[:,:,1].T.dot(grids[...,0]) + B[:,:,2].T.sum(1)[:,None]
        
        dvph = np.zeros((len(self.freq_sel),len(self.lat_ax),len(self.lon_ax)))
        dvph[:,any_nonzero] = rho_contrib+vs_contrib
        return dvph
    
    def old_get(self,model):
        mineos_approx = self.mineos_approx
        A = mineos_approx.polynomials_rho[:,self.freq_sel]
        B = mineos_approx.polynomials_dvs[:,self.freq_sel]
        grids = model()
        rho_contrib = np.einsum('ij,...i->j...',A[:,:,0],grids[...,1]**2) + np.einsum('ij,...i->j...',A[:,:,1],grids[...,1]) + A[:,:,2].sum(0)[:,None,None]
        vs_contrib = np.einsum('ij,...i->j...',B[:,:,0],grids[...,0]**2) + np.einsum('ij,...i->j...',B[:,:,1],grids[...,0]) + B[:,:,2].sum(0)[:,None,None]
        return rho_contrib+vs_contrib
    
    def calc_dvph_grids(self,grids):
        A = self.mineos_approx.polynomials_rho[:,self.freq_sel]
        B = self.mineos_approx.polynomials_dvs[:,self.freq_sel]
        rho_contrib = np.einsum('ij,...i->j...',A[:,:,0],grids[...,1]**2) + np.einsum('ij,...i->j...',A[:,:,1],grids[...,1]) + A[:,:,2].sum(0)[:,None,None]
        vs_contrib = np.einsum('ij,...i->j...',B[:,:,0],grids[...,0]**2) + np.einsum('ij,...i->j...',B[:,:,1],grids[...,0]) + B[:,:,2].sum(0)[:,None,None]
        return rho_contrib+vs_contrib

    def __call__(self,model):
        if self.flatten:
            return self.alternate_get(model).flatten()
        else:
            return self.alternate_get(model).reshape((self.n_freq,-1)).T
        
class TessMineosSHForward:
    def __init__(self,lmax,tess_mineos_forward):
        self.tess_mineos_forward = tess_mineos_forward
        self.lmax=lmax
        self.leg = legendrestuff.FloatLegCollection(self.lmax)
        self.leg.get_shifted_coeffs()
        self.l_vec = np.arange(self.leg.max_n+1)
    
    def calc_tess_coeff(self,points,props):
        N = points[1]+0.5*props[1]
        S = points[1]-0.5*props[1]
        W = points[0]-0.5*props[0]
        E = points[0]+0.5*props[0]
        integral_lat = self.leg.integrate_sin(*NS_to_theta(N,S))
        fourier_series = self.leg.fourier_series_tesseroid(W,E,self.leg.max_n)
        lmax = integral_lat.shape[0]-1
        coeffs = np.zeros((lmax+1,2*lmax+1))
        # Adjust coefficients to convention of Deuss
        coeffs[:,:lmax+1] = fourier_series[None,:,0] * integral_lat[:,:]
        coeffs[:,lmax+1:] = -fourier_series[None,1:,1] * integral_lat[:,1:]
        coeffs[:,0] = coeffs[:,0] * 2
        return coeffs*np.sqrt(np.pi/2.0)
    
    def calc_single(self,points,props):
        curve = self.tess_mineos_forward.calc_single(points,props[:3],props[3:])
        coeffs = self.calc_tess_coeff(points,props)
        return coeffs[None] * curve[:,None,None]
    
    def __call__(self,model,i=None):
        if i is None:
            return np.sum(np.array([self(model,k) for k in range(model.k)]),0)                
        else:
            return self.calc_single(model.points[i],model.props[i])

class GravityForwardSH(TessMineosSHForward):
    def __init__(self,lmax,height,dens_ix = 4,reference_model_interpolator=None):
        self.lmax=lmax
        self.leg = legendrestuff.FloatLegCollection(self.lmax)
        self.leg.get_shifted_coeffs()
        self.l_vec = np.arange(self.leg.max_n+1)
        self.height = height
        self.r = height + 6371
        self.dens_ix = dens_ix
        
        self.reference_model_interpolator = reference_model_interpolator
        if self.reference_model_interpolator is None:
            self.use_relative_dens = False
        else:
            self.use_relative_dens = True
            
    def calc_single(self,points,props):
        coeffs = self.calc_tess_coeff(points,props)
        r1 = points[2] - 0.5*props[2]
        r2 = points[2] + 0.5*props[2]
        
        G = 6.67428e-11
        
        l_vec = self.l_vec
        r = self.r
        grav_kernel = 4.0 * np.pi * G * (l_vec+1) / (2.0*l_vec+1) / (l_vec+3) * r * 1e3
        grav_kernel = grav_kernel * ((r2/r)**(l_vec+3) - (r1/r)**(l_vec+3))
        if self.use_relative_dens:
            return grav_kernel[:,None] * coeffs * props[self.dens_ix]/100 * self.reference_model_interpolator(points[2])
        else:
            return grav_kernel[:,None] * coeffs * props[self.dens_ix]

class TessMineosForwardSpline:
    @classmethod
    def from_multi(cls,mineos_approxs,freq_sel=None):
        """Create a Forward operator for several overtones at once
        The individual approximators must all use the same depth-axis for this to work.
        """
        obj = cls.__new__(cls)
        obj.dep_ax = mineos_approxs[0].depths
        # Staple the different coefficients from the sub_things together
        poly_dvs = np.hstack(([mineos_approx.polynomials_dvs[:,:,1] for mineos_approx in mineos_approxs]))
        poly_rho = np.hstack(([mineos_approx.polynomials_rho[:,:,1] for mineos_approx in mineos_approxs]))
        poly_dvp = np.hstack(([mineos_approx.polynomials_dvp[:,:,1] for mineos_approx in mineos_approxs]))
        obj.ref_out = np.vstack([mineos_approx.ref_out for mineos_approx in mineos_approxs])
        obj.n_vector = np.hstack([np.ones(mineos_approxs[i].polynomials_dvs.shape[1])*i for i in range(len(mineos_approxs))])
        
        obj.fit_spline(poly_dvs,poly_rho,poly_dvp)

        obj.polynomials_dvs = poly_dvs
        obj.polynomials_rho = poly_rho
        obj.polynomials_dvp = poly_dvp

        if freq_sel is None:
            obj.freq_sel = np.arange(len(obj.n_vector),dtype=int)
        else:
            obj.freq_sel = freq_sel
        return obj
        
    
    def fit_spline(self,poly_dvs,poly_rho,poly_dvp):
        self.spline_coeffs = np.zeros((poly_dvs.shape[1],poly_dvs.shape[0]+4,3)) #vs, rho, vp
        spacing = self.dep_ax[1] - self.dep_ax[0]
        for i in range(self.spline_coeffs.shape[0]):
            t,c,k = interpolate.splrep(self.dep_ax,poly_dvs[:,i]/spacing,s=0)
            self.spline_coeffs[i,:,0] = c
            t,c,k = interpolate.splrep(self.dep_ax,poly_rho[:,i]/spacing,s=0)
            self.spline_coeffs[i,:,1] = c
            t,c,k = interpolate.splrep(self.dep_ax,poly_dvp[:,i]/spacing,s=0)
            self.spline_coeffs[i,:,2] = c
            
            self.t,self.k = t,k     

    
    def __init__(self,mineos_approx,freq_sel=None):
        self.mineos_approx = mineos_approx
        self.dep_ax = mineos_approx.depths
        self.ref_out = mineos_approx.ref_out
        if freq_sel is None:
            self.freq_sel = np.arange(mineos_approx.ref_out.shape[0],dtype=int)
        else:
            self.freq_sel = freq_sel
        
        self.fit_spline(self.mineos_approx.polynomials_dvs[:,:,1],self.mineos_approx.polynomials_rho[:,:,1],self.mineos_approx.polynomials_dvp[:,:,1])
    
            
    def calc_single(self,points,sizes,values,freq_sel_override = None):
        """Calculate phase velocity effect of a single tesseroid
        Important: points contains (lon,lat,radius) of center, sizes similarly. 
        values is (dvs,drho), both in per cent!
        """
        if freq_sel_override is None:
            freq_sel = self.freq_sel
        else:
            freq_sel = freq_sel_override
        r1,r2 = points[2] - 0.5*sizes[2],points[2]+0.5*sizes[2]
        # Calculate integral of basis-splines
        _,B_integral = interpolate.splint(6371-r2,6371-r1,(self.t,self.spline_coeffs[0,:,0],self.k),True)
        # result = values[0] * (self.spline_coeffs[...,0] * B_integral).sum(1) + values[1] * (self.spline_coeffs[...,1] * B_integral).sum(1)
        result = values[0] * (self.spline_coeffs[freq_sel,:,0] * B_integral).sum(1) + values[1] * (self.spline_coeffs[freq_sel,:,1] * B_integral).sum(1)
        return result
    
    def __call__(self,model,i=None):
        if i is None:
            results = np.zeros((model.k,len(self.freq_sel)))
            for i in range(model.k):
                results[i] = self.calc_single(model.points[i],model.sizes[i],model.values[i])
        else:
            results = self.calc_single(model.points[i],model.sizes[i],model.values[i])
        return results
    
class GridWrapper:
    """Turns results from TessMineosForwardSpline into Grid
    """
    
    def __init__(self,wrapped,loni,lati):
        self.loni = loni
        self.lati = lati
        self.shape = loni.shape + (len(wrapped.freq_sel),)
        self.wrapped = wrapped
        
    def __call__(self,model,i=None):
        results = np.zeros(self.shape)
        if i is None:
            for j in range(model.k):
                mask = model.calc_2D_mask(j,x=(self.loni[...,None],self.lati[...,None],None))
                curve = self.wrapped(model,j)
                results[mask,:] += curve[None,:]
        else:
            mask = model.calc_2D_mask(i,x=(self.loni[...,None],self.lati[...,None],None))
            curve = self.wrapped(model,i)
            results[mask,:] = curve[None,:]
        return results
    
    def calc_single(self,points,props):
        klonk = TesseroidModel(points[None,:],props[None,:])
        return self(klonk,i=0)

class UpdateWrapper:
    """Class that caches intermediate results for use with GeneticTesseroidModel
    To work, the wrapped object needs to be callable and implement `calc_single`

    """
    def __init__(self,wrapped):
        self.wrapped = wrapped
        self.cache = collections.OrderedDict()
        self.maxsize = 4
    
    def retrieve_cache(self,tess_id):
        if tess_id in self.cache:
            self.cache.move_to_end(tess_id)
            return self.cache[tess_id]
        else:
            return None
        
    def store_cache(self,tess_id,result):
        self.cache[tess_id] = result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(False)
    
    def clear(self):
        self.cache.clear()

    def calc_previous(self,model):
        if model.origin[0] == 0:
            previous = 0
        else:
            previous = self.wrapped.calc_single(model.origin[3],model.origin[4])
        return previous
    
    def calc_current(self,model):
        if model.origin[0] == 1:
            current = 0
        else:
            ksel = model.origin[2]
            current = self.wrapped(model,ksel)
        return current
    
    def calc_update(self,model,force_recalc=False):
        retrieve = not ((self.retrieve_cache(model.tess_id)) is None)
        if retrieve and not force_recalc:
            result = self.retrieve_cache(model.tess_id)
        else:
            recalc = force_recalc or (not hasattr(model,'origin')) or (model.origin is None) or (self.retrieve_cache(model.origin[1]) is None)
            if recalc:
                result = self.wrapped(model)
                current,previous = 0,0

            else:
                current = self.calc_current(model)
                previous = self.calc_previous(model)

                result = self.retrieve_cache(model.origin[1]) - previous + current
            
        self.store_cache(model.tess_id,result)
        return result

class LayerStackModel3D(transdim.PointModel):
    """Class for describing a stack of layers in 3D
    Mainly tested for cartesian applications, but should also work 
    in spherical geometry.

    notebook:: A5A
    """
    y_extent = 1e6
    
    def __init__(self,points,props,n_layers,lims=[-np.inf,np.inf]):
        # Sort points
        if points.ndim == 1 or points.shape[1]==1:
            ix = np.argsort(np.squeeze(points))
            self.points = points[ix]
            self.props = props[ix]
        else:
            self.points = points
            self.props = props
            
        self.n_layers = n_layers
        self.n_props = props.shape[1]//n_layers - 1
        self.lims = lims
        self.ix_interpolator = interpolate.NearestNDInterpolator(self.points,np.arange(self.k,dtype=int))

        if self.ndim == 1:
            if len(points)>1:
                self.interpolator = interpolate.interp1d(np.squeeze(self.points),self.props,kind='nearest',bounds_error=False,axis=0,fill_value='extrapolate')
        else:
            self.interpolator = interpolate.NearestNDInterpolator(self.points,self.props)

    @property
    def ndim(self):
        return self.points.shape[1]
    
    @property
    def k(self):
        return self.points.shape[0]
    
    @property
    def thickness(self):
        return self.props[:,:self.n_layers]
    
    @property
    def values(self):
        return self.props[:,self.n_layers:].reshape((self.k,self.n_layers,-1))
    
    @property
    def bots(self):
        return np.cumsum(self.thickness,axis=1)
    
    def to_rectangles_3D(self,xi,yi,val_ix=None,flatten=False):
        dx,dy = xi[0,1]-xi[0,0],yi[1,0]-yi[0,0]
        bots = self.bots
        ix = self.ix_interpolator((xi,yi))
        tops = np.hstack((np.zeros((self.k,1)),bots[:,:-1]))
        centers = np.zeros(xi.shape+(self.n_layers,3))
        centers[...,0] = xi[:,:,None]
        centers[...,1] = yi[:,:,None]
        sizes = np.zeros(xi.shape+(self.n_layers,3))
        sizes[...,0] = dx
        sizes[...,1] = dy
        for k in range(self.k):
            centers[ix==k,:,2] = 0.5 * (tops[k]+bots[k])
            sizes[ix==k,:,2] = (bots[k]-tops[k])
        
        if val_ix is None:
            return centers,sizes
        else:
            raw_vals = self.values[:,:,val_ix]
            vals = np.zeros((xi.shape+(self.n_layers,)))
            for k in range(self.k):
                vals[ix==k] = raw_vals[k]
        
            return centers,sizes,vals
    
    def __call__(self,x):
        ix = self.ix_interpolator(x)
        return self.props[ix]
    
    def birth(self,new_point,new_prop):
        """Return a new PointModel with additional an additional point.
        """
        new_points = np.vstack((self.points,new_point))
        new_props = np.vstack((self.props,new_prop))
        return LayerStackModel3D(new_points,new_props,self.n_layers,self.lims)

    def death(self,ksel):
        new_points = np.delete(self.points,ksel,axis=0)
        new_props = np.delete(self.props,ksel,axis=0)
        return LayerStackModel3D(new_points,new_props,self.n_layers,self.lims)

    def change_param(self,ksel,new_props):
        return LayerStackModel3D(self.points,new_props,self.n_layers,self.lims)
    
    def move(self,ksel,new_points):
        return LayerStackModel3D(new_points,self.props,self.n_layers,self.lims)

class TransformingLayerStackModel(LayerStackModel3D):
    """Can be used to derive vp-vs-rho from any parameters

    This is useful - for example - if you want to use the vp-vs-ratio as free
    parameter in the inversion, or you want to make a one-parameter inversion,
    where you e.g. only specify density and derive the velocities.
    """
    def __init__(self,points,props,n_layers,transformer,n_props,lims=[-np.inf,np.inf]):
        super().__init__(points,props,n_layers,lims)
        self.transformer = transformer
        # Overwrite n_props, because this is determined by the transformer
        self.n_props = n_props
        
    @property
    def values(self):
        return self.transformer(super().values)
    
    @property
    def thickness(self):
        return np.clip(self.props[:,:self.n_layers],0,None)
    
    def copy(self):
        return TransformingLayerStackModel(self.points,self.props,self.n_layers,self.transformer,self.n_props,self.lims)
    
    def birth(self,new_point,new_prop):
        new_points = np.vstack((self.points,new_point))
        new_props = np.vstack((self.props,new_prop))
        return TransformingLayerStackModel(new_points,new_props,self.n_layers,self.transformer,self.n_props,self.lims)

    def death(self,ksel):
        new_points = np.delete(self.points,ksel,axis=0)
        new_props = np.delete(self.props,ksel,axis=0)
        return TransformingLayerStackModel(new_points,new_props,self.n_layers,self.transformer,self.n_props,self.lims)

    def change_param(self,ksel,new_props):
        return TransformingLayerStackModel(self.points,new_props,self.n_layers,self.transformer,self.n_props,self.lims)
    
    def move(self,ksel,new_points):
        return TransformingLayerStackModel(new_points,self.props,self.n_layers,self.transformer,self.n_props,self.lims)

class LayerStackGravMag3D:
    """Cartesian forward calculater of gravity with LayerStacks

    notebook:: A5A
    """
    def __init__(self,stations,component="grav",val_ix=0,max_dep=100,rho_halfspace = 3200):
        self.stations = stations
        self.component = component
        self.val_ix = val_ix
        self.max_dep = max_dep
        self.rho_halfspace = rho_halfspace
        
    def calc_grav(self,model,xi,yi):
        centers,sizes,dens = model.to_rectangles_3D(xi,yi,val_ix=self.val_ix,flatten=False)
        # Add halfspace
        rest = self.max_dep - sizes[...,2].sum(-1)
        centers_h = centers[...,0,:].copy()
        centers_h[...,2] = self.max_dep - 0.5 * rest
        sizes_h = sizes[...,0,:].copy()
        sizes_h[...,2] = rest
        dens_h = np.ones(xi.shape) * self.rho_halfspace
        A = cartesian.prism_sensitivity_matrix_grav(centers.reshape((-1,3)),sizes.reshape((-1,3)),self.stations)
        B = cartesian.prism_sensitivity_matrix_grav(centers_h.reshape((-1,3)),sizes_h.reshape((-1,3)),self.stations)
        return A.dot(dens.flatten()) + B.dot(dens_h.reshape((-1)))
    
class LayerStackDisba:
    """Cartesian forward calculator of dispersion curves with LayerStacks

    notebook:: A5A
    """
    def __init__(self,vp_hs,vs_hs,rho_hs,periods,sel_ix=[0,1,2],mode=None):
        self.vp_hs = vp_hs
        self.vs_hs = vs_hs
        self.rho_hs = rho_hs
        self.periods = periods
        self.sel_ix = sel_ix
        self.mode = mode
    
    def calc_phase_vel(self,model):
        phase_vel = np.zeros((model.k,len(self.periods)))
        for k in range(model.k):
            disba_input = np.zeros((model.n_layers+1,4))
            disba_input[:model.n_layers,0] = model.thickness[k]
            disba_input[:model.n_layers,1:4] = model.values[k,:,self.sel_ix].T
            
            disba_input[-1,3] = self.rho_hs
            disba_input[-1,1] = self.vp_hs
            disba_input[-1,2] = self.vs_hs
            obj = disba.PhaseDispersion(*disba_input.T)
            phase_vel[k] = obj(self.periods,wave=self.mode)[1]
        return phase_vel

class CartesianRayTracer:
    """Raytracer with *straight lines* in cartesian coordinates

    Supports two kinds of approximations: (a) velocity gets interpolated
    along the line or (b) travel times in a fixed grid are pre-calculated
    and then used during calculation. (a) is more accurate, but can be 
    slower with a lot of rays.

    notebook:: A5A
    """
    def __init__(self,eq,stations,stepsize):
        self.eq = eq
        self.stations = stations
        self.stepsize = stepsize
        self.rays = dict()
        self.ray_pars = np.zeros((len(stations),len(eq),5)) # Start, ray vector, length
        self.pd = spatial.distance.cdist(stations,eq)
        for i in range(len(stations)):
            for j in range(len(eq)):
                alpha = np.linspace(0,1,int(self.pd[i,j]//stepsize))
                ray = np.zeros((len(alpha),3))
                ray[:,0] = alpha
                ray[:,1] = (stations[i,0]-eq[j,0])*alpha + eq[j,0]
                ray[:,2] =  (stations[i,1]-eq[j,1])*alpha + eq[j,1]
                self.rays[i,j] = ray
                self.ray_pars[i,j,:2] = eq[j]
                self.ray_pars[i,j,2:4] = (stations[i]-eq[j])/self.pd[i,j]
                self.ray_pars[i,j,4] = self.pd[i,j]
    
    
    def trivial_trace(self,model):
        ps = np.zeros((model.k,)+self.pd.shape)
        k_vec = np.arange(model.k,dtype=int)
        for i,j in self.rays:
            ray = self.rays[i,j]
            ix = model.ix_interpolator((ray[:,1],ray[:,2]))
            comp = ix[:,None] == k_vec[None,:]
            ps[:,i,j] = comp.sum(0) * self.stepsize
        return ps
    
    
    def precalc_grid(self,xi,yi):
        self.xi = xi
        self.yi = yi
        # Convert from cell centered to edges
        dx = xi[0,1] - xi[0,0]
        dy = yi[1,0] - yi[0,0]
        x = xi[0,:] + dx/2
        x = np.append(x[0]-dx,x)
        y = yi[:,0] + dy/2
        y = np.append(y[0]-dy,y)
        alpha_x = (x[None,None,:] - self.ray_pars[:,:,None,0])/self.ray_pars[:,:,None,2]
        alpha_x[alpha_x<0] = np.nan
        alpha_x[alpha_x>self.ray_pars[:,:,None,4]] = np.nan
        y_rays = alpha_x * self.ray_pars[:,:,None,3] + self.ray_pars[:,:,None,1]
        
        alpha_y = (y[None,None,:] - self.ray_pars[:,:,None,1])/self.ray_pars[:,:,None,3]
        alpha_y[alpha_y<0] = np.nan
        alpha_y[alpha_y>self.ray_pars[:,:,None,4]] = np.nan
        x_rays = alpha_y * self.ray_pars[:,:,None,2] + self.ray_pars[:,:,None,0]
                
        hit_count = np.zeros(self.pd.shape+xi.shape)
        
        start_row = ((self.eq[:,1] - y[0])//dy).astype(int)
        start_col= ((self.eq[:,0] - x[0])//dx).astype(int)
        
        end_row = ((self.stations[:,1] - y[0])//dy).astype(int)
        end_col= ((self.stations[:,0] - x[0])//dx).astype(int)
        
        
        for i,j in self.rays:
            glorgon = np.zeros((len(x)+len(y)+2,3))
            glorgon[:len(x),0] = alpha_x[i,j]
            glorgon[:len(x),1] = np.nan
            glorgon[:len(x),2] = np.arange(len(x))
            
            glorgon[len(x):-2,0] = alpha_y[i,j]
            glorgon[len(x):-2,1] = np.arange(len(y))
            glorgon[len(x):-2,2] = np.nan
            
            glorgon[-2] = 0,start_row[j],start_col[j]
            glorgon[-1] = self.pd[i,j],end_row[i],end_col[i]
            
            ix = np.argsort(glorgon[:,0])
            glorgon=glorgon[ix]
            a,r,c = glorgon[0]
            r = int(r)
            c = int(c)
            for k in range(1,glorgon.shape[0]):
                d = glorgon[k,0] - a
                if np.isnan(d):
                    break
                hit_count[i,j,r,c] = d
                a = a+d
                if ~np.isnan(glorgon[k,1]):
                    r = int(glorgon[k,1])
                if ~np.isnan(glorgon[k,2]):
                    c = int(glorgon[k,2])
        
        self.hit_count = hit_count
        
        return x_rays,y_rays,hit_count,glorgon
    
    def grid_trace(self,model):
        ps = np.zeros((model.k,)+self.pd.shape)
        ix = model.ix_interpolator((self.xi,self.yi))
        for k in range(model.k):
            ps[k] = self.hit_count[:,:,ix==k].sum((2))
        return ps
    
    def calc_time(self,phase_vel,ps):
        return (ps[:,None,:,:]/phase_vel[:,:,None,None] ).sum(0)
    
class RF_TT:
    """Calculates traveltimes of receiver functions
    
    notebook:: A5A
    """
    def __init__(self,ix):
        self.ix = ix
        self.func_dict = dict(PS=self._PS,PPPS=self._PPPS,PSPS=self._PSPS)
    
    def calc_RF(self,thickness,vp,vs,type='PS'):
        raw = self.func_dict[type](thickness,vp,vs)
        return raw[:,:self.ix+1].sum(1)
    
    def _PS(self,thickness,vp,vs):
        return thickness*(1.0/vs - 1.0/vp)
    
    def _PPPS(self,thickness,vp,vs):
        return thickness*(1.0/vs + 1.0/vp)
    
    def _PSPS(self,thickness,vp,vs):
        return thickness*(2.0/vs)
    
    def H_kappa(self,dT,code,H_vals,k_vals,vp):
        Hi,ki = np.meshgrid(H_vals,k_vals)
        vsi = vp/ki
        
        raw = self.func_dict[code](Hi,vp,vsi)
        return raw

class SimpleRefraction:
    """Calculates critical distances of refracted waves in layered medium
    """
    def __init__(self,v_halfspace):
        self.v_halfspace = v_halfspace
    def calc_angles(self,thickness,v):
        # Waves running at the top of layer i+1 (nothing in top layer)
        # Last angle is for refraction at half space
        # angle[:,i,j] -> wave propagating in layer i that will be refracted at bottom of layer j, so travel in layer j+1
        angles = np.zeros((thickness.shape[0],thickness.shape[1],thickness.shape[1]))
        for i in range(thickness.shape[1]):
            for j in range(thickness.shape[1]-1):
                angles[:,i,j] = np.arcsin(v[:,i]/v[:,j+1])
        
        angles[:,:,-1] = np.arcsin(v/self.v_halfspace)        
        return angles
    
    def calc_critical(self,thickness,v):
        blocker = np.triu(np.ones((thickness.shape[1],thickness.shape[1])))
        angles = self.calc_angles(thickness,v)
        ds = thickness[:,None]/v[:,None] * np.cos(angles) * blocker[None]
        return 2*ds.sum(1)