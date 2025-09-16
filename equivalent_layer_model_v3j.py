# -*- coding: utf-8 -*-

#import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.interpolate as spint
import scipy.sparse as sparse 

import numba
from numba import jit, config, prange


def igrf_comp(D,I,igrf):
    """Convert D, I, IGRF in X, Y, Z.
    The returned matrix has the same shape as the inputs, except that an axis is
    added at the end containing the three xyz values
    """
    I_rad = I/180.0*np.pi
    D_rad = D/180.0*np.pi

    x = np.cos(I_rad) * np.cos(D_rad) * (igrf)
    y = np.cos(I_rad) * np.sin(D_rad) * (igrf)
    z = np.sin(I_rad) * (igrf)
    return np.dstack((x,y,z))

def ned2ecef(lon,lat,height,earth_radius=6371000.0):
    """Convert from lon,lat,height into Earth-Centered-Earth-Fixed coordinate system.
    The returned matrix has the same shape as the inputs, except that an axis is
    added at the end containing the three coordinates xyz
    """
    lat_rad=lat/180.0*np.pi
    lon_rad=lon/180.0*np.pi

    x = np.cos(lat_rad) * np.cos(lon_rad) * (earth_radius+height)
    y = np.cos(lat_rad) * np.sin(lon_rad) * (earth_radius+height)
    z = np.sin(lat_rad) * (earth_radius+height)
    #return np.dstack((x,y,z))
    ecefs = np.zeros((len(lon), 3))
    ecefs[:,0] = x
    ecefs[:,1] = y
    ecefs[:,2] = z

    return ecefs

def ned2ecefv(lon,lat,height,vector_field,earth_radius=6371000.0):
    """Convert a vector field defined in terms of north-eastdown into ecef vector
    the first two dimensions of the points and vector_field must agree and the
    last dimension of vector_field needs to be size three and contain the actual
    ned components
    """

    lat_rad=lat/180.0*np.pi
    lon_rad=lon/180.0*np.pi

    # Calculate the NED base vectors
    b_down = -np.dstack((np.cos(lat_rad) * np.cos(lon_rad),
                        np.cos(lat_rad) * np.sin(lon_rad),
                        np.sin(lat_rad)))
    b_north = np.dstack((-np.sin(lat_rad) * np.cos(lon_rad),
                        -np.sin(lat_rad) * np.sin(lon_rad),
                        np.cos(lat_rad)))
    b_east = np.dstack((-np.sin(lon_rad),
                        np.cos(lon_rad),
                        np.zeros(lon.shape)))
    
    # Some broadcasting magic to allow arbitrary input size
    # Note that with tuples "+" concatenates tuples!
    slice_tuple = (slice(None),)*lon.ndim + (None,)*(vector_field.ndim-1-lon.ndim)
    b_north = b_north[slice_tuple]
    b_east = b_east[slice_tuple]
    b_down = b_down[slice_tuple]

    #ecef_vector_field = np.zeros(vector_field.shape)
    ecef_vector_field = np.zeros((len(lon), 3))
    ecef_vector_field1 = np.zeros((len(lon), 3))


#    V = np.zeros((len(lon), 3))
#    V[:,:] = vector_field
    V = vector_field

    R=np.zeros((len(lon),3,3))
    #R=np.zeros((len(lon), len(lon),3,3))
    R[:,:,0]=b_north
    R[:,:,1]=b_east
    R[:,:,2]=b_down
    
    # WS: This should give the same result as the for loop below
    ecef_vector_field = np.einsum('kij,kj->ki',R,V)

    return ecef_vector_field

#test = ned2ecefv(lon, lat, -dipole_depth, vec_igrf)
#print(test)

def ecef2nedv(lon,lat,height,ecef_vector_field,earth_radius=6371000.0):
    # TODO(Judith): Implementiere analog zu ned2ecefv
    """Convert a vector field defined in terms of north-eastdown into ecef vector
    the first two dimensions of the points and vector_field must agree and the
    last dimension of vector_field needs to be size three and contain the actual
    ned components
    """
    lat_rad=lat/180.0*np.pi
    lon_rad=lon/180.0*np.pi

    # Calculate the NED base vectors
    b_down = -np.dstack((np.cos(lat_rad) * np.cos(lon_rad),
                        np.cos(lat_rad) * np.sin(lon_rad),
                        np.sin(lat_rad)))
    b_north = np.dstack((-np.sin(lat_rad) * np.cos(lon_rad),
                        -np.sin(lat_rad) * np.sin(lon_rad),
                        np.cos(lat_rad)))
    b_east = np.dstack((-np.sin(lon_rad),
                        np.cos(lon_rad),
                        np.zeros(lon.shape)))
    
    # Some broadcasting magic to allow arbitrary input size
    # Note that with tuples "+" concatenates tuples!
    slice_tuple = (slice(None),)*lon.ndim + (None,)*(ecef_vector_field.ndim-1-lon.ndim)
    b_north = b_north[slice_tuple]
    b_east = b_east[slice_tuple]
    b_down = b_down[slice_tuple]

    #ecef_vector_field = np.zeros(vector_field.shape)
    vector_field = np.zeros((len(lon), 3))
    #R_inv = np.zeros((len(lon), 3,3))
    #R=np.zeros((len(lon),3,3))
    R=np.zeros((len(lon), len(lon), 3, 3))
    R[:,:,:,0]=b_north
    R[:,:,:,1]=b_east
    R[:,:,:,2]=b_down

    #E = np.zeros((len(lon), 3)) #,  ((len(lon), len(lon), 3))
    E = ecef_vector_field

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        print("Error")
        pass
   
    # WS: This should give the same result as the for loop below
    vector_field = np.einsum('...kij,...kj->...ki',R_inv,E)

    return vector_field
 
#aero_test = ned2ecefv(lons, lats, hgridi, vec_igrf)
#print(np.shape(aero_test)) 
#ned_vector = ecef2nedv(lon_ax,lat_ax,hgridi,aero_test,earth_radius=6371000.0)
#print(ned_vector)

def mag_dipole_matrix(dipoles,cell_sizes,magnetizations,stations,**kwargs):
    """Calculate the magnetic effect of a set of dipoles on a set of stations.
    This is a calculation in cartesian coordinates. 
    To conserve a bit of memory, this function splits the data into partitions.
    
    Parameters
    ----------
    dipoles : np.array
        Shape is (n_dipoles,3) and contains the XYZ coordinates of each dipole
    cell_sizes : np.array
        "Cell sizes" of the dipoles (n) (to define weights in inversion)
    magnetizations: np.array
        Shape is (n_dipoles,3) and contains the XYZ of the dipole magnetization
    stations: np.narray
        Shape is (n_stations,3) and contains the XYZ coordinates of each station
    verbose : bool, optional
        If true, additional info about partitioning is printed to stdout
    max_size : int, optional
        The size of the partitions in bytes. The actual memory use is a bit higher,
        so you might have to fiddle around a bit until you have a good result.
    Returns
    -------
    design_matrix: np.array
        Shape (n_stations,n_dipoles,3) containing the effect of all dipoles on every
        station
    """
    
    max_size = kwargs.get("max_size",1000000000) # in bytes
    verbose = kwargs.get("verbose",False)
    N_stations,N_dipoles = stations.shape[0],magnetizations.shape[0]
    
    partitions = max(1,np.ceil(8 * N_stations * N_dipoles / max_size).astype(int))
    ixs = np.array_split(np.arange(N_stations,dtype=int),partitions)
    design_matrix = np.zeros((N_stations,N_dipoles,3))
    
    if verbose:
        print(partitions)
    
    for ix in ixs:
        r = stations[ix,None,:] - dipoles[None,:,:]
        r_val = np.sqrt((r**2).sum(-1))
        T1 = 3.0*(magnetizations[None,:,:]*r).sum(-1)/(r_val**5)
        T2 = 1.0/(r_val**3)
        print(np.shape(T1), np.shape(T2), np.shape(r), np.shape(magnetizations))

        design_matrix[ix] =  (T1[:,:,None] * r - T2[:,:,None] * magnetizations[None, :,:])
        design_matrix[ix] = design_matrix[ix] * cell_sizes[None,:,None]  #Gewichtung mit Zellgrösse
    return design_matrix

@jit(nopython=True)    
#@jit(nopython=True, parallel=True)    
def para_routine_I(mod_vec, stations_ECEF, dipoles_ECEF, igrf_ECEF_stat, igrf_ECEF_dip, cell_sizes, dist_thresh, init_stations_ECEF, first_idx_DGRF, nr_idx_DGRF):
    """Run this part numba parallelized routine"""
    dat_vec = np.zeros(nr_idx_DGRF)

    for p in range(first_idx_DGRF,nr_idx_DGRF+first_idx_DGRF): #loop over data points
        
        r = stations_ECEF[p,None,:] - dipoles_ECEF  
        
        r_val = np.sqrt((r**2).sum(-1))

        if dist_thresh > 0.0:
            idx_near = r_val <= dist_thresh
            
            r = r[idx_near,:]
            r_val = r_val[idx_near]
            igrf_ECEF_dip1 = igrf_ECEF_dip[idx_near,:]
            cell_sizes1 = cell_sizes[idx_near]
        else:
            igrf_ECEF_dip1 = igrf_ECEF_dip
            cell_sizes1 = cell_sizes

                   
        T1 = cell_sizes1*3.0*(igrf_ECEF_dip1*r).sum(-1)/(r_val**5)
        T2 = cell_sizes1*1.0/(r_val**3)
        
        matrix_row = ((igrf_ECEF_stat[p,0]*(T1 * r[:,0] - T2 * igrf_ECEF_dip1[:,0])) + (igrf_ECEF_stat[p,1]*(T1 * r[:,1] - T2 * igrf_ECEF_dip1[:,1])) + (igrf_ECEF_stat[p,2]*(T1 * r[:,2] - T2 * igrf_ECEF_dip1[:,2]))) / np.sqrt((igrf_ECEF_stat[p,:]**2).sum(-1))

        if dist_thresh > 0.0:
            dat_vec[p-first_idx_DGRF] = matrix_row.dot(mod_vec[idx_near])
        else:
            dat_vec[p-first_idx_DGRF] = matrix_row.dot(mod_vec)
        
        #print("hallo", p, dat_vec[p-first_idx_DGRF])


    return dat_vec


def design_matrix_multi_model_vec(stations_ECEF, dipoles_ECEF, cell_sizes, igrf_ECEF_stat, igrf_ECEF_dip, mod_vec, dist_thresh, first_idx_DGRF, nr_idx_DGRF):
    """Calculate the multiplication of the sensitivity matrix and model vector line by line - internal loop
    
    Parameters
    ----------
    stations_ECEF : np.array(shape=(n,3))
        earth centered coordinate system
    dipoles_ECEF : np.array(shape=(m,3))
        earth centered coordinate system
    cell_sizes : np.array
        "Cell sizes" of the dipoles (n) (to define weights in inversion)
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
    mod_vec : np.array
        Magnetic model vector
    dist_thresh : <= 0.0 dense matrix; otherwise distance in m, where the matrix entries are ignored
    first_idx_DGRF : np.array
        index of the first data point used for specific DGRFs (associated with different reference years) 
    nr_idx_DGRF: np.array
        The number of data points used a specific DGRF
       
    Returns
    -------
    dat_vec : np.array
        Data vector of F(m)
    """
    dat_vec = np.zeros(igrf_ECEF_stat.shape[0])
    ndip, dummy = dipoles_ECEF.shape
    
    less_core = 0
    
    if dist_thresh > 0.0:    
        print("Above a distance of ", dist_thresh, "m, the equivalent sources are not considered ")
            
    for i in range(first_idx_DGRF.size): #Loop over the dipoles of the different reference years
        if nr_idx_DGRF[i] > 0:
            
            tmp_igrf_ECEF_dip = igrf_ECEF_dip[i,:,:]
            tmp_stations_ECEF = np.zeros([ndip,3])
                        
            numba.config.NUMBA_NUM_THREADS = numba.config.NUMBA_DEFAULT_NUM_THREADS - less_core
            print("Mod. Number of used cores:", numba.config.NUMBA_NUM_THREADS)
            
            #Run this stuff parallel
            tmp_dat_vec = para_routine_I(mod_vec, stations_ECEF, dipoles_ECEF, igrf_ECEF_stat, tmp_igrf_ECEF_dip, cell_sizes, dist_thresh, tmp_stations_ECEF, first_idx_DGRF[i], nr_idx_DGRF[i])
            
            numba.config.NUMBA_NUM_THREADS = numba.config.NUMBA_DEFAULT_NUM_THREADS
            print("Default Number of used cores:", numba.config.NUMBA_NUM_THREADS)


            dat_vec[first_idx_DGRF[i]:first_idx_DGRF[i]+nr_idx_DGRF[i]] = dat_vec[first_idx_DGRF[i]:first_idx_DGRF[i]+nr_idx_DGRF[i]] + tmp_dat_vec
                        
    return dat_vec


def calculate_design_matrix_multi_model_vec(stations,dipoles,cell_sizes,igrf_stat,igrf_dip, mod_vec, dist_thresh, first_idx_DGRF, nr_idx_DGRF):
    """Calculate the multiplication of the sensitivity matrix and model vector line by line 
    
    Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    cell_sizes : np.array
        "Cell sizes" of the dipoles (n) (to define weights in inversion)
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
    mod_vec : np.array
        Magnetic model vector
    dist_thresh : <= 0.0 dense matrix; otherwise distance in m, where the matrix entries are ignored
    first_idx_DGRF : np.array
        index of the first data point used for specific DGRFs (associated with different reference years) 
    nr_idx_DGRF: np.array
        The number of data points used a specific DGRF
       
    Returns
    -------
    dat_vec : np.array
        Data vector of F(m)
    """
    
    # Shape assumptions:
    # stations nx3 (lon,lat, height (in meters))
    # dipoles nx3 (lon,lat, height (in meters))
    # IGRF is in NED
    stations_ECEF = ned2ecef(stations[:,0],stations[:,1],stations[:,2])
    dipoles_ECEF = ned2ecef(dipoles[:,0],dipoles[:,1],dipoles[:,2])
    
    igrf_ECEF_dip = np.zeros(igrf_dip.shape)
            
    for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years
        tmp_igrf_dip = np.squeeze(igrf_dip[i,:,:])
        tmp_igrf_ECEF_dip = ned2ecefv(dipoles[:,0],dipoles[:,1],dipoles[:,2],tmp_igrf_dip)
        igrf_ECEF_dip[i,:,:] = tmp_igrf_ECEF_dip
    
    igrf_ECEF_stat = ned2ecefv(stations[:,0],stations[:,1],stations[:,2],igrf_stat)
 
    #loop to determine the design matrix and determine the multiplication wih model vector
    dat_vec = design_matrix_multi_model_vec(stations_ECEF, dipoles_ECEF, cell_sizes, igrf_ECEF_stat, igrf_ECEF_dip, mod_vec, dist_thresh, first_idx_DGRF, nr_idx_DGRF)

    return dat_vec


def calculate_sparse_design_matrix_multi_model_vec(stations,dipoles,cell_sizes,igrf_stat,igrf_dip, mod_vec, dist_thresh, max_data, first_idx_DGRF, nr_idx_DGRF):
    """Calculate the multiplication of a sparse sensitivity matrix and model vector line by line
        Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    cell_sizes : np.array
        "Cell sizes" of the dipoles (n) (to define weights in inversion)
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
    mod_vec : np.array
        Magnetic model vector
    dist_thresh : <= 0.0 dense matrix; otherwise distance in m, where the matrix entries are ignored
    first_idx_DGRF : np.array
        index of the first data point used for specific DGRFs (associated with different reference years) 
    nr_idx_DGRF: np.array
        The number of data points used a specific DGRF

       
    Returns
    -------
    dat_vec : np.array
        Data vector of F(m)
    """
    
    # IGRF is in NED
    stations_ECEF = ned2ecef(stations[:,0],stations[:,1],stations[:,2])
    dipoles_ECEF = ned2ecef(dipoles[:,0],dipoles[:,1],dipoles[:,2])
    
    igrf_ECEF_dip = np.zeros(igrf_dip.shape)
        
    for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years
        tmp_igrf_dip = np.squeeze(igrf_dip[i,:,:])
        tmp_igrf_ECEF_dip = ned2ecefv(dipoles[:,0],dipoles[:,1],dipoles[:,2],tmp_igrf_dip)
        igrf_ECEF_dip[i,:,:] = tmp_igrf_ECEF_dip

    igrf_ECEF_stat = ned2ecefv(stations[:,0],stations[:,1],stations[:,2],igrf_stat)
    
    N_stations,N_dipoles = stations_ECEF.shape[0],dipoles_ECEF.shape[0]

    idx_b = 0   #finish the loop if =1
    block_row =  round(max_data/N_dipoles) #define the number of rows in a block    
    min_row = 0
    max_row = block_row
    
    dat_vec = np.linspace(0.0,0.0, N_stations)
    
    if dist_thresh > 0.0:    
        print("Above a distance of ", dist_thresh, "m, the equivalent sources are not considered ")
    
    while idx_b == 0: #loop over all stations (rows)
    
        if max_row < N_stations:
            block_idx = np.arange(min_row, max_row)     
        else:
            block_idx = np.arange(min_row, N_stations)
            idx_b = 1
            max_row = N_stations
        
        N_stations_blk = max_row - min_row
            
        ano_design_matrix_blk = sparse.csr_matrix((N_stations_blk,N_dipoles))
        
        r = stations_ECEF[block_idx,None,:] - np.squeeze(dipoles_ECEF[None,:,:]) #three components of distance of dipoles and measuring locations        
        r_val = np.sqrt((r**2).sum(-1))  


        if dist_thresh > 0.0:
            used_val = r_val < dist_thresh #Find all values below a distance
        else: # if the distance is not defined, all data are taken
            used_val = r_val > -1.0        
        
        
        x_list, y_list  = np.where(used_val)  #row and colum number of nonzero tuples
        x_list = x_list + min_row
        
                
        for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years of the DGRF models
            if nr_idx_DGRF[i] > 0:
                idx_select = np.where((x_list >= first_idx_DGRF[i]) & (x_list < first_idx_DGRF[i] + nr_idx_DGRF[i]))
                                
                x_list_select = x_list[idx_select]
                y_list_select = y_list[idx_select]
                
                
                #bring distance components into sparse (csr-format) matrices
                r_x = sparse.coo_matrix((r[x_list_select - min_row, y_list_select,0], (x_list_select - min_row, y_list_select)), shape=(N_stations_blk,N_dipoles)).tocsr() 
                r_y = sparse.coo_matrix((r[x_list_select - min_row, y_list_select,1], (x_list_select - min_row, y_list_select)), shape=(N_stations_blk,N_dipoles)).tocsr()
                r_z = sparse.coo_matrix((r[x_list_select - min_row, y_list_select,2], (x_list_select - min_row, y_list_select)), shape=(N_stations_blk,N_dipoles)).tocsr()
                
                #From here on everything is sparse:        
                r_val = np.sqrt(r_x.power(2) + r_y.power(2) + r_z.power(2))

                T1 = 3.0*(r_x.multiply(np.squeeze(igrf_ECEF_dip[i,:,0])) + r_y.multiply(np.squeeze(igrf_ECEF_dip[i,:,1])) + r_z.multiply(np.squeeze(igrf_ECEF_dip[i,:,2]))).multiply(r_val.power(-5))
        
                T2 = r_val.power(-3)
        
                del r_val
        
                T1 = T1.multiply(cell_sizes) #weight from the cell sizes
                T2 = T2.multiply(cell_sizes) #weight from the cell sizes

                design_matrix_x =  T1.multiply(r_x) - T2.multiply(np.squeeze(igrf_ECEF_dip[i,:,0]))
                design_matrix_y =  T1.multiply(r_y) - T2.multiply(np.squeeze(igrf_ECEF_dip[i,:,1]))
                design_matrix_z =  T1.multiply(r_z) - T2.multiply(np.squeeze(igrf_ECEF_dip[i,:,2]))
        
                del T1
                del T2

                del r_x
                del r_y
                del r_z
        
                design_matrix_x = design_matrix_x.transpose()
                design_matrix_y = design_matrix_y.transpose()
                design_matrix_z = design_matrix_z.transpose()
                
                tmp_design_matrix =  (design_matrix_x.multiply(np.squeeze(igrf_ECEF_stat[min_row:max_row,0])) + design_matrix_y.multiply(np.squeeze(igrf_ECEF_stat[min_row:max_row,1])) + design_matrix_z.multiply(np.squeeze(igrf_ECEF_stat[min_row:max_row,2])))
                
                del design_matrix_x
                del design_matrix_y
                del design_matrix_z
        
                tmp_design_matrix = tmp_design_matrix.multiply(1.0/np.sqrt((igrf_ECEF_stat[min_row:max_row,:]**2).sum(-1)))
                ano_design_matrix_blk = ano_design_matrix_blk + tmp_design_matrix.transpose()

                del tmp_design_matrix
        
        
        dat_vec[min_row:max_row] = ano_design_matrix_blk.dot(mod_vec)
        
        del ano_design_matrix_blk

        min_row = max_row
        max_row = max_row + block_row 
        
    return dat_vec


def assign_block(lon,lat,southern,northern,eastern,western):
    """Assign points to blocks

    Parameters
    ----------
    lon : np.array
        Longitudes of points
    lat : np.array
        Latitudes of points
    southern : np.array
        Southern boundaries of the blocks
    northern : np.array
        Northern boundaries of the blocks
    eastern : dict
        Eastern boundaries of the blocks
    western : dict
        Western boundaries of the blocks
    
    Returns
    -------
    block_row : np.array(dtype=int)
        The north-south block index of all points
    block_col : np.array(dtype=int)
        The east-west block index of all points
    """
    block_row = np.argmax(lat[None,...]<northern.reshape(northern.shape+(1,)*lat.ndim),axis=0)
    block_col = np.zeros(lon.shape,dtype=int)
    for i in np.unique(block_row):
        sel = block_row == i
        block_col[sel] = np.argmax(lon[None,sel]<eastern[i][:,None],axis=0)
    return block_row,block_col

class BlockIterator:
    """Create an iterator over all rows and columns of a block definition

    Note that this is not straightforward, because the number of blocks
    is different for each row due to the convergence at the poles.
    """
    def __init__(self,southern,eastern):
        self.southern = southern
        self.eastern = eastern
    
    def __iter__(self):
        self.i,self.j = 0,0
        return self
    
    def __next__(self):
        result = self.i,self.j
        if result[0] == len(self.southern):
            raise StopIteration
        if self.j == len(self.eastern[self.i])-1:
            self.i = self.i + 1
            self.j = 0
        else:
            self.j = self.j + 1

        return result

def calculate_ano_design_matrix(stations,dipoles, cell_sizes, igrf_stat, igrf_dip, first_idx_DGRF, nr_idx_DGRF):
    """Calculate the magnetic anomaly design matrix

    Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    cell_sizes : np.array(shape=(m,1))
        "Cell sizes" of the dipoles (to define weights in inversion)
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array(shape=(m,3))
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
    first_idx_DGRF : np.array
        index of the first data point used for specific DGRFs (associated with different reference years) 
    nr_idx_DGRF: np.array
        The number of data points used a specific DGRF
            
    Returns
    -------
    ano_design_matrix : np.array(shape=(n,m))
        Each row gives the effect of all dipoles on a station, assuming a 
        source strength of one at the dipole.
    """
    # Shape assumptions:
    # stations nx3 (lon,lat, height (in meters))
    # dipoles nx3 (lon,lat, height (in meters))
    # IGRF is in NED
    stations_ECEF = ned2ecef(stations[:,0],stations[:,1],stations[:,2])
    dipoles_ECEF = ned2ecef(dipoles[:,0],dipoles[:,1],dipoles[:,2])

    igrf_ECEF_dip = np.zeros(igrf_dip.shape)
        
    for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years
        tmp_igrf_dip = np.squeeze(igrf_dip[i,:,:])
        tmp_igrf_ECEF_dip = ned2ecefv(dipoles[:,0],dipoles[:,1],dipoles[:,2],tmp_igrf_dip)
        igrf_ECEF_dip[i,:,:] = tmp_igrf_ECEF_dip
    
    igrf_ECEF_stat = ned2ecefv(stations[:,0],stations[:,1],stations[:,2],igrf_stat)
    
 
    ndipoles, ncomp = dipoles.shape
    
    design_matrix_ECEF = np.empty(shape=[0, ndipoles,3])

    for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years of the DGRF models
        if nr_idx_DGRF[i] > 0:
            
            tmp_matrix_ECEF = mag_dipole_matrix(dipoles_ECEF,cell_sizes,np.squeeze(igrf_ECEF_dip[i,:,:]),stations_ECEF[first_idx_DGRF[i]: first_idx_DGRF[i] + nr_idx_DGRF[i],:])
            print(design_matrix_ECEF.shape, tmp_matrix_ECEF.shape)
            design_matrix_ECEF = np.append(design_matrix_ECEF, tmp_matrix_ECEF, axis=0)
    
    ano_design_matrix = (igrf_ECEF_stat[:,None,:] * design_matrix_ECEF).sum(-1) / np.sqrt((igrf_ECEF_stat[:,None,:]**2).sum(-1))
        
    return ano_design_matrix


def calculate_ano_design_matrix_sparse(stations,dipoles,cell_sizes,igrf_stat,igrf_dip, dist_thresh, max_data, first_idx_DGRF, nr_idx_DGRF):
    """Calculate the magnetic anomaly design matrix - use a sparse matrix with sensitivities only filled above a threshold 

    Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    cell_sizes : np.array(shape=(m,1))
        "Cell sizes" of the dipoles (to define weights in inversion)
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array(shape=(m,3))
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
    dist_thresh : <= 0.0 dense matrix; otherwise distance in m, where the matrix entries are ignored
    max_data : Max number of data points managed in blocks
    first_idx_DGRF : np.array
        index of the first data point used for specific DGRFs (associated with different reference years) 
    nr_idx_DGRF: np.array
        The number of data points used a specific DGRF
            
    Returns
    -------
    ano_design_matrix : scipy.sparse.csr_matrix(shape=(n,m))
        Each row gives the effect of all dipoles on a station, assuming a 
        source strength of one at the dipole.
    """
    
    # Shape assumptions:
    # stations nx3 (lon,lat, height (in meters))
    # dipoles nx3 (lon,lat, height (in meters))
    # IGRF is in NED
    stations_ECEF = ned2ecef(stations[:,0],stations[:,1],stations[:,2])
    dipoles_ECEF = ned2ecef(dipoles[:,0],dipoles[:,1],dipoles[:,2])
    
    igrf_ECEF_dip = np.zeros(igrf_dip.shape)
        
    for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years
        tmp_igrf_dip = np.squeeze(igrf_dip[i,:,:])
        tmp_igrf_ECEF_dip = ned2ecefv(dipoles[:,0],dipoles[:,1],dipoles[:,2],tmp_igrf_dip)
        igrf_ECEF_dip[i,:,:] = tmp_igrf_ECEF_dip

    igrf_ECEF_stat = ned2ecefv(stations[:,0],stations[:,1],stations[:,2],igrf_stat)
    
    N_stations,N_dipoles = stations_ECEF.shape[0],dipoles_ECEF.shape[0]
    ano_design_matrix = sparse.csr_matrix((N_stations,N_dipoles))     
    
    idx_b = 0   #finish the loop if =1
    block_col =  round(max_data/N_stations) #define the number of columns in a block    
    min_col = 0
    max_col = block_col

    #consider only values above threshold
    while idx_b == 0: #loop over all dipoles (columns)
    
        if max_col < N_dipoles:
            block_idx = np.arange(min_col, max_col)     
        else:
            block_idx = np.arange(min_col, N_dipoles)
            idx_b = 1
            max_col = N_dipoles
        
        r = stations_ECEF[:,None,:] - np.squeeze(dipoles_ECEF[None,block_idx,:]) #three components of distance of dipoles and measuring locations        
        r_val = np.sqrt((r**2).sum(-1))  
        
        used_val = r_val < dist_thresh #Find all values below a distance
                
        x_list, y_list  = np.where(used_val)  #row and colum number of nonzero tuples
        y_list = y_list + min_col
        
        
        for i in range (first_idx_DGRF.size): #Loop over the dipoles of the different reference years of the DGRF models
            if nr_idx_DGRF[i] > 0:
                idx_select = np.where((x_list >= first_idx_DGRF[i]) & (x_list < first_idx_DGRF[i] + nr_idx_DGRF[i]))
                x_list_select = x_list[idx_select]
                y_list_select = y_list[idx_select]
                
                #print(i)
                #print(r.shape, x_list_select.max(),  y_list_select.max(), N_stations,N_dipoles)
                
                #bring distance components into sparse (csr-format) matrices
                r_x = sparse.coo_matrix((r[x_list_select, y_list_select - min_col,0], (x_list_select, y_list_select)), shape=(N_stations,N_dipoles)).tocsr() 
                r_y = sparse.coo_matrix((r[x_list_select, y_list_select - min_col,1], (x_list_select, y_list_select)), shape=(N_stations,N_dipoles)).tocsr()
                r_z = sparse.coo_matrix((r[x_list_select, y_list_select - min_col,2], (x_list_select, y_list_select)), shape=(N_stations,N_dipoles)).tocsr()
                
                #From here on everything is sparse:        
                r_val = np.sqrt(r_x.power(2) + r_y.power(2) + r_z.power(2))
                        
                T1 = 3.0*(r_x.multiply(np.squeeze(igrf_ECEF_dip[i,:,0])) + r_y.multiply(np.squeeze(igrf_ECEF_dip[i,:,1])) + r_z.multiply(np.squeeze(igrf_ECEF_dip[i,:,2]))).multiply(r_val.power(-5))
             
        
                T2 = r_val.power(-3)
        
                del r_val
        
                T1 = T1.multiply(cell_sizes) #weight from the cell sizes
                T2 = T2.multiply(cell_sizes) #weight from the cell sizes

                design_matrix_x =  T1.multiply(r_x) - T2.multiply(np.squeeze(igrf_ECEF_dip[i,:,0]))
                design_matrix_y =  T1.multiply(r_y) - T2.multiply(np.squeeze(igrf_ECEF_dip[i,:,1]))
                design_matrix_z =  T1.multiply(r_z) - T2.multiply(np.squeeze(igrf_ECEF_dip[i,:,2]))
        
        
                del T1
                del T2

                del r_x
                del r_y
                del r_z
        
                design_matrix_x = design_matrix_x.transpose()
                design_matrix_y = design_matrix_y.transpose()
                design_matrix_z = design_matrix_z.transpose()
                
                tmp_design_matrix = (design_matrix_x.multiply(np.squeeze(igrf_ECEF_stat[:,0])) + design_matrix_y.multiply(np.squeeze(igrf_ECEF_stat[:,1])) + design_matrix_z.multiply(np.squeeze(igrf_ECEF_stat[:,2])))
        
                del design_matrix_x
                del design_matrix_y
                del design_matrix_z
        
                tmp_design_matrix = tmp_design_matrix.multiply(1.0/np.sqrt((igrf_ECEF_stat**2).sum(-1)))
        
                ano_design_matrix = ano_design_matrix + tmp_design_matrix.transpose()

                del tmp_design_matrix

        del r

        
        print("finished", max_col, "of", N_dipoles)
        print("mean number of entries per row: ", round(len(y_list)/(max_col-min_col)), "; total number:", N_stations)
        if y_list.any():
            print("Maximum value:", ano_design_matrix.max(),"\n")
                    
        
        min_col = max_col
        max_col = max_col + block_col 
        
    
    return ano_design_matrix


def determine_active_equivalent_sources(stations,dipoles,radius, quad_thres, pts_thres, max_data):

    """Determine if an equivalent source is active or not - based on how data points are distributed around the equivalent sources in terms in their radial distributions and their total numbers

    Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    radius : radius in m, in which the data points around the equivalent sources are considered
    quad_thres : mimimum number of quadrants that have to be filled with data around an eqyuivalent source point
    pts_thresd: minimum number of data points that have to present in the quadrants
    max_data : Max number of data points managed in blocks

    
    return 
    
    fac_equi_source: vector (shape= n) that indicates if equivalent sources are considered (==1) or ignored (== 0) after modifications

    """
    
    # Shape assumptions:
    # stations nx3 (lon,lat, height (in meters))
    # dipoles nx3 (lon,lat, height (in meters))
    # IGRF is in NED
    stations_ECEF = ned2ecef(stations[:,0],stations[:,1],stations[:,2])
    dipoles_ECEF = ned2ecef(dipoles[:,0],dipoles[:,1],dipoles[:,2])
    
    N_stations,N_dipoles = stations_ECEF.shape[0],dipoles_ECEF.shape[0]
    vec_quad1 = np.zeros(N_dipoles)
    vec_quad2 = np.zeros(N_dipoles)
    vec_quad3 = np.zeros(N_dipoles)
    vec_quad4 = np.zeros(N_dipoles)
    
    idx_b = 0   #finish the loop if =1
    block_col =  round(max_data/N_stations) #define the number of columns in a block
    min_col = 0
    max_col = block_col

    #consider only values above threshold
    while idx_b == 0: #loop over all dipoles (columns)
    
        if max_col < N_dipoles:
            block_idx = np.arange(min_col, max_col)     
        else:
            block_idx = np.arange(min_col, N_dipoles)
            idx_b = 1
            max_col = N_dipoles
            
        r = stations_ECEF[:,None,0:2] - np.squeeze(dipoles_ECEF[None,block_idx,0:2]) #three components of distance of dipoles and measuring locations        
        r_val = np.sqrt((r**2).sum(-1))
        used_val = r_val < radius #Find all values below the distance
        
        del r_val 
        
        x_list, y_list  = np.where(used_val)  #row and colum number of nonzero tuples
        dat = np.ones(x_list.shape)
        
        #bring distance components into sparse (csr-format) matrices
        act_dipoles_quad1 = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr() 
        act_dipoles_quad2 = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr()
        act_dipoles_quad3 = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr() 
        act_dipoles_quad4 = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr() 
                
        #Find the data in the four quadrants
        r_0 = np.squeeze(r[:,:,0])
        r_1 = np.squeeze(r[:,:,1]) 
        
        del r 
        
        used_val = r_0 < 0
        x_list, y_list  = np.where(used_val)  
        dat = np.ones(x_list.shape)
        x_neg = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr()
        
        used_val = r_0 > 0
        x_list, y_list  = np.where(used_val)
        dat = np.ones(x_list.shape)
        x_pos = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr()
        
        used_val = r_1 < 0
        x_list, y_list  = np.where(used_val)
        dat = np.ones(x_list.shape)
        y_neg = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr()
        
        used_val = r_1 > 0
        x_list, y_list  = np.where(used_val)
        dat = np.ones(x_list.shape)
        y_pos = sparse.coo_matrix((dat, (x_list, y_list + min_col)), shape=(N_stations,N_dipoles)).tocsr()

        #multiply the sparse matrices with distances and angle distributions
        act_dipoles_quad1 = act_dipoles_quad1.multiply(x_pos.multiply(y_pos))
        act_dipoles_quad2 = act_dipoles_quad2.multiply(x_pos.multiply(y_neg))
        act_dipoles_quad3 = act_dipoles_quad3.multiply(x_neg.multiply(y_pos))
        act_dipoles_quad4 = act_dipoles_quad4.multiply(x_neg.multiply(y_neg))
        
        #sum the points in quadrants to vectors
        vec_quad1 = vec_quad1 + act_dipoles_quad1.sum(axis=0)
        vec_quad2 = vec_quad2 + act_dipoles_quad2.sum(axis=0)
        vec_quad3 = vec_quad3 + act_dipoles_quad3.sum(axis=0)
        vec_quad4 = vec_quad4 + act_dipoles_quad4.sum(axis=0)
                        
        print("finished", max_col, "of", N_dipoles)
        
        min_col = max_col
        max_col = max_col + block_col 
        
 
    fac_equi_source = np.zeros((1,N_dipoles))
        
    #Select the active equivalent sources based on criteria
        #Select only equivalent sources with enough data points
    not_used_val = vec_quad1 <= pts_thres
    vec_quad1[not_used_val] = 0
    not_used_val = vec_quad2 <= pts_thres
    vec_quad2[not_used_val] = 0
    not_used_val = vec_quad3 <= pts_thres
    vec_quad3[not_used_val] = 0
    not_used_val = vec_quad4 <= pts_thres
    vec_quad4[not_used_val] = 0
    
        #identify the number of quadrants with data points
    used_val1 = vec_quad1 >= 1
    used_val2 = vec_quad1 >= 1
    used_val3 = vec_quad3 >= 1
    used_val4 = vec_quad4 >= 1
        
    if quad_thres == 1:
        final_used = np.logical_or(np.logical_or(used_val1,used_val2),np.logical_or(used_val3,used_val4))
        final_used = np.squeeze(final_used)
        fac_equi_source[final_used] = 1
    elif quad_thres == 2:
        final_used = np.logical_or(np.logical_and(used_val1,used_val4),np.logical_and(used_val2,used_val3)) #quadrants have to be mirrowed
        final_used = np.squeeze(final_used)
        fac_equi_source[final_used] = 1        
    elif quad_thres == 3:
        final_used = np.logical_or(np.logical_or(np.logical_and(used_val1,used_val2,used_val3),np.logical_and(used_val1,used_val2,used_val4)),np.logical_or(np.logical_and(used_val1,used_val3,used_val4), np.logical_and(used_val2,used_val3,used_val4)))
        final_used = np.squeeze(final_used)
        fac_equi_source[final_used] = 1
    else :
        final_used = np.logical_and(np.logical_and(used_val1,used_val2),np.logical_and(used_val3,used_val4))
        final_used = np.squeeze(final_used)
        fac_equi_source[final_used] = 1
        
    return fac_equi_source

def equivalent_source_block_inversion(block_definition,dipoles,aeromag,igrf_NED_dip,igrf_NED_stat,
                                    far_field = None,rcond=0.1,verbose=False):
    """Invert for equivalent sources within a block
    
    Within each block, the strength of the equivalent sources (dipoles) is estimated using
    only the magnetic field data (`aeromag`) within that same block. A far_field effect
    can be subtracted first for repeated inversions.

    Parameters
    ----------
    block_definition : tuple
        Gives the southern, northern, eastern and western boundaries of the blocks. Note that 
        southern and northern are merely lists, whereas eastern and western are dicts of list
    dipoles : np.array(shape=(n,3))
        Gives the longitude, latitude and height (in m) of each dipole
    aeromag : np.array(shape=(n,4))
        Gives the longitude, latitude, height (in m) and magnetic anomaly for the measurements
    igrf_NED_dip : np.array(shape=(n,3))
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
    igrf_NED_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    far_field : np.array(),optional
        Gives the far-field correction (i.e. the magnetic effect of all neighboring blocks).
        Needs to be the same length as `aeromag`
    rcond : float, optional
        The relative accuracy used to solve for the equivalent sources. See `np.linalg.lstsq`
    verbose : bool, optional
        Output additional information during run-time.
    
    """

    southern,northern,eastern,western = block_definition
    
    block_dip = np.array(assign_block(dipoles[:,0],dipoles[:,1],southern,northern,eastern,western))
    block_stat = np.array(assign_block(aeromag[:,0],aeromag[:,1],southern,northern,eastern,western))
    
    if far_field is None:
        far_field = np.zeros(aeromag.shape[0])

    equivalent_sources = np.zeros(dipoles.shape[0])
    predicted = np.zeros(aeromag.shape[0])
    for i,j in BlockIterator(southern,eastern):
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)

        if in_block_dip.sum() == 0:
            continue
        elif in_block_stat.sum() == 0:
            continue

        if verbose:
            print(i,j,end = " ")

        ano_design_matrix = calculate_ano_design_matrix(aeromag[in_block_stat],dipoles[in_block_dip],
                                                        igrf_NED_stat[in_block_stat],igrf_NED_dip[in_block_dip])
        temp = np.linalg.lstsq(ano_design_matrix,aeromag[in_block_stat,3]-far_field[in_block_stat],rcond=rcond)[0]
        equivalent_sources[in_block_dip] = temp
        predicted[in_block_stat] = ano_design_matrix.dot(temp)
    return equivalent_sources,predicted

def predict_at_auxilliary(block_definition,dipoles,auxiliary,igrf_NED_dip,igrf_NED_aux,equivalent_sources,verbose=False):
    """Predict magnetic field at arbitrary locations
    This will only calculate the magnetic effect of sources on stations in the same block!
    No far-field will be calculated!
    """
    ## Visualize the predicted field
    southern,northern,eastern,western = block_definition

    block_dip = np.array(assign_block(dipoles[:,0],dipoles[:,1],southern,northern,eastern,western))
    block_aux = np.array(assign_block(auxiliary[:,0],auxiliary[:,1],southern,northern,eastern,western))
    
    auxiliary_field = np.zeros(auxiliary.shape[0])
    for i,j in BlockIterator(southern,eastern):
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        in_block_aux = (block_aux[0] == i) & (block_aux[1] == j)

        if in_block_dip.sum() == 0:
            continue
        
        if verbose:
            print(i,j,end = " ")

        ano_design_matrix = calculate_ano_design_matrix(auxiliary[in_block_aux],dipoles[in_block_dip.flatten()],
                                                        igrf_NED_aux[in_block_aux],igrf_NED_dip[in_block_dip.flatten()])
        auxiliary_field[in_block_aux] = ano_design_matrix.dot(equivalent_sources[in_block_dip])
    return auxiliary_field


#Building a differential operator matrix
def build_1st_order_diffmatrix(nx, ny):

    """To calculate the difference matrix for a first order regularization in an inversion. The model cells are assumed to be arranged in a regular array of nx, ny number of cells in x and y-direction. The resulting matrices are associated with the differences in x- and y-direction and the complete difference matrix can be easily determined out of them (see e.g. PhD from Thomas Günther).  
    """

    #prepare difference operator in x-direction
    nxy1 = (nx-1)*ny

    raw = np.empty(2*nxy1)
    col = np.empty(2*nxy1)
    val = np.empty(2*nxy1)

    t= 0

    for k in range(ny):
        for j in range(nx-1):
        
            raw[t] = t
            raw[t + nxy1] = t    
        
            col[t] = k*nx + j
            col[t + nxy1] = k*nx + (j+1)
            
    
            val[t]        =  1.0
            val[t + nxy1] = -1.0
        
            t= t+1

    #print(x_pos)        
    #print(y_pos)
    #print(val)
   
    C_x = sparse.coo_matrix((val, (raw, col)), shape=((nx-1)*ny,nx*ny)).tocsr()  #add -1 and 1 values to sparse difference operator matrix in x-direction 
    #C_x_dense = C_x.todense()
    #print(C_x_dense)

    
    #####################################
    #prepare difference operator in y-direction
    nxy1 = nx*(ny-1)
    
    raw = np.empty(2*nxy1)
    col = np.empty(2*nxy1)
    val = np.empty(2*nxy1)

    t= 0

    for k in range(ny-1):
        for j in range(nx):
        
            raw[t] = t
            raw[t + nxy1] = t    

            col[t] = k*nx + j
            col[t + nxy1] = (k+1)*nx + j

            val[t]        =  1.0
            val[t + nxy1] = -1.0
        
            t= t+1

    #print(x_pos)        
    #print(y_pos)
    #print(val)
   
    C_y = sparse.coo_matrix((val, (raw, col)), shape=((ny-1)*nx,nx*ny)).tocsr()  #add -1 and 1 values to sparse difference operator matrix in x-direction 
    #C_y_dense = C_y.todense()
    #print(C_y_dense)
    
    return C_x, C_y

def rewweight_diffmatrix(C, weight):
    
    """change the weight in the original difference matrix C. Weighting is performed by using pre-calculated weights for all inversion cells. All non-zero entries of a row are re-filled by weight values averaged by the number of cells involved in the corresponding difference (and present as non zero elements in the row)   
    """
        
    nrow_C, ncol_C = np.shape(C)
    
    (row,col,val) = sparse.find(C)
    
    for i in range(nrow_C):
        
        idx = np.where(row == i)
           
        for j in range(len(idx[0])):
            
            div = len(idx[0])
                      
            for k in range(len(idx[0])):
                if val[idx[0][j]] < 0.0:
                    val[idx[0][j]] = val[idx[0][j]] - weight[col[idx[0][k]]]/div
                else:
                    val[idx[0][j]] = val[idx[0][j]] + weight[col[idx[0][k]]]/div
                    
            if val[idx[0][j]] < 0.0:
                val[idx[0][j]] = val[idx[0][j]] + 1.0
            else:
                val[idx[0][j]] = val[idx[0][j]] - 1.0
                    
    C = sparse.coo_matrix((val, (row, col)), shape=C.shape).tocsr() 
    
    return C
                                                       