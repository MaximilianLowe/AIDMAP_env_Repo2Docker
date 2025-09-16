import numpy as np

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

# -*- coding: utf-8 -*-

#import packages
import numpy as np
import scipy.sparse as sparse


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

    R=np.zeros((len(lon), 3, 3))
    R[:,:,0]=b_north
    R[:,:,1]=b_east
    R[:,:,2]=b_down

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        print("Error")
        pass
   
    # WS: This should give the same result as the for loop below
    vector_field = np.einsum('...kij,...kj->...ki',R_inv,ecef_vector_field)

    return vector_field

def mag_dipole_matrix(dipoles,magnetizations,stations,**kwargs):
    """Calculate the magnetic effect of a set of dipoles on a set of stations.
    This is a calculation in cartesian coordinates. 
    To conserve a bit of memory, this function splits the data into partitions.
    
    Parameters
    ----------
    dipoles : np.array
        Shape is (n_dipoles,3) and contains the XYZ coordinates of each dipole
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
        #print(np.shape(T1), np.shape(T2), np.shape(r), np.shape(magnetizations))

        design_matrix[ix] =  (T1[:,:,None] * r - T2[:,:,None] * magnetizations[None, :,:])
    return design_matrix

def calculate_kernel_matrix_spatial(loni,lati,station_height,dipole_depth,dipole_magnetization_NED):
    """
    Calculate the magnetic effect of a collection of dipoles at stations at coincident locations

    Parameters
    ----------
    lon : np.array
        Vector of the longitudes of the measurements and the dipoles (coincidence assumed).
    lat : np.narray
        Vector of the latitudes
    station_height:float
        The (constant!) height of the stations. In meters!
    Dipole_depth: float
        The (constant!) depth of the dipoles. Positive down and in meters!
    """
    
    #loni, lati = np.meshgrid(lon, lat)
    
    # TODO(Judith):
    # Convert from lon,lat,height/depth to Earth-Centered-Earth-Fixed (ECEF) coordinate system
    stations_ECEF = ned2ecef(loni,lati,station_height) 
    dipoles_ECEF = ned2ecef(loni,lati,-dipole_depth)
    # TODO(Judith): Convert magnetization from NED to ECEF coordinate basis.
    magnetization_ECEF = ned2ecefv(loni,lati,-dipole_depth,dipole_magnetization_NED)
    # Call mag_dipole_matrix
    design_matrix_ECEF = mag_dipole_matrix(dipoles_ECEF,magnetization_ECEF,stations_ECEF)
    # TODO(Judith): Convert from ECEF coordinate system back to NED
    #design_matrix_NED = ecef2nedv(lon,lat,station_height,design_matrix_ECEF)

    design_matrix_NED = np.zeros((len(loni), len(loni), 3))
#     for a in range(len(lon)):
#         design_matrix_NED[a,:,:] = ecef2nedv(loni,lati,station_height,design_matrix_ECEF[a, :, :])
    design_matrix_NED = ecef2nedv(loni,lati,station_height,design_matrix_ECEF)
    # Keep only the vertical component
    design_matrix_D = design_matrix_NED[:,:,2]
    return design_matrix_D

def build_1st_order_diffmatrix(nx, ny):

    """To calculate the difference matrix for a first order regularization in an inversion. 
    The model cells are assumed to be arranged in a regular array of nx, ny number of cells in x and y-direction. 
    The resulting matrices are associated with the differences in x- and y-direction and the complete difference matrix 
    can be easily determined out of them (see e.g. PhD from Thomas Günther).  
    """

    #prepare difference operator in x-direction
    nxy1 = (nx-1)*ny

    row = np.empty(2*nxy1)
    col = np.empty(2*nxy1)
    val = np.empty(2*nxy1)

    t= 0

    for k in range(ny):
        for j in range(nx-1):
        
            row[t] = t
            row[t + nxy1] = t    
        
            col[t] = k*nx + j
            col[t + nxy1] = k*nx + (j+1)
            
    
            val[t]        =  1.0
            val[t + nxy1] = -1.0
        
            t= t+1
   
    C_x = sparse.coo_matrix((val, (row, col)), shape=((nx-1)*ny,nx*ny)).tocsr()  #add -1 and 1 values to sparse difference operator matrix in x-direction 
    
    #####################################
    #prepare difference operator in y-direction
    nxy1 = nx*(ny-1)
    
    row = np.empty(2*nxy1)
    col = np.empty(2*nxy1)
    val = np.empty(2*nxy1)

    t= 0

    for k in range(ny-1):
        for j in range(nx):
        
            row[t] = t
            row[t + nxy1] = t    

            col[t] = k*nx + j
            col[t + nxy1] = (k+1)*nx + j

            val[t]        =  1.0
            val[t + nxy1] = -1.0
        
            t= t+1

    C_y = sparse.coo_matrix((val, (row, col)), shape=((ny-1)*nx,nx*ny)).tocsr()  #add -1 and 1 values to sparse difference operator matrix in x-direction 
    return C_x, C_y

def build_1st_order_central_differences(nx,ny,direction=0):
    """Build matrix for first order differences which exist everywhere
    (not just the internal part). This is not numerically stable, but makes
    it easier to locate where the derivatves are actually calculated
    """
    def get_ix(i,j):
        return i*nx + j
    row = []
    col = []
    val = []
    for i in range(0,ny):
        for j in range(0,nx):
            ix = i*nx + j
            if direction == 1:
                ## x derivatives
                if j==0 or j==nx-1:
                    continue
                ix_l = get_ix(i,j-1)
                ix_r = get_ix(i,j+1)
                row.append(ix)
                row.append(ix)
                col.append(ix_l)
                col.append(ix_r)
                val.append(-0.5)
                val.append(0.5)
            elif direction == 0:
                if i==0 or i==ny-1:
                    continue
                ## y derivatives
                ix_l = get_ix(i-1,j)
                ix_r = get_ix(i+1,j)
                row.append(ix)
                row.append(ix)
                col.append(ix_l)
                col.append(ix_r)
                val.append(-0.5)
                val.append(0.5)
    ## Calculate edges
    if direction == 0:
        for j in range(0,nx):
            ix = get_ix(0,j)
            ix_2 = get_ix(1,j)
            row.append(ix)
            row.append(ix)
            col.append(ix)
            col.append(ix_2)
            val.append(-1)
            val.append(1)
            ## Other edge
            ix = get_ix(ny-1,j)
            ix_2 = get_ix(ny-2,j)
            row.append(ix)
            row.append(ix)
            col.append(ix)
            col.append(ix_2)
            val.append(1)
            val.append(-1)
    elif direction == 1:
        for i in range(0,ny):
            ix = get_ix(i,0)
            ix_2 = get_ix(i,1)
            row.append(ix)
            row.append(ix)
            col.append(ix)
            col.append(ix_2)
            val.append(-1)
            val.append(1)
            ## Other edge
            ix = get_ix(i,nx-1)
            ix_2 = get_ix(i,nx-2)
            row.append(ix)
            row.append(ix)
            col.append(ix)
            col.append(ix_2)
            val.append(1)
            val.append(-1)
    C_x = sparse.coo_matrix((val, (row, col)), shape=(nx*ny,nx*ny)).tocsr()
    return C_x


def calculate_ano_design_matrix(stations,dipoles,igrf_stat,igrf_dip):
    """Calculate the magnetic anomaly design matrix

    Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array(shape=(m,3))
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down
            
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
    igrf_ECEF_dip = ned2ecefv(dipoles[:,0],dipoles[:,1],dipoles[:,2],igrf_dip)
    igrf_ECEF_stat = ned2ecefv(stations[:,0],stations[:,1],stations[:,2],igrf_stat)
    
    design_matrix_ECEF = mag_dipole_matrix(dipoles_ECEF,igrf_ECEF_dip,stations_ECEF)
    
    ano_design_matrix = (igrf_ECEF_stat[:,None,:] * design_matrix_ECEF).sum(-1) / np.sqrt((igrf_ECEF_stat[:,None,:]**2).sum(-1))
    
    return ano_design_matrix

def calculate_ano_design_matrix_multi_year(stations,dipoles,igrf_stat,igrf_dip,station_epochs):
    """Calculate the magnetic anomaly design matrix

    Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    igrf_stat : np.array(shape=(n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array(shape=(y,m,3))
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down. A new igrf is given for each epoch
    station_epochs: np.array(shape(n),dtype=int)
        Gives the epoch of the IGRF to use for this station
    Returns
    -------
    ano_design_matrix : np.array(shape=(n,m))
        Each row gives the effect of all dipoles on a station, assuming a 
        source strength of one at the dipole.
    """

    used_epochs = np.unique(station_epochs)
    ano_design_matrix = np.zeros((stations.shape[0],dipoles.shape[0]))
    for u in  used_epochs:
        sel = station_epochs == u
        ano_design_matrix[sel] = calculate_ano_design_matrix(stations[sel],dipoles,igrf_stat[sel],igrf_dip[u,:,:])
    
    return ano_design_matrix


def calculate_ano_design_matrix_all_years(stations,dipoles,igrf_stat,igrf_dip):
    """Calculate the magnetic anomaly design matrix

    This calculates the matrix for all epochs at the corresponding stations 
    (unlike calculate_ano_design_matrix_multi_year) 
        Parameters
    ----------
    stations : np.array(shape=(n,3))
        Longitude, latitude and height (in meters) of prediction locations
    dipoles : np.array(shape=(m,3))
        Longitude, latitude and height (in meters) of dipoles
    igrf_stat : np.array(shape=(y,n,3))
        Magnetic field for magnetic measurement. This is used to calculate the magnetic anomaly.
    igrf_dip : np.array(shape=(y,m,3))
        Gives the strength of the magnetizing field (typically IGRF) at the dipole locations. The
        order of the coordinates is North-East-Down. A new igrf is given for each epoch
    Returns
    -------
    ano_design_matrix : np.array(shape=(y,n,m))
        Each row gives the effect of all dipoles on a station, assuming a 
        source strength of one at the dipole.
    """

    ano_design_matrix = np.zeros((igrf_stat.shape[0],stations.shape[0],dipoles.shape[0]))
    for i in  range(igrf_stat.shape[0]):
        ano_design_matrix[i] = calculate_ano_design_matrix(stations,dipoles,igrf_stat[i],igrf_dip[i])
    
    return ano_design_matrix