import numpy as np
import subprocess
import io
import os
import tempfile
      
def masspoint_calc_FTG_2(lon,lat,depths,mass,lons,lats,heights,**kwargs):
    """Calculate gravity field from a collection of point masses
    
    lon, lat, depths, mass have arbitrary but identical shape
    lons, lats and heights are vectors
    
    Units
    ---
    depths are POSITIVE DOWN in km
    heights is POSITIVE UP in m
    mass is in kg
    returns gravity in SI units
    
    kwargs
    ---
    calc_mode can be grad, grav or potential
    
    Returns
    ---
    The sensitivity matrix -> effect of each mass on all of the stations 
    """
    calc_mode = kwargs.get("calc_mode","grad")
    G = 6.67428e-11
    lon = lon[...,None]
    lat = lat[...,None]
    depths =depths[...,None]
    mass = mass[...,None]
    dLon = lon - lons
    coslat1 = np.cos(lat/180.0*np.pi)
    cosPsi = (coslat1 * np.cos(lats/180.0*np.pi) * np.cos(dLon/180.0*np.pi) +
            np.sin(lat/180.0*np.pi) * np.sin(lats/180.0*np.pi))
    cosPsi[cosPsi>1] = 1
    
    KPhi = (np.cos(lats/180.0*np.pi) * np.sin(lat/180.0*np.pi) - 
            np.sin(lats/180.0*np.pi) * coslat1 * np.cos(dLon/180.0*np.pi))
    

    rStat = (heights + 6371000.0)
    rTess = (6371000.0 - 1000.0*depths)
    spherDist = np.sqrt(rTess**2 + rStat**2
                            - 2 * rTess * rStat*cosPsi)

    dx = KPhi * rTess
    dy = rTess * coslat1 * np.sin(dLon/180.0*np.pi)
    dz = rTess * cosPsi - rStat
    if calc_mode == "potential":
        T = np.zeros(spherDist.shape+(1,))
        T[...,0] = G * mass / spherDist 
    elif calc_mode == "grav":
        T = np.zeros(spherDist.shape+(1,))
        T[...,0] = dz / (spherDist**3) * mass * G
    else:
        T = np.zeros(spherDist.shape+(6,))
        T[...,0] = ((3.0*dy*dy/(spherDist**5) - 1.0/(spherDist**3))*mass*G)
        T[...,1] = ((3.0*dx*dy/(spherDist**5))*mass*G)
        T[...,2] = ((3.0*dy*dz/(spherDist**5))*mass*G)
        T[...,3] = ((3.0*dx*dx/(spherDist**5) - 1.0/(spherDist**3))*mass*G)
        T[...,4] = ((3.0*dx*dz/(spherDist**5))*mass*G)
        T[...,5] = ((3.0*dz*dz/(spherDist**5)- 1.0/(spherDist**3))*mass*G)

    return T  


def memory_save_masspoint_calc_FTG_2(lon,lat,depths,mass,lons,lats,heights,**kwargs):
    """Calculate gravity effect of a collection of point masses in chunks to save memory.
    max_size gives the maximum size in bytes used for the sensitivity matrix
    See docu for masspoint_calc_FTG_2
    """
    N = lon.size
    M = lons.size
    calc_mode = kwargs.get("calc_mode","grad")
    max_size = kwargs.get("max_size",1000000000) # in bytes
    verbose = kwargs.get("verbose",False)
    
    if calc_mode == "potential":
        T = np.zeros(lons.shape+(1,))
    elif calc_mode == "grav":
        T = np.zeros(lons.shape+(1,))
    else:
        T = np.zeros(lons.shape+(6,))
    
    partitions = np.ceil(8 * N * M / max_size).astype(int)
    if verbose:
        print('Number of partitions ',partitions)
    
    ixs = np.array_split(np.arange(M,dtype=int),partitions)
    for ix in ixs:
        design_matrix = masspoint_calc_FTG_2(lon,lat,depths,mass,lons[ix],lats[ix],heights[ix],**kwargs)
        T[ix,:] += design_matrix.sum(tuple(range(design_matrix.ndim-2)))
        if verbose:
            print('Parition  done')
    return T


def tess_to_pointmass(lon,lat,dx,dy,tops,bottoms,dens,hsplit=1,vsplit=1):
    """Convert a single tesseroid into a set of point masses
    The tesseroid will be split horizontally and vertically according to hsplit and vsplit
    """
    depths = np.zeros((vsplit,hsplit,hsplit))
    lon0 = np.zeros(depths.shape)
    lat0 = np.zeros(depths.shape)
    masses = np.zeros(depths.shape)
    dlon = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dx
    dlat = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dy
    for k in range(vsplit):
        if k==0:
            top = tops
        else:
            top = (bottoms-tops)/(1.0*vsplit)*k + tops
        if k==vsplit-1:
            bot = bottoms
        else:
            bot = (bottoms-tops)/(1.0*vsplit)*(k+1) + tops
        r_top = 6371-top
        r_bot = 6371-bot
        r_term = (r_top**3-r_bot**3)/3.0
        for i in range(hsplit):
            for j in range(hsplit):
                lon0[k,i,j] = lon + dlon[j]
                lat0[k,i,j] = lat + dlat[i]
                depths[k,i,j] = 0.5 * (top+bot)
                surface_Area = dx*(np.pi/(180.0*hsplit)) *np.cos(
                    lat0[k,i,j]/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
                masses[k,i,j] = surface_Area*dens*r_term*1e9

    return lon0,lat0,depths,masses

def tesses_to_pointmass(lon,lat,dx,dy,tops,bottoms,dens,hsplit=1,vsplit=1):
    """Convert several equal-sized tesseroids into a set of point masses
    """
    depths = np.zeros((vsplit,hsplit,hsplit)+lon.shape)
    lon0 = np.zeros(depths.shape)
    lat0 = np.zeros(depths.shape)
    masses = np.zeros(depths.shape)
    dlon = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dx
    dlat = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dy
    for k in range(vsplit):
        if k==0:
            top = tops
        else:
            top = (bottoms-tops)/(1.0*vsplit)*k + tops
        if k==vsplit-1:
            bot = bottoms
        else:
            bot = (bottoms-tops)/(1.0*vsplit)*(k+1) + tops
        r_top = 6371-top
        r_bot = 6371-bot
        r_term = (r_top**3-r_bot**3)/3.0
        for i in range(hsplit):
            for j in range(hsplit):
                lon0[k,i,j] = lon + dlon[j]
                lat0[k,i,j] = lat + dlat[i]
                depths[k,i,j] = 0.5 * (top+bot)
                surface_Area = dx*(np.pi/(180.0*hsplit)) *np.cos(
                    lat0[k,i,j]/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
                masses[k,i,j] = surface_Area*dens*r_term*1e9

    return lon0,lat0,depths,masses

def tesses_to_pointmass_max_size(lon,lat,dx,dy,tops,bottoms,dens,max_horz,max_vert,verbose=False):
    """Convert several tesseroids into a set of point masses defined by maximum size
    """
    n_pointmass = 0
    dz = bottoms - tops
    dep_centers = 0.5 * (tops+bottoms)
    
    for i in range(len(lon)):
        n_pointmass += np.ceil(dx[i]/max_horz) * np.ceil(dy[i]/max_horz) * np.ceil(dz[i] / max_vert)
    
    n_pointmass = int(n_pointmass)
    if verbose:
        print('Using %d point masses to represent tesseroid' % n_pointmass)
    lon0 = np.zeros(n_pointmass)
    lat0 = np.zeros(n_pointmass)
    depths = np.zeros(n_pointmass)
    masses = np.zeros(n_pointmass)
    
    counter = 0
    
    for i in range(len(lon)):
        lon_split =  np.ceil(dx[i]/max_horz)
        lat_split =  np.ceil(dy[i]/max_horz)
        dep_split =  np.ceil((bottoms[i]-tops[i]) / max_vert)
        loc_lon = (2*np.arange(0,lon_split,1)-lon_split+1)/(2.0*lon_split) * dx[i] + lon[i]
        loc_lat = (2*np.arange(0,lat_split,1)-lat_split+1)/(2.0*lat_split) * dy[i] + lat[i]
        loc_dep = (2*np.arange(0,dep_split,1)-dep_split+1)/(2.0*dep_split) * dz[i] + dep_centers[i]
        loni,lati,depi = np.meshgrid(loc_lon,loc_lat,loc_dep)
        lon0[counter:counter+loni.size] = loni.flat
        lat0[counter:counter+loni.size] = lati.flat
        depths[counter:counter+loni.size] = depi.flat
        r = 6371-depi.flatten()
        r_term = dz[i]/dep_split/3 * (3*r**2 + (dz[i]/dep_split)**2)
        surface_Area = dx[i]*(np.pi/(180.0*lon_split)) *np.cos(lati.flatten()/180.0*np.pi) * 2 * np.sin(dy[i]/(360.0*lat_split)*np.pi)
        masses[counter:counter+loni.size] = surface_Area*dens[i]*r_term * 1e9
        counter = counter + loni.size
    return lon0,lat0,depths,masses

def pillar_to_pointmass(lon,lat,depths,dx,dy,dz,hsplit=1,vsplit=1):
    """Convert a vertical column (pillar) to point masses
    lon,lat,depths refer to the lllcorner, such that each cell goes from lon to lon + dx, lat to lat+dy 
    and depths to depths+dz
    depths are positive down and in km.
    
    Output shape
    lon_0,lat_0 : (1,1,hsplit,hsplit)
    depths_pm: (nz,vsplit,1,1)
    volume: (nz,vsplit,hsplit,hsplit)
    """
    # Use existing function to determine horizontal location of point masses
    lon0,lat0,_,_ = tess_to_pointmass(lon+0.5*dx,lat+0.5*dy,dx,dy,depths[0],depths[0]+dz,1.0,hsplit,vsplit=1)
    lon0 = lon0[None,:,:,:]
    lat0 = lat0[None,:,:,:]
    depths_pm = depths[:,None,None,None] + np.arange(1,vsplit+1)[None,:,None,None]/(vsplit+1.) * dz
    surface_area =dx*(np.pi/(180.0*hsplit)) *np.cos(lat0/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
    volume = (6371000.0 - 1000.0*depths_pm)**2 * dz / vsplit * 1000 * surface_area
    
    return lon0,lat0,depths_pm,volume

def S_einsum(n,x,y):
    k = np.arange(1,n,1)
    Tx = np.cos(np.einsum('...,i',np.arccos(x),k))
    Ty = np.cos(np.einsum('...,i',np.arccos(y),k))
    x_ax = list(range(x.ndim))+[x.ndim+y.ndim]
    y_ax = list(range(x.ndim,x.ndim+y.ndim))+[x.ndim+y.ndim]
    
    return (2.0*np.einsum(Tx,x_ax,Ty,y_ax) + 1)/n

    
def cheb_nodes(n):
    return np.cos((2.0*np.arange(1,n+1)-1)*np.pi/(2.0*n))

def pillar_to_pointmass_chebby(lon,lat,depths,dx,dy,dz,cheb_order,hsplit=1,vsplit=1,density=None):
    """Convert a vertical column (pillar) to point masses at the chebyshev nodes of given order
    Output shape
    lon_0,lat_0 : (1,1,hsplit,hsplit)
    depths_pm: (cheb_order,1,1,1)
    volume_chebby: (cheb_order,1,hsplit,hsplit)
    """
    # Use existing function to determine horizontal location of point masses
    lon0,lat0,_,_ = tess_to_pointmass(lon+0.5*dx,lat+0.5*dy,dx,dy,depths[0],depths[0]+dz,1.0,hsplit,vsplit=1)
    lon0 = lon0[None,:,:,:]
    lat0 = lat0[None,:,:,:]
    depths_pm = depths[:,None,None,None] + np.arange(1,vsplit+1)[None,:,None,None]/(vsplit+1.) * dz
    surface_area =dx*(np.pi/(180.0*hsplit)) *np.cos(lat0/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
    volume = (6371000.0 - 1000.0*depths_pm)**2 * dz/vsplit * 1000 * surface_area
    if density is None:
        density = np.ones(len(depths))
    mass = volume * density[:,None,None,None]
    
    ym = cheb_nodes(cheb_order)
    mid = 0.5 * (depths_pm.min() + depths_pm.max())
    pillar_length = (depths_pm.max()-depths_pm.min())
    depths_pm_scale = (depths_pm - mid) * 2 / pillar_length
    depths_pm_scale = np.clip(depths_pm_scale,-1,1)
    temp = (S_einsum(cheb_order,ym,depths_pm_scale) * mass)
    mass_chebby = temp.sum((1,2))
    
    depths_chebby = ym * pillar_length/2. + mid
    
    return lon0,lat0,depths_chebby[:,None,None,None],mass_chebby[:,None,:,:]


def grid_to_pointmasses(lonGrid,latGrid,top_grid,bottom_grid,dens_grid,
                          hsplit=1,vsplit=1):
    """Convert a tesseroid layer grid to a set of point masses
    The layer is described by top and bottom and a laterally varying density value
    Depths are in KILOMETERS (positive DOWN)
    """
    ny,nx = lonGrid.shape[0],lonGrid.shape[1]
    dx = np.abs(lonGrid[0,1]-lonGrid[0,0])
    dy = np.abs(latGrid[1,0]-latGrid[0,0])
    lon0 = np.zeros((vsplit,hsplit*ny,hsplit*nx))
    lat0 = np.zeros((vsplit,hsplit*ny,hsplit*nx))
    depths = np.zeros(lon0.shape)
    masses = np.zeros(lat0.shape)
    thick = np.zeros(lat0.shape)
    r_top = 6371-top_grid
    r_bot = 6371-bottom_grid
    r_term = (r_top**3-r_bot**3)/3.0
    dlon = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dx
    dlat = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dy
    for k in range(vsplit):
        if k==0:
            top = top_grid
        else:
            top = (bottom_grid-top_grid)/(1.0*vsplit)*k + top_grid
        if k==vsplit-1:
            bot = bottom_grid
        else:
            bot = (bottom_grid-top_grid)/(1.0*vsplit)*(k+1) + top_grid
        r_top = 6371-top
        r_bot = 6371-bot
        r_term = (r_top**3-r_bot**3)/3.0
        for i in range(hsplit):
            for j in range(hsplit):
                lon0[k,i::hsplit,j::hsplit] = lonGrid + dlon[j]
                lat0[k,i::hsplit,j::hsplit] = latGrid + dlat[i]
                depths[k,i::hsplit,j::hsplit] = 0.5 * (top+bot)
                surface_Area = dx*(np.pi/(180.0*hsplit)) *np.cos(
                    lat0[k,i::hsplit,j::hsplit]/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
                #surface_Area = dx*dy*(np.pi/(180.0*hsplit))**2 * np.cos(lat0[i::hsplit,j::hsplit]/180.0*np.pi)
                masses[k,i::hsplit,j::hsplit] = surface_Area*dens_grid*r_term*1e9
                thick[k,i::hsplit,j::hsplit] = (bot-top)
    return lon0.flatten(),lat0.flatten(),depths.flatten(),masses.flatten(),thick.flatten()

def grid_to_pointmasses_2(lonGrid,latGrid,top_grid,bottom_grid,dens_grid,
                          hsplit=1,vsplit=1):
    """Convert a tesseroid grid to a set of point masses (ND-shaped output)
    """
    ny,nx = lonGrid.shape[0],lonGrid.shape[1]
    dx = np.abs(lonGrid[0,1]-lonGrid[0,0])
    dy = np.abs(latGrid[1,0]-latGrid[0,0])
    lon0 = np.zeros((vsplit,hsplit,hsplit,ny,nx))
    lat0 = np.zeros((vsplit,hsplit,hsplit,ny,nx))
    depths = np.zeros(lon0.shape)
    masses = np.zeros(lat0.shape)
    thick = np.zeros(lat0.shape)
    r_top = 6371-top_grid
    r_bot = 6371-bottom_grid
    r_term = (r_top**3-r_bot**3)/3.0
    dlon = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dx
    dlat = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dy
    for k in range(vsplit):
        if k==0:
            top = top_grid
        else:
            top = (bottom_grid-top_grid)/(1.0*vsplit)*k + top_grid
        if k==vsplit-1:
            bot = bottom_grid
        else:
            bot = (bottom_grid-top_grid)/(1.0*vsplit)*(k+1) + top_grid
        r_top = 6371-top
        r_bot = 6371-bot
        r_term = (r_top**3-r_bot**3)/3.0
        for i in range(hsplit):
            for j in range(hsplit):
                lon0[k,i,j,:,:] = lonGrid + dlon[j]
                lat0[k,i,j,:,:] = latGrid + dlat[i]
                depths[k,i,j,:,:] = 0.5 * (top+bot)
                surface_Area = dx*(np.pi/(180.0*hsplit)) *np.cos(
                    lat0[k,i,j,:,:]/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
                #surface_Area = dx*dy*(np.pi/(180.0*hsplit))**2 * np.cos(lat0[i::hsplit,j::hsplit]/180.0*np.pi)
                masses[k,i,j,:,:] = surface_Area*dens_grid*r_term*1e9
                thick[k,i,j,:,:] = (bot-top)
    return lon0,lat0,depths,masses,thick

def mag_dipole(dipoles,magnetizations,stations,**kwargs):
    """
    dim 1: N_stations
    dim 2: N_dipoles
    dim 3: 3
    """
    
    max_size = kwargs.get("max_size",1000000000) # in bytes
    verbose = kwargs.get("verbose",False)
    N_stations,N_dipoles = stations.shape[0],magnetizations.shape[0]
    
    partitions = max(1,np.ceil(8 * N_stations * N_dipoles / max_size).astype(int))
    ixs = np.array_split(np.arange(N_stations,dtype=int),partitions)
    field = np.zeros((N_stations,3))
    
    if verbose:
        print('Using %d partitions for mag_dipole' % partitions)
    
    for ix in ixs:
        r = stations[ix,None,:] - dipoles[None,:,:]
        r_val = np.sqrt((r**2).sum(-1))
        T1 = 3.0*(magnetizations[None,:,:]*r).sum(-1)/(r_val**5)
        T2 = 1.0/(r_val**3)
        field[ix] =  (T1[:,:,None] * r - T2[:,:,None] * magnetizations).sum(1)
    return field