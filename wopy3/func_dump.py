import numpy as np
from scipy.special import kv,gamma

def get_pairwise_geo_distance(lon,lat):
    # Calculate spherical distance
    N = len(lon)
    pd = np.zeros((N,N))
    coslat = np.cos(lat/180.0*np.pi)
    sinlat = np.sin(lat/180.0*np.pi)
    
    for i in range(N):
        dx = lon - lon[i]
        cosdx = np.cos(dx/180.0*np.pi)
        pd[i,:] = coslat[i]*coslat*cosdx + sinlat[i]*sinlat
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    np.fill_diagonal(pd,0.0)
    return pd

def get_pairwise_cross_distance(lon1,lat1,lon2,lat2):
    N1 = len(lon1)
    N2 = len(lon2)
    
    coslat1 = np.cos(lat1/180.0*np.pi)
    sinlat1 = np.sin(lat1/180.0*np.pi)

    coslat2 = np.cos(lat2/180.0*np.pi)
    sinlat2 = np.sin(lat2/180.0*np.pi)
    
    pd = np.zeros((N1,N2))
    for i in range(N1):
        dx = lon2 - lon1[i]
        cosdx = np.cos(dx/180.0*np.pi)
        pd[i,:] = coslat1[i] * coslat2 * cosdx + sinlat1[i] * sinlat2
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    return pd
    
def get_all_geo_distance(lon,lat,lon0,lat0):
    N = len(lon)
    coslat = np.cos(lat/180.0*np.pi)
    sinlat = np.sin(lat/180.0*np.pi)
    
    coslat0 = np.cos(lat0/180.0*np.pi)
    sinlat0 = np.sin(lat0/180.0*np.pi)
        

    dx = lon - lon0
    cosdx = np.cos(dx/180.0*np.pi)
    pd = coslat0*coslat*cosdx + sinlat0*sinlat
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    return pd

def get_pairwise_geo_azim(lon,lat):
    N = len(lon)
    paz = np.zeros((N,N))
    coslat = np.cos(lat/180.0*np.pi)
    sinlat = np.sin(lat/180.0*np.pi)
    tanlat = np.tan(lat/180.0*np.pi)
    
    for i in range(N):
        dx = lon - lon[i]
        cosdx = np.cos(dx/180.0*np.pi)
        sindx = np.sin(dx/180.0*np.pi)
        paz[i,:] = 180.0/np.pi*np.arctan2(sindx,(coslat[i]*tanlat - sinlat[i]*cosdx))
    return paz

def minor_arc_great_circle(lon1,lat1,lon2,lat2,fineness=100,return_cartesian=False,use_spacing=False,return_alpha=False):
    """Connect two points along the minor arc of great circle
    """
    sind = lambda x:np.sin(x/180.0*np.pi)
    cosd = lambda x:np.cos(x/180.0*np.pi)
    
    dist = np.arccos(cosd(lat1)*cosd(lat2)*cosd(lon1-lon2) + sind(lat1)*sind(lat2))/np.pi*180.0
    
    # Convert to cartesian coordinates using normal and tangential vector
    
    r1 = np.stack((cosd(lat1)*cosd(lon1),cosd(lat1)*sind(lon1),sind(lat1)))
    r2 = np.stack((cosd(lat2)*cosd(lon2),cosd(lat2)*sind(lon2),sind(lat2)))
    N = np.cross(r1,r2)
    N = N / np.linalg.norm(N)
    T = np.cross(N,r1)
    
    if use_spacing:
        alpha = np.arange(0,dist+fineness,fineness)[:,None]    
    else:
        alpha = np.linspace(0,dist,fineness)[:,None]
    
    p = cosd(alpha) * r1 + sind(alpha) * T
    
    latp = np.arcsin(p[:,2])*180.0/np.pi
    lonp = np.arctan2(p[:,1],p[:,0])*180.0/np.pi
    if return_cartesian:
        return lonp,latp,p
    elif return_alpha:
        return lonp,latp,alpha
    else:
        return lonp,latp

def get_semivariogram(pd,val,h,binWidth):
    N = pd.shape[0]
    valid = np.where((pd >= h-0.5*binWidth) & (pd <= h+0.5*binWidth))
    dz = (val[valid[0]] - val[valid[1]])**2
    return np.sum(dz) / (2.0 * len(dz)),len(dz)/2

def get_semivariogram_lags(pd,val,hmax,binWidth):
    nbins = int(hmax/binWidth) + 1
    hs = np.arange(0,nbins,1) * binWidth + 0.5 * binWidth
    gamma = np.zeros((nbins))
    bincount = np.zeros((nbins))
    for i in range(nbins):
        gamma[i],bincount[i] = get_semivariogram(pd,val,hs[i],binWidth)
    return hs,gamma,bincount

def get_aniso_semivariogram(pd,paz,val,h,az,binWidths):
    dh = binWidths[0]
    daz = binWidths[1]
    
    valid = np.where((pd >= h-0.5*dh) & (pd <= h+0.5*dh) & 
                    (paz >= az-0.5*daz) & (paz <= az+0.5*daz))
    dz = (val[valid[0]]- val[valid[1]])**2
    return np.sum(dz) / (2.0 * len(dz)),len(dz)/2

def get_aniso_semivariogram_lags(pd,paz,val,hmax,dh,naz):
    nbins = int(hmax/dh) + 1
    hs = np.arange(0,nbins,1) * dh + 0.5 * dh
    
    gamma = np.zeros((nbins,naz))
    bincount = np.zeros((nbins,naz))
    
    azs = np.arange(0,naz,1) * 360.0 / naz + 180.0 / naz -180.0
    
    for i in range(nbins):
        for j in range(naz):
            gamma[i,j],bincount[i,j] = get_aniso_semivariogram(
                    pd,paz,val,hs[i],azs[j],(dh,360.0/naz))
    return hs,azs,gamma,bincount

def get_covariogram(pd,vals,h,binWidth):
    N1 = pd.shape[0]
    N2 = pd.shape[1]
    Z1 = vals[0]
    Z2 = vals[1]
    valid = np.where((pd >= h-0.5*binWidth) & (pd <= h+0.5*binWidth))
    dz = -(Z1[valid[0]] - Z1[valid[1]]) * (Z2[valid[1]] - Z2[valid[0]])
    return np.sum(dz) / (2.0 * len(dz)),len(dz)/2

def get_covariogram_lags(pd,vals,hmax,binWidth):
    nbins = int(hmax/binWidth) + 1
    hs = np.arange(0,nbins,1) * binWidth + 0.5 * binWidth
    gamma = np.zeros((nbins))
    bincount = np.zeros((nbins))
    for i in range(nbins):
        gamma[i],bincount[i] = get_covariogram(pd,vals,hs[i],binWidth)
    return hs,gamma,bincount

def get_semivariogram_sphere(data,binWidth=100,lon0=0.0,lat0=0.0,radius=np.inf,size=None,max_dist = None):
    """Calculate semivariogram in spherical geometry
    
    Only uses points which are in a radius around the point lon0,lat0. If you want to use all the points,
    set `radius = np.inf`. 
    
    Parameters
    ----------
    data : np.array
        The columns should contain lon,lat,value
    binWidth : real,optional
        How big the bins for the distance should be, when calculating the semivariogram.
    lon,lat0 : float,optional
        Location of the point around which the semivariogram should be calculated.
    radius : float, optional
        How far points from `(lon0,lat0)` should be used for calculating the semivariogram.
    size : int, optional
        Take a random subset of the data points only
    max_dist : float, optional
        Maximum distance for which to calculate semivariogram
    
    Returns
    -------
    bins : np.array of float
        Distance bins (in km)
    gamma : np.array of float
        Semivariogram values
    bincount : np.array of int
        Number of pairs in each distance bin.
    """
    
    if size is None:
        size = data.shape[0]
    
    d0 = get_all_geo_distance(data[:,0],data[:,1],lon0,lat0)
    useable = np.where((110.0*d0<=radius))[0]
    
    if useable.size==0:
        return np.nan,np.nan,np.nan
    if(len(useable)<size):
        print('Warning less than %d useable points'%size)
        size = len(useable) - 1
       
    
    sel = np.random.choice(useable,size,False)
    pd = get_pairwise_geo_distance(data[sel,0],data[sel,1])
    val = data[sel,2]
    
    if max_dist is None:
        max_dist = min(radius,pd.max()*110.0)
    
    bins,gamma,bincount  = get_semivariogram_lags(pd,val,max_dist/110.0,binWidth/110.0)
    
    return bins*110.0,gamma,bincount    
    
def simple_kriging(lon,lat,d,vals,x0,covarFunction,covarParams):
    N = d.shape[0]
    sigma = covarFunction(d,covarParams)
    
    d0 = get_all_geo_distance(lon,lat,x0[0],x0[1])
    sigma0 = covarFunction(d0,covarParams)
    weights = np.linalg.solve(sigma,sigma0)
    return np.dot(weights,vals)
    
def simple_kriging_grid(lon,lat,d,vals,xlims,nums,covarFunction,covarParams,nugget=10):
    N = d.shape[0]
    sigma = covarFunction(d,covarParams) + nugget*np.diag(np.ones((N)))
    
    x = np.linspace(xlims[0][0],xlims[0][1],nums[0])
    y = np.linspace(xlims[1][0],xlims[1][1],nums[1])
    gridded = np.zeros((nums[0],nums[1]))
    
    for i in range(nums[0]):
        for j in range(nums[1]):
            d0 = get_all_geo_distance(lon,lat,x[i],y[j])
            sigma0 = covarFunction(d0,covarParams)
            weights = np.linalg.solve(sigma,sigma0)
            gridded[i,j] = np.sum(weights * vals)
    return x,y,gridded

def C_gaussian(d,covarParams):
    return covarParams[0] * np.exp(-3*(d/covarParams[1])**2)
    
def reg_G(nu,x):
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x>0] = (kv(nu,x[x>0]) * x[x>0]**nu) / (gamma(nu) * 2**(nu-1))
    y[x==0] = 1.0
    return y
    
def C_matern(d,pars):
     sigma_0 = pars[0]
     nu = pars[1]
     rho = pars[2]
     return sigma_0*reg_G(nu,d/rho)