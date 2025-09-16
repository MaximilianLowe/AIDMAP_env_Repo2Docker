"""SH_filter contains routines for dealing with spherical harmonics.
.. warning:: All spherical harmonics are Schmidt semi-normalized!

Spherical harmonic coefficients
-------------------------------
We make use of real spherical harmonics (SH) in this module.

In most cases, spherical harmonic coefficients are stored in two
one-dimensional arrays, for cosine and sine part. This format
is *redundant*, so even 'unnecessary' parts are stored. The order
of stored coefficients is p00, p10, p11, p20,p21,p22, ...

The *stacked* coefficient vector is found by concatenating
the vector of cosine coeffs with the vector of sine coeffs.

The *A-B* representation is mainly used for plotting and some
calculations. Here, the cosine coeffs are stored in a matrix A,
such that A[l,m] contains coefficient plm. And similarly for 
the sine coefficients in matrix B[l,m]
"""

import subprocess
import logging
try:
    import mpmath
    found_mpmath = True
except ImportError:
    found_mpmath = False
import numpy as np
import numpy.polynomial.legendre as legendre
try:
    from scipy.special import sph_harm
    from scipy.interpolate import interp1d
    found_scipy = True
except ImportError:
    found_scipy = False
try:
    import sympy.physics.wigner as wigner
except ImportError:
    print('Sympy not found, coef_rotate not available!')


def np_Plm_sch(lmax,theta):
    return Plm_sch_manual(lmax,theta)    

def mpmath_Plm_sch(lmax,theta):
    """Calculate associated legendre polynomials using mpmath
    Parameters
    ----------
    lmax: int
        Highest degree to use
    theta: array-like
        Values of the argument
    
    Returns
    -------
    Plm: np.ndarray
        Evaluations of the legendre polynomials in coefficient vector form
    """
    Plm = np.zeros((lmax*(lmax+1)//2 + lmax + 1))
    for l in range(lmax+1):
        for m in range(l+1):
            if m<1:
                y = float(sph_harm(m,l,0.0,theta).real)
            else:
                y = float(spherharm(l,m,theta,0.0).real)
            if np.isnan(y):
                y = float(spherharm(l,m,theta,0.0).real)
            if m>0:
                y = (np.sqrt(2) * y)
            index = l*(l+1)//2 + m
            Plm[index] = y / np.sqrt((2*l+1)/(4.0*np.pi)) * (-1)**m
    return Plm
    
def Plm_sch_manual(lmax,theta):
    """Calculate associated legendre polynomials using recursion formula
    Parameters
    ----------
    lmax: int
        Highest degree to use
    theta: array-like
        Values of the argument
    
    Returns
    -------
    Plm: np.ndarray
        Evaluations of the legendre polynomials in coefficient vector form
    """
    Plm = np.zeros((lmax*(lmax+1)//2 + lmax + 1))
    PlmMat = coeff_vec_to_AB(np.hstack((Plm,Plm)))[0]
    for l in range(lmax+1):
        for m in range(l+1):
            if l==0:
                PlmMat[l,m] = 1.0
            elif l==1 and m==1:
                PlmMat[l,m] = np.sin(theta)
            elif l==1 and m==0:
                PlmMat[l,m] = (2.0*l - 1)/np.sqrt(l**2 - m**2)*np.cos(theta)*PlmMat[l-1, m] 
            elif l==m:
                PlmMat[l,m] = np.sqrt(1 - 1.0/(2.0*l))*np.sin(theta)*PlmMat[l-1, m-1]
            else:
                PlmMat[l,m] = (2.0*l - 1)/np.sqrt(l**2 - m**2)*np.cos(theta)*PlmMat[l-1, m] 
                PlmMat[l,m] = PlmMat[l,m] - np.sqrt(((l-1.0)**2 - m**2) / (1.0*l**2 - m**2)) * PlmMat[l-2, m]
    Plm = AB_to_coeff_vec(PlmMat,PlmMat)
    return Plm[0:len(Plm)//2]

def np_dPlm_sch(Plm,theta):
    """Calculate derivative of associated legendre polynomials using recursion
    Parameters
    ----------
    lmax: int
        Highest degree to use
    theta: array-like
        Values of the argument
    
    Returns
    -------
    Plm: np.ndarray
        Derivative of the legendre polynomials in coefficient vector form
    """
    PlmMat = coeff_vec_to_AB(np.hstack((Plm,Plm)))[0]
    lmax = PlmMat.shape[0]-1
    dPlmMat = np.zeros((lmax+1,lmax+1))
    for l in range(lmax+1):
        for m in range(l+1):
            if l==0:
                dPlmMat[l,m] = 0
            elif l==1 and m==1:
                dPlmMat[l,m] = np.cos(theta)
            elif l==1 and m==0:
                dPlmMat[l, m] = (2.0*l - 1)/np.sqrt(l*l - m*m)*(np.cos(theta)*dPlmMat[l-1, m] -
                    np.sin(theta)*PlmMat[l-1, m])
            elif l==m:
                dPlmMat[l,m] = np.sqrt(1 - 1.0/(2.0*l))*(np.sin(theta)*dPlmMat[l-1, l-1] 
                                + np.cos(theta)*PlmMat[l-1, l-1])
            else:
                dPlmMat[l, m] = (2.0*l - 1)/np.sqrt(l*l - m*m)*(np.cos(theta)*dPlmMat[l-1, m] -
                    np.sin(theta)*PlmMat[l-1, m]) - np.sqrt(((l-1)**2 - m*m)/(1.0*l*l - m*m))*dPlmMat[l-2, m]
    dPlm = AB_to_coeff_vec(dPlmMat,dPlmMat)
    return dPlm[0:len(dPlm)//2]
    
def get_SHS_design_matrix_grid(lon_lat,lmax,shape=None):
    """Calculate the Spherical Harmonic Synthesis matrix for a equi-angular grid
    Parameters
    ----------
    lon_lat: tuple of np.ndarray
        Contains the lon and lat values of the grid 
    lmax: int
        Highest degree to use
    shape: tuple, optional
        If shape is given, lon_lat is not used. Instead a global grid is constructed
        that goes from -180 to 180 in longitude and -90 to 90 in latitude
    Returns
    -------
    A_design: np.ndarray
        Contains the evaluated cosine spherical harmonics at the flattened grid locations. 
        By calculating `A_design.dot(coeff_vec)`, spherical harmonic synthesis is carried out.
    B_design: np.ndarray
        Contains the evaluated sine spherical harmonics at the flattened grid locations.
    """
    lon,lat = lon_lat
    if shape is None:
        nx = len(lon)
        ny = len(lat)
    else:
        ny = shape[0]
        nx = shape[1]
        dx = 360.0/nx
        dy = 180.0/ny
        lon = np.linspace(-180,180,nx)
        lat = np.linspace(-90,90,ny)
    theta = (90-lat) * np.pi / 180.0
    phi = lon * np.pi / 180.0
    ncoeff = lmax*(lmax+1)//2 + lmax + 1
    big_Plm = np.zeros((ncoeff,ny))
    for i in range(ny):
        big_Plm[:,i] = np_Plm_sch(lmax,theta[i])
    
    cos_vec = np.zeros((ncoeff,nx))
    sin_vec = np.zeros((ncoeff,nx))
    for m in range(lmax+1):
        k = np.arange(m,lmax+1)
        indices = k*(k+1)//2 + m
        cos_vec[indices,:] = np.cos(m*phi)
        sin_vec[indices,:] = np.sin(m*phi)
    
    A_design = (big_Plm[:,:,None] * cos_vec[:,None,:]).reshape((ncoeff,nx*ny))
    B_design = (big_Plm[:,:,None] * sin_vec[:,None,:]).reshape((ncoeff,nx*ny))
    return A_design.T,B_design.T
    
def get_SHA_design_matrix_grid(lon_lat,lmax,shape=None):
    """Get SHA design matrix from trapez rule integration
    """
    lon,lat = lon_lat
    for_design = get_SHS_design_matrix_grid((lon,lat),lmax,shape=shape)
    return integrate_design_matrix((lon,lat),np.hstack((for_design[0],for_design[1])),shape=shape)
    
def integrate_design_matrix(lon_lat,input_design,shape=None):
    lon,lat = lon_lat
    
    if shape is None:
        nx = len(lon)
        ny = len(lat)
    else:
        ny = shape[0]
        nx = shape[1]
        dx = 360.0/nx
        dy = 180.0/ny
        lon = np.linspace(-180,180,nx)
        lat = np.linspace(-90,90,ny)
    lmax = int((np.sqrt(input_design.shape[1]//2*8+1)-3)//2)
    print(lmax)
    theta = (90-lat) * np.pi / 180.0
    phi = lon * np.pi / 180.0
    ncoeff = lmax*(lmax+1)/2 + lmax + 1
    big_Plm = np.zeros((ncoeff,ny))
    dtheta = np.gradient(theta)[None,:,None]
    dphi = np.gradient(phi)[None,None,:]
    
    l_vec = np.zeros((ncoeff))
    index = 0
    for l in range(lmax+1):
        l_vec[index:index+l+1] = l
        index = index+l+1
    
    weights_theta = np.ones((ny))
    weights_theta[0] = 0.5
    weights_theta[-1] = 0.5
    weights_phi = np.ones((nx))
    weights_phi[0] = 1.0
    weights_phi[-1] = 1.0
    
    l_vec = l_vec[:,None,None]
    sin_theta = np.sin(theta)[None,:,None]
    weights_theta = weights_theta[None,:,None]
    weights_phi = weights_phi[None,None,:]
    temp1 = input_design[:,:ncoeff].T.reshape((ncoeff,ny,nx)) * weights_theta * weights_phi * sin_theta * (2.0*l_vec + 1.0) / (4.0*np.pi) * dtheta *dphi
    temp2 = input_design[:,ncoeff:].T.reshape((ncoeff,ny,nx)) * weights_theta * weights_phi * sin_theta * (2.0*l_vec + 1.0) / (4.0*np.pi) * dtheta *dphi
    return np.vstack((temp1.reshape((ncoeff,ny*nx)),temp2.reshape((ncoeff,ny*nx))))

def get_field_design_matrix_grid(lon_lat,lmax,a,r,shape=None):
    """
    Construct the design matrix that relates the Gauss coefficients (g,h) to the
    3-component vector magnetic field at a specific height.
    The gauss coefficients are assumed to be in a vector ordered like this
    g00,g10,g11,g20,g21,g22...,gnn,h00,h10,h11,...,hnn
    Note that I'm leaving room for a 00 component, even though that doesn't
    exist in magnetics. This is to allow the same code to be used for gravity, 
    where such coefficients do exist and are relevant in some cases.
    
    The field is given in NED system for output. The order of the components
    is likewise NED.
    """
    lon,lat = lon_lat
    if shape is None:
        nx = len(lon)
        ny = len(lat)
    else:
        ny = shape[0]
        nx = shape[1]
        dx = 360.0/nx
        dy = 180.0/ny
        lon = np.linspace(-180,180,nx)
        lat = np.linspace(-90,90,ny)
    theta = (90-lat) * np.pi / 180.0
    phi = lon * np.pi / 180.0
    ncoeff = lmax*(lmax+1)//2 + lmax + 1
    q = a/r
    if not isinstance(r,np.ndarray):
        q = np.reshape(q,(1,1,1))
    else:
        q = q[None,:,:]
    big_Plm = np.zeros((ncoeff,ny))
    big_dPlm = np.zeros((ncoeff,ny))
    
    for i in range(ny):
        big_Plm[:,i] = np_Plm_sch(lmax,theta[i])
        big_dPlm[:,i] = np_dPlm_sch(big_Plm[:,i],theta[i])
        
    cos_vec = np.zeros((ncoeff,nx))
    sin_vec = np.zeros((ncoeff,nx))
    m_vec = np.zeros((ncoeff))
    for m in range(lmax+1):
        k = np.arange(m,lmax+1)
        indices = k*(k+1)//2 + m
        cos_vec[indices,:] = np.cos(m*phi)
        sin_vec[indices,:] = np.sin(m*phi)
        m_vec[indices] = m

    l_vec = np.zeros((ncoeff))
    index = 0
    for l in range(lmax+1):
        l_vec[index:index+l+1] = l
        index = index+l+1
    
    l_vec = l_vec[:,None,None]
    m_vec = m_vec[:,None,None]
    
    big_design_matrix = np.zeros((3,nx*ny,2*ncoeff))
    logging.debug('%s %s %s',cos_vec[:,None,:].shape,big_Plm[:,:,None].shape,l_vec.shape)
    # North design matrix
    A_design = (q**(l_vec+2.0) * big_dPlm[:,:,None] * cos_vec[:,None,:])
    A_design = A_design.reshape((ncoeff,nx*ny))
    B_design = (q**(l_vec+2.0) * big_dPlm[:,:,None] * sin_vec[:,None,:]).reshape((ncoeff,nx*ny))
    big_design_matrix[0,:,0:ncoeff] = A_design.T
    big_design_matrix[0,:,ncoeff:] = B_design.T
    # East design matrix
    # TODO: Deal with theta==0, there is a special formula for this case
    A_design = ( (m_vec * q**(l_vec+2.0)) * big_Plm[:,:,None] * sin_vec[:,None,:]) / (np.sin(theta)[None,:,None])
    A_design = A_design.reshape((ncoeff,nx*ny)) 
    B_design = -( (m_vec * q**(l_vec+2.0)) * big_Plm[:,:,None] * cos_vec[:,None,:]) / (np.sin(theta)[None,:,None])
    B_design = B_design.reshape((ncoeff,nx*ny))
    big_design_matrix[1,:,0:ncoeff] = A_design.T
    big_design_matrix[1,:,ncoeff:] = B_design.T
    # Down design matrix
    A_design = -( (q**(l_vec+2.0) * (l_vec+1.0) ) * big_Plm[:,:,None] * cos_vec[:,None,:])
    A_design = A_design.reshape((ncoeff,nx*ny))
    B_design = -( (q**(l_vec+2.0) * (l_vec+1.0) ) * big_Plm[:,:,None] * sin_vec[:,None,:]).reshape((ncoeff,nx*ny))
    big_design_matrix[2,:,0:ncoeff] = A_design.T
    big_design_matrix[2,:,ncoeff:] = B_design.T
    return big_design_matrix
    
def get_field_design_matrix_points(lons,lats,heights,lmax,a):
    """
    Get the field design matrix at discrete points (not on a grid)
    The field is given in NED system for output. The order of the components
    is likewise NED.
    """
    theta = (90-lats) * np.pi / 180.0
    phi = lons * np.pi / 180.0
    ncoeff = lmax*(lmax+1)//2 + lmax + 1
    r = a + heights
    q = a/r
    npoints = len(lons)
    big_Plm = np.zeros((ncoeff,npoints))
    big_dPlm = np.zeros((ncoeff,npoints))
    
    for i in range(npoints):
        big_Plm[:,i] = np_Plm_sch(lmax,theta[i])
        big_dPlm[:,i] = np_dPlm_sch(big_Plm[:,i],theta[i])
        
    cos_vec = np.zeros((ncoeff,npoints))
    sin_vec = np.zeros((ncoeff,npoints))
    m_vec = np.zeros((ncoeff))
    for m in range(lmax+1):
        k = np.arange(m,lmax+1)
        indices = k*(k+1)/2 + m
        cos_vec[indices,:] = np.cos(m*phi)
        sin_vec[indices,:] = np.sin(m*phi)
        m_vec[indices] = m

    l_vec = np.zeros((ncoeff))
    index = 0
    for l in range(lmax+1):
        l_vec[index:index+l+1] = l
        index = index+l+1
    
    big_design_matrix = np.zeros((3,npoints,2*ncoeff))
    print(cos_vec[:,None,:].shape,big_Plm[:,:,None].shape,l_vec[:,None,None].shape)
    # North design matrix
    A_design = (q[None,:]**(l_vec[:,None]+2) * big_dPlm * cos_vec)
    A_design = A_design.reshape((ncoeff,npoints))
    B_design = (q[None,:]**(l_vec[:,None]+2) * big_dPlm * sin_vec).reshape((ncoeff,npoints))
    big_design_matrix[0,:,0:ncoeff] = A_design.T
    big_design_matrix[0,:,ncoeff:] = B_design.T
    # East design matrix
    # TODO: Deal with theta==0, there is a special formula for this case
    A_design = ( (m_vec[:,None] * q[None,:]**(l_vec[:,None]+2)) * big_Plm * sin_vec) / (np.sin(theta)[None,:])
    A_design = A_design.reshape((ncoeff,npoints)) 
    B_design = -( (m_vec[:,None] * q[None,:]**(l_vec[:,None]+2)) * big_Plm * cos_vec) / (np.sin(theta)[None,:])
    B_design = B_design.reshape((ncoeff,npoints))
    big_design_matrix[1,:,0:ncoeff] = A_design.T
    big_design_matrix[1,:,ncoeff:] = B_design.T
    # Down design matrix
    A_design = -( (q[None,:]**(l_vec[:,None]+2) * (l_vec[:,None]+1.0) ) * big_Plm * cos_vec)
    A_design = A_design.reshape((ncoeff,npoints))
    B_design = -( (q[None,:]**(l_vec[:,None]+2) * (l_vec[:,None]+1.0) ) * big_Plm * sin_vec).reshape((ncoeff,npoints))
    big_design_matrix[2,:,0:ncoeff] = A_design.T
    big_design_matrix[2,:,ncoeff:] = B_design.T
    return big_design_matrix



def get_total_field_anomaly(Bcrust,Bcore):
    dB = Bcrust+Bcore
    return np.sqrt(dB[:,0]**2 + dB[:,1]**2 + dB[:,2]**2) - np.sqrt(
            Bcore[:,0]**2 + Bcore[:,1]**2 + Bcore[:,2]**2)
    
def AB_to_gh(A,B,a,r):
    lmax = A.shape[0]-1
    g = A*0
    h = B*0
    q = (a/r)
    for l in range(lmax+1):
        g[l,...] = A[l,...] / ((l+1) * q**(l+2))
        h[l,...] = B[l,...] / ((l+1) * q**(l+2))
    return g,h

def gh_to_Bz(g,h,design_matrix,a,r):
    lmax = g.shape[0]
    q = (a/r)
    A = g*0
    B = h*0
    for l in range(lmax+1):
        A[l,:] = g[l,:] * (l+1) * q**(l+2)
        B[l,:] = h[l,:] * (l+1) * q**(l+2)
    field = design_matrix[0].dot(A)
    
def sh_power_lowes(g,h,r0=6371.2,r=6371.2):
    lmax = g.shape[0] - 1
    lowesPower = np.zeros((lmax+1,)+g.shape[2:])
    for l in range(lmax):
        lowesPower[l,...] = (l+1.0) * ((r0/r)**(2*l+4)) * np.sum(g[l,...]**2+h[l,...]**2,0)
    return lowesPower

def sh_power_maus(g,h,r0=6371.2,r=6371.2):
    lmax = g.shape[0] -1
    mausPower = np.zeros((lmax+1,)+g.shape[2:])
    for l in range(lmax+1):
        mausPower[l] = (l+1)/(2.0*l+1) * (r0/r)**(2*l+4) * np.sum(g[l,...]**2+h[l,...]**2,0)
    return mausPower

def sh_degree_variance(g,h):
    """
    This is defined in such a way that a white noise field has a flat spectrum
    """
    lmax = g.shape[0] - 1
    power = np.zeros((lmax+1,)+g.shape[2:])
    for l in range(lmax):
        power[l,...] = np.sum(g[l,...]**2+h[l,...]**2,0)/(2.0*l+1)**2
    return power
    
    
def AB_to_power(A,B,h=400.0):
    gh = AB_to_gh(A,B,6371.0,6371.0+h)
    power = sh_power_lowes(*gh,r0=6371.0,r=6371.0)
    return power

    
def coeff_vec_to_AB(coeff):
    lmax = int((np.sqrt(len(coeff)//2*8+1)-3)//2)
    if coeff.ndim>1:
        A = np.zeros((lmax+1,lmax+1,)+coeff.shape[1:])
    else:
        A = np.zeros((lmax+1,lmax+1,1))
    B = np.zeros(A.shape)
    for l in range(lmax+1):
        for m in range(l+1):
            index = l*(l+1)//2 + m
            A[l,m,...] = coeff[index,...]
            B[l,m,...] = coeff[index+len(coeff)//2,...]
    return np.squeeze(A),np.squeeze(B)

def AB_to_coeff_vec(A,B):
    lmax = A.shape[0]-1
    if A.ndim==2:
        A_coeff = np.zeros((lmax*(lmax+1)//2+lmax+1,1))
        B_coeff = np.zeros((lmax*(lmax+1)//2+lmax+1,1))
    else:
        A_coeff = np.zeros((lmax*(lmax+1)//2+lmax+1,)+A.shape[2:])
        B_coeff = np.zeros((lmax*(lmax+1)//2+lmax+1,)+A.shape[2:])
    
    for l in range(lmax+1):
        for m in range(l+1):
            index = l*(l+1)//2 + m
            A_coeff[index,...] = A[l,m,...]
            B_coeff[index,...] = B[l,m,...]
    A_coeff = np.squeeze(A_coeff)
    B_coeff = np.squeeze(B_coeff)
    return np.concatenate((A_coeff,B_coeff),axis=0)
        

def load_igrf_coeffs(fname):
    with open(fname) as f:
        lines = f.readlines()
    tokens = lines[3].split()
    years = np.array(list(map(float,tokens[3:-1])))
    glist = []
    hlist = []
    for line in lines[4:]:
        tokens = line.split()
        if tokens[0]=='g':
            glist.append(list(map(float,tokens[1:-1])))
        else:
            hlist.append(list(map(float,tokens[1:-1])))
    igrfDict = dict()
    igrfDict["years"] = years
    glist = np.array(glist)
    hlist = np.array(hlist)
    lmax = int(max(glist[:,0].max(),hlist[:,0].max()))
    g = np.zeros((lmax+1,lmax+1,len(years)))
    h = np.zeros((lmax+1,lmax+1,len(years)))
    for i in range(len(years)):
        g[glist[:,0].astype(int),glist[:,1].astype(int),i] = glist[:,i+2]
        h[hlist[:,0].astype(int),hlist[:,1].astype(int),i] = hlist[:,i+2]
    igrfDict["g"] = g
    igrfDict["h"] = h
    return igrfDict

def load_cof(fname,skiprows=13):
    raw_data = np.loadtxt(fname,skiprows=skiprows,dtype={'names':('l','m','gnm','hnm'),
                                                        'formats':('i','i','f','f')})
    lmax = raw_data['l'].max()
    g = np.zeros((lmax+1,lmax+1))
    h = np.zeros((lmax+1,lmax+1))
    g[raw_data['l'],raw_data['m']] = raw_data['gnm']
    h[raw_data['l'],raw_data['m']] = raw_data['hnm']
    
    return g,h

def load_shc(fname,skiprows=5):
    raw_data = np.loadtxt(fname,skiprows=skiprows,dtype={'names':('l','m','gnm'),
                                                        'formats':('i','i','f')})
    lmax = raw_data['l'].max()
    g = np.zeros((lmax+1,lmax+1))
    h = np.zeros((lmax+1,lmax+1))
    pos_ix = raw_data['m']>=0
    neg_ix = raw_data['m']<=0
    g[raw_data['l'][pos_ix],raw_data['m'][pos_ix]] = raw_data['gnm'][pos_ix]
    h[raw_data['l'][neg_ix],-raw_data['m'][neg_ix]] = raw_data['gnm'][neg_ix]
    return g,h

def interp_igrf(igrfDict,decimalYear):
    interpolator1 = interp1d(igrfDict["years"],igrfDict["g"])
    interpolator2 = interp1d(igrfDict["years"],igrfDict["h"])
    gInterp  = interpolator1(decimalYear)
    hInterp  = interpolator2(decimalYear)
    return gInterp,hInterp

class Coef_rotator:
    """Rotate the spherical harmonic expansion along latitude direction
    Calculate how the SH expansion would change under a rotation of the coordinate
    system along the y-axis (-> change in latitude). If you rotate by 90 deg.
    you can put the poles at the equator.
    """
    def __init__(self,lmax,gamma):
        wiggi = []
        for i in range(lmax+1):
            wiggi.append(wigner.wigner_d_small(i,gamma).evalf())
        self.wiggi = wiggi
        self.lmax = lmax
    def rotate(self,g,h):
        assert g.shape[0] == self.lmax+1
        g_rot = np.zeros(g.shape)
        h_rot = np.zeros(g.shape)
        for l in range(1,self.lmax):
            A = np.zeros(2*l+1)
            A[:l] = h[l,1:l+1][::-1]
            A[l:] = g[l,:l+1]
            A_rot= np.array(self.wiggi[l]).dot(A)
            g_rot[l,:l+1] = A_rot[l:]
            h_rot[l,1:l+1] = A_rot[:l][::-1]
        return g_rot,h_rot
    
def generate_from_power_sphere(lon,lat,coef):
    """Create grid according to given isotropic power spectral density function
    
    Use function `lstsq_legendre_series_tol` to obtain the coefficients if you use a covariance function
    
    Parameters
    ----------
    lon,lat: np.array
        Will be converted into meshgrid and used to calculate the random function at those points
    coef : np.array
        Coefficients of the legendre series of the covariance function
    """
    lmax = coef.size - 1
    A_design,B_design = get_SHS_design_matrix_grid((lon,lat),lmax)

    l_mat = np.zeros(A_design.shape[1],dtype=int)
    counter = 0
    for l in range(lmax+1):
        l_mat[counter:counter+l+1] = l
        counter = counter + l + 1

    A_vec = np.random.randn(A_design.shape[1]) * np.sqrt(np.abs(coef[l_mat]))
    B_vec = np.random.randn(A_design.shape[1]) * np.sqrt(np.abs(coef[l_mat]))
    
    rando_field = (A_design.dot(A_vec) + B_design.dot(B_vec)).reshape((lat.size,lon.size))
    return rando_field

def lstsq_legendre_series(gam,vals,lmax):
    """Fit a least-squares legendre series
    Fit the values of a function given at locations gam with a Legendre series of order lmax
    Parameters
    ----------
    gam: np.array
        Points at which the function is given
    vals: np.array
        Function values
    lmax: int
        Degree of Legendre Series to use
    
    Returns
    -------
    coef : np.array
        Legendre coefficients (low to high)
    residual : float
        Achieved L2 misfit. If this is too high, consider running with a higher lmax.
    max_err : float
        Maximum deviation (infinity norm)
    """
    
    cos_ga = np.cos(gam)
    coef,(residuals,rank,_,_) = legendre.legfit(cos_ga,vals,lmax,full=True)
    max_err = np.max(np.abs(legendre.legval(cos_ga,coef) - vals))
    return coef,residuals,max_err
    
def lstsq_legendre_series_tol(gam,vals,abs_tol,l_start=10,l_inc=10,l_limit=400,epsilon = 0.8,verbose=False):
    """Fit a least-squares legendre series with specified absolute error tolerance
    
    The degree l is increased until the error tolerance is reached. 
    
    Parameters
    ----------
    gam: np.array
        Points at which the function is given
    vals: np.array
        Function values
    abs_tol: float
        Maximum allowed absolute deviation allowed from vals
    l_start : int,optional
        First l_value to try
    l_inc : int,optional
        First increment of l
    l_limit : int,optional
        Upper limit for l. If l==l_limit, it means that the error tolerance is probably not reached
    epsilon : real,optional
        Should be <1. When calculating the next value of l to use, abs_tol * epsilon is used,
        since the formula for the degree l to use has a tendency to underestimate.
    """
    coef,residuals,old_err = lstsq_legendre_series(gam,vals,l_start)
    old_l = l_start
    l = l_start + l_inc
    while old_err>abs_tol and l<=l_limit:
        coef,residuals,new_err = lstsq_legendre_series(gam,vals,l)
        # Update l under the assumption that max_err decays exponentially with l
        next_l = old_l + (np.log(epsilon*abs_tol) - np.log(old_err))/(np.log(new_err)-np.log(old_err)) * (l-old_l)
        next_l = min(l_limit,max(int(np.ceil(next_l)),1))
        if verbose:
            print('l=%d -> Error %.2e l=%d -> New Error %.2e, next l=%d'%(old_l,old_err,l,new_err,next_l))
        old_l = l
        l = next_l
        old_err = new_err
    
    return coef,residuals,old_err

def generation_helper(lon,lat,design_matrix,covariance_function,n_gam=400):
    """Generate grid according to covariance function

    Parameters
    ----------
    lon : np.array
    lat : np.array
    design_matrix : tuple of np.array
        Two design matrices for the cosine and sine part of real spherical
        harmonics. The rows correspond to the meshgrid of lon and lat.
        The cols define the maxmium degree `lmax`
    covariance_function : func
        Takes distances in km and returns the covariance between two points
    n_gam : int, optional
        How many evaluations of the covariance_function should be used to
        estimate its Legendre Series.
    """
    A_design,B_design  = design_matrix
    lmax = int((np.sqrt(2*A_design.shape[1]/2*8+1)-3)/2)    
    gam = np.linspace(0,180,n_gam)
    Cs = covariance_function(gam*110.0)
    
    coef,residuals,max_err = lstsq_legendre_series(gam/180.0*np.pi,Cs,lmax)
    l_mat = np.zeros(A_design.shape[1],dtype=int)
    counter = 0
    for l in range(lmax+1):
        l_mat[counter:counter+l+1] = l
        counter = counter + l + 1

    A_vec = np.random.randn(A_design.shape[1]) * np.sqrt(np.abs(coef[l_mat]))
    B_vec = np.random.randn(A_design.shape[1]) * np.sqrt(np.abs(coef[l_mat]))
    
    return (A_design.dot(A_vec) + B_design.dot(B_vec)).reshape((lat.size,lon.size)),coef,max_err