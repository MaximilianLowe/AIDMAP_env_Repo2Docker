import numpy as np
import scipy.signal.windows as windows



def multitaper_2d(z,dx,dy,NW=4,Kmax=None,additional_returns=False):
    if Kmax is None:
        Kmax = 2*NW-1
    w = windows.dpss(z.shape[0],4,Kmax)[:,:,None] * windows.dpss(z.shape[1],NW,Kmax)[:,None,:]
    f = np.fft.fftn(z[None,:,:]*w,axes=(1,2))
    kx = 2.0*np.pi * np.fft.fftfreq(z.shape[1],dx)
    ky = 2.0*np.pi * np.fft.fftfreq(z.shape[0],dy)
    kxi,kyi = np.meshgrid(kx,ky)
    kri = np.sqrt(kxi**2+kyi**2)
    
    bins = np.linspace(0,min(kx.max(),ky.max()),min(z.shape[0],z.shape[1])//2)
    bin_ix = np.floor(kri/(bins[1]-bins[0])).astype(int)
    P = np.array([np.mean(np.abs(f[:,bin_ix==i])**2) for i in range(len(bins))])
    if additional_returns:
        sigma = np.array([np.std(np.abs(f[:,bin_ix==i])**2) for i in range(len(bins))])
        bincount = np.array([np.sum(bin_ix==i) for i in range(len(bins))])
        return bins,P,bin_ix,sigma,bincount
    else:
        return bins,P,bin_ix



def generate_from_power_2d(Nx,Ny,Lx,Ly,P_func,remove_inf = True,additional_returns = False):
    """Create grid according to given isotropic power spectral density function
    
    Parameters
    ----------
    Nx,Ny: int
        Number of elements in x and y direction. Must be odd
    Lx,Ly : double
        Size of domain
    P_func : callable
        Function of radial wavenumber that generates power spectral density
    remove_inf : boolean
        Whether infinite returns from P_func should be set to zero.
        This is useful, when e.g. dealing with power laws, which are
        infinite at kr==0
    additional_returns : boolean
        Whether to return the 2-D Power spectral density and 2-D covariance
        function
    """
    assert Nx % 2 == 1
    assert Ny % 2 == 1
    
    # x and y are one longer than needed, because I want to calculate dk
    x = np.linspace(0,Lx,Nx+1)
    y = np.linspace(0,Ly,Ny+1)
    xi,yi = np.meshgrid(x[:-1],y[:-1])

    kx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(Nx,x[1]-x[0]))
    ky = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(Ny,y[1]-y[0]))
    kxi,kyi = np.meshgrid(kx,ky)
    
    kri = np.sqrt(kxi**2+kyi**2)
    P_raw = P_func(kri)
    if remove_inf:
        P_raw[np.isinf(P_raw)] = 0.
    
    P = P_raw / Lx / Ly # For some reason its not 2pi/L
    phase = np.random.random((Ny,Nx)) * 2.0 * np.pi
    phase[:Ny//2+1,:Nx//2+1] = -phase[Ny//2:,Nx//2:][::-1,::-1] # Mirror the ++ to the -- quadrant
    phase[Ny//2:,:Nx//2+1] = -phase[:Ny//2+1,Nx//2:][::-1,::-1] # Mirror the +- to the -+ quadrant
    phase[Ny//2,Nx//2] = 0.
    a = np.sqrt(P) * np.exp(1j*phase)
    z = np.fft.ifft2(np.fft.ifftshift(a)) * Nx * Ny
    z = z.real
    # Calculate IFFT of Power spectrum
    C = np.fft.ifft2(np.fft.ifftshift(P)) * Nx * Ny
    C = C.real
    # There is a factor of two here, which I dont understand, but it appears to be necessary
    C = 0.5 * C
    if additional_returns:
        return z,P,C
    else:
        return z
        
        
        
def estimate_2d_power_spectrum(z,dx,dy,window_func=None):
    if window_func is None:
        f = np.fft.fft2(z)    
    else:
        w = window_func(z.shape[0])[:,None] * window_func(z.shape[1])[None,:]
        f = np.fft.fft2(z*w)
    kx = 2.0*np.pi * np.fft.fftfreq(z.shape[1],dx)
    ky = 2.0*np.pi * np.fft.fftfreq(z.shape[0],dy)
    kxi,kyi = np.meshgrid(kx,ky)
    kri = np.sqrt(kxi**2+kyi**2)
    
    bins = np.linspace(0,min(kx.max(),ky.max()),min(z.shape[0],z.shape[1])//2)
    bin_ix = np.floor(kri/(bins[1]-bins[0])).astype(int)
    P = np.array([np.mean(np.abs(f[bin_ix==i])**2) for i in range(len(bins))])
    return bins,P,bin_ix