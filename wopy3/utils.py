import re
import numpy as np
import string
import linecache
import time
import random
import scipy.spatial as spatial
import scipy.spatial as spatial

def read_cpt(filename):
    with open(filename) as file:
        lines = file.readlines()
    
    x = []
    r = []
    g = []
    b = []
    for line in lines:
        if line.startswith("#"):
            continue
        tokens = re.split('/|\s*',line)
        if tokens[0] == 'B' or tokens[0] == 'F' or tokens[0] =='N':
            continue
        
        x.append(float(tokens[0]))
        r.append(float(tokens[1]))
        g.append(float(tokens[2]))
        b.append(float(tokens[3]))
        xtemp = float(tokens[4])
        rtemp = float(tokens[5])
        gtemp = float(tokens[6])
        btemp = float(tokens[7])
    
    x.append(xtemp)
    r.append(rtemp)
    g.append(gtemp)
    b.append(btemp)
    
    x = np.array(x)
    r = np.array(r) /255.
    g = np.array(g) /255.
    b = np.array(b) /255.
    
    xnorm = (x-x.min())/(x.max()-x.min())
    
    red = []
    blue = []
    green = []
    
    for i in range(len(x)):
        red.append([xnorm[i],r[i],r[i]])
        green.append([xnorm[i],g[i],g[i]])
        blue.append([xnorm[i],b[i],b[i]])
    colordict = {"red":red,"green":green,"blue":blue}
    return colordict

def add_letters(axs,x=0.90,y=0.90,bbox_props=None):
    for i,ax in enumerate(axs.flatten()):
        ax.text(x,y,string.ascii_lowercase[i],size=20,transform=ax.transAxes,bbox=bbox_props)

def add_letters_with_brackets(axs,x=0.90,y=0.90,size=20,bbox_props=None):
    for i,ax in enumerate(axs.flatten()):
        ax.text(x,y,'('+string.ascii_lowercase[i]+')',size=size,transform=ax.transAxes,bbox=bbox_props)

def get_random_string(length):
    """Generate a random string consisting of n lowercase letters
    """
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def get_comma_separated_float(config,section,val):
    """Extract list of floats from specified section of configparser object
    """
    if not config[section].get(val):
        return []
    else:
        return list(map(float,config[section].get(val).split(",")))

def read_arcinfo(filename,nHeader=6):
    headerDict = {}
    for i in range(nHeader):
        line = linecache.getline(filename,i+1)
        tokens = line.split()
        headerDict[tokens[0].lower()] = tokens[1].lower()
    ncol = int(headerDict['ncols'])
    nrow = int(headerDict['nrows'])
    x0 = float(headerDict['xllcenter'])
    y0 = float(headerDict['yllcenter'])
    cellsize = float(headerDict['cellsize'])
    
    data = np.flipud(np.loadtxt(filename,skiprows=nHeader).reshape((nrow,ncol)))
    x = np.linspace(x0,x0+(ncol-1)*cellsize,ncol)
    y = np.linspace(y0,y0+(nrow-1)*cellsize,nrow)
    
    return x,y,data
    
def write_arcinfo(filename,x,y,z,fmt='%d'):
    ncol = x.shape[0]
    nrow = y.shape[0]
    x0 = x[0]
    y0 = y[0]
    cellsize = y[1] - y[0]
    headerDict ={}
    headerDict['ncols'] = ncol
    headerDict['nrows'] = nrow
    headerDict['xllcenter'] = x0
    headerDict['yllcenter'] = y0
    headerDict['cellsize'] = cellsize
    headerDict['nodata_value'] = -9999.9
    headerList = []
    toolazy = ['ncols','nrows','xllcenter','yllcenter','cellsize','nodata_value']
    
    for key in toolazy:
        headerList.append("%s %s\n" % (key,headerDict[key]))
    header = ''.join(headerList)
    
    np.savetxt(filename,np.flipud(z),header=header.strip(),comments='',fmt=fmt)

def block_props(x,y,z,block,maxDev=None):
    """
    Calculate block mean and std of data in x,y,z
    block is a vector like GMT -R option with 
    western, eastern, southern and northern block boundary
    
    If maxDev is not None, all individual values in z that 
    deviate by more than maxDev*std from the mean will b removed
    """
    valid = (x>=block[0]) & (x<= block[1]) & (y>=block[2]) & (y<= block[3])
    m = np.mean(z[valid])
    s = np.std(z[valid])
    N = np.where(valid)
    
    if not (maxDev is None):
        absDev = np.abs(z-m)/s
        valid = (x>=block[0]) & (x<= block[1]) & (y>=block[2]) & (y<= block[3]) & (absDev<=maxDev)
        m = np.mean(z[valid])
        s = np.std(z[valid])
        N = np.where(valid)    
    return m,s,N

def blockify(x,y,z,corners,blocksize,maxDev=None):
    """
    Returns blockwise means and standard deviations of property
    described by x,y,z. 
    corners is a vector like GMT -R option. So corners[0] is
    left edge of leftmost block and corners[1] is right edge of
    rightmost block.
    Block size is size of blocks
    number of blocks is calculated from corners
    """
    W = np.arange(corners[0],corners[1],blocksize)
    E = W + blocksize
    S = np.arange(corners[2],corners[3],blocksize)
    N = S + blocksize
    nx = len(E)
    ny = len(N)

    m = np.zeros((ny,nx))
    s = np.zeros((ny,nx))
    
    for i in range(ny):
        for j in range(nx):
            m[i,j],s[i,j],_ = block_props(x,y,z,(W[j],E[j],S[i],N[i]),maxDev)
    return m,s
                  
def blockify_xyz(filename,blocksize,lonSwitch=True,corners=None):
    """
    Turn the data from filename into blocks of size blocksize. It
    is assumed that the file contains 3 columns lon, lat, value
    with no header.
    
    Optionally longitudes above 180 can be converted to negative
    longitudes. 
    By default the corners of the blocks are chosen according to
    the extension of the data. But this can be overwritten if
    a smaller region is desired.
    
    """
    data = np.loadtxt(filename)
    lon = data[:,0]
    lat = data[:,1]
    vals = data[:,2]
    if lonSwitch:
        lon[lon>180] = lon[lon>180] - 360
    if corners is None:
        corners = (lon.min(),lon.max(),lat.min(),lat.max())
    m,s = blockify(lon,lat,vals,corners,blocksize)
    return m,s

def get_blocklims(corners,blocksize):
    W = np.arange(corners[0],corners[1],blocksize)
    E = W + blocksize
    S = np.arange(corners[2],corners[3],blocksize)
    N = S + blocksize
    return W,E,S,N

def read_werami_685(fname,headerrows=13,P_x=True):
    with open(fname,'r') as f:
        lines = f.readlines()
    data = np.loadtxt(fname,skiprows=headerrows)
    lines = lines[:headerrows]
    # Find pressure info
    for i,line in enumerate(lines):
        if 'P(bar)' in line:
            P_info = i
            break
    for i,line in enumerate(lines):
        if 'T(K)' in line:
            T_info = i
            break
           
    P_0 = float(lines[P_info+1].strip())
    dP = float(lines[P_info+2].strip())
    nP = int(lines[P_info+3].strip())
    P_axis = np.linspace(P_0,P_0+dP*(nP-1),nP)
    T_0 = float(lines[T_info+1].strip())
    dT = float(lines[T_info+2].strip())
    nT = int(lines[T_info+3].strip())
    T_axis = np.arange(T_0,T_0+dT*(nT),dT)
    
    if P_x:
        data = data.reshape((nT,nP,data.shape[1]))
    else:
        data = data.reshape((nP,nT,data.shape[1]))
    
    return P_axis,T_axis,data


class ETA:
    """Slightly configurable object to report progress to command line
    """
    def __init__(self,n,step=1.0,wait_time=0):
        self.n = n
        self.ix = 0
        self.wait_time = wait_time
        self.t_start = time.time()
        self.step = step
        self.last_message = 0
        self.last_time = self.t_start
        
    def __call__(self):
        t = time.time()
        self.ix = self.ix + 1
        if 100.0*(self.ix-self.last_message)/self.n >= self.step and (t-self.last_time) > self.wait_time:
            print('%.1f %% complete ETA %.2f s' % (((100.0*self.ix)/self.n),(t-self.t_start)/self.ix*(self.n-self.ix)))
            self.last_message = self.ix#
            self.last_time = t
            return True

class LazyTuple(tuple):
    """Just a sub-class of tuple that has .copy()
    """
    def __init__(self,t):
        self.t=t
    def copy(self):
        return LazyTuple(self.t)

def estimate_total_correlation(X):
    """Calculate the correlation coefficient of matrix observations
    The matrix X can have arbitrary dimensions, but it is assumed
    that the first dimension contains the rows of observations
    """
    return np.corrcoef(X.reshape((X.shape[0],-1)),rowvar=False)

def estimate_spatial_correlation(X):
    """Calculate spatial correlation of matrix observations
    First dimension are the repetitions, second dimension is
    spatial position and last dimension is quantity in question
    """
    R = []
    for i in range(X.shape[2]):
        R.append(np.corrcoef(X[:,:,i],rowvar=False))
    return R    

class Enumerator:
    """Used to generate a dict for converting strings to ints
    """
    def __init__(self):
        self.internal = dict()
        self.back = []
        self.k = 0
    
    @staticmethod
    def from_dict(d):
        enum = Enumerator.__new__(Enumerator)
        enum.internal = d
        enum.back = [None] * len(d)
        for key in d:
            enum.back[d[key]] = key
        enum.k = len(d)
        return enum
    
    def add(self,s):
        """Add a new string element, returns the index if it is new
        """
        if s in self.internal:
            return None
        else:
            self.internal[s] = self.k
            self.back.append(s)
            self.k = self.k+1
            return self.k-1
        
def distribute_circles(samples,domain_size,n_candidates=10000):
    """Distribute circles of given size in a rectangular domain
    notebook:: D
    
    Parameters
    ----------
        samples :
        Desired radii of the circles
    domain_size :
        Model extent
    n_candidates :
        How many random points to generate to try to place the circles.
        Must be larger than len(samples).
    """
    Lx,Ly = domain_size
    samples = np.sort(samples)[::-1]
    candidates = np.random.random((n_candidates,2))
    candidates[:,0] *= Lx
    candidates[:,1] *= Ly
    alive = np.ones(len(candidates),dtype=bool)

    points = np.zeros((len(samples),2))
    radii = np.zeros(len(samples))
    for i in range(len(samples)):
        if i>0:
            # Check distance with existing circles
            cdist = spatial.distance.cdist(points[:i],candidates[alive])
            options = np.all(cdist > (radii[:i][:,None]+samples[i]),axis=0)
            if options.sum() == 0:
                print('Nothing found for sample',i)
                continue
            else:
                points[i] = candidates[alive][np.argmax(options)]
        elif i==0:
            points[i] = candidates[0]

        radii[i] = samples[i]
        dx = points[i,0] - candidates[:,0]
        dy = points[i,1] - candidates[:,1]
        r = np.sqrt(dx**2+dy**2)
        alive[radii[i]>r] = False
        if alive.sum() == 0:
            print('No more alive')
            break
    return points,radii

def in_range(x,interval):
    return (x>=interval[0]) & (x<=interval[1])

def in_box(lon,lat,lon_lim,lat_lim):
    lon_ok = in_range(lon,lon_lim)
    lat_ok = in_range(lat,lat_lim)
    return lon_ok & lat_ok