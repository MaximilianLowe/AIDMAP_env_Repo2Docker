import numpy as np

def mag_dipole(dipoles,magnetizations,stations,**kwargs):
    """
    dim 1: N_stations
    dim 2: N_dipoles
    dim 3: 3
    """
    
    max_size = kwargs.get("max_size",1000000000) # in bytes
    N_stations,N_dipoles = stations.shape[0],magnetizations.shape[0]
    
    partitions = np.ceil(8 * N_stations * N_dipoles / max_size).astype(int)
    ixs = np.array_split(np.arange(N_stations,dtype=int),partitions)
    field = np.zeros((N_stations,3))
    
    for ix in ixs:
        r = stations[ix,None,:] - dipoles[None,:,:]
        r_val = np.sqrt((r**2).sum(-1))
        T1 = 3.0*(magnetizations[None,:,:]*r).sum(-1)/(r_val**5)
        T2 = 1.0/(r_val**3)
        field[ix] =  (T1[:,:,None] * r - T2[:,:,None] * magnetizations).sum(1)
    return field

def grid_to_pointmasses_cartesian(xi,yi,top_grid,bottom_grid,dens_grid,
                          hsplit=1,vsplit=1):
    """Convert a tesseroid grid to a set of point masses (ND-shaped output)
    """
    ny,nx = xi.shape[0],xi.shape[1]
    dx = np.abs(xi[0,1]-xi[0,0])
    dy = np.abs(yi[1,0]-yi[0,0])
    x0 = np.zeros((vsplit,hsplit,hsplit,ny,nx))
    y0 = np.zeros((vsplit,hsplit,hsplit,ny,nx))
    depths = np.zeros(x0.shape)
    masses = np.zeros(x0.shape)
    
    x_shift = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dx
    y_shift = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dy
    for k in range(vsplit):
        if k==0:
            top = top_grid
        else:
            top = (bottom_grid-top_grid)/(1.0*vsplit)*k + top_grid
        if k==vsplit-1:
            bot = bottom_grid
        else:
            bot = (bottom_grid-top_grid)/(1.0*vsplit)*(k+1) + top_grid
        for i in range(hsplit):
            for j in range(hsplit):
                x0[k,i,j,:,:] = xi + x_shift[j]
                y0[k,i,j,:,:] = yi + y_shift[i]
                depths[k,i,j,:,:] = 0.5 * (top+bot)
                
                masses[k,i,j,:,:] = 1e9 * dx * dy / hsplit**2 / vsplit * dens_grid
    return x0,y0,depths,masses

def grid_to_pointmasses_cartesian_chebby(xi,yi,top_grid,bottom_grid,dens_grid,hsplit=1,cheb_order=1):
    # Use existing function to determine horizontal location of point masses
    x0,y0,depths_pm,masses = grid_to_pointmasses_cartesian(xi,yi,top_grid,bottom_grid,dens_grid,hsplit=hsplit,vsplit=cheb_order)
    
    ym = wopy3.forward.forward_calculators.cheb_nodes(cheb_order)
    mid = 0.5 * (depths_pm.min() + depths_pm.max())
    pillar_length = (depths_pm.max()-depths_pm.min())
    depths_pm_scale = (depths_pm - mid) * 2 / pillar_length
    depths_pm_scale = np.clip(depths_pm_scale,-1,1)
    temp = (wopy3.forward.forward_calculators.S_einsum(cheb_order,ym,depths_pm_scale) * masses)
    mass_chebby = temp.sum((1,2))
    
    depths_chebby = ym * pillar_length/2. + mid
    return x0,y0,depths_chebby[:,None,None,None,None],mass_chebby[:,None,:,:,:]


def grid_to_pointmasses_cartesian_chebby_alternative(xi,yi,top_grid,bottom_grid,dens_grid,hsplit=1,cheb_order=1):
    # Use existing function to determine horizontal location of point masses
    # lon_0,lat_0 : (cheb_order,1,hsplit,hsplit)
    # depths_pm: (cheb_order,1,1,1)
    # volume_chebby: (cheb_order,1,hsplit,hsplit)
    x0,y0,depths_pm,masses = grid_to_pointmasses_cartesian(xi,yi,top_grid,bottom_grid,dens_grid,hsplit=hsplit,vsplit=3*cheb_order)
    x0 = x0[:cheb_order,:,:,:,:]
    y0 = y0[:cheb_order,:,:,:,:]
    
    ym = wopy3.forward.forward_calculators.cheb_nodes(cheb_order)
    mid = 0.5 * (top_grid + bottom_grid)
    pillar_length = (bottom_grid - top_grid)
    depths_pm_scale = (depths_pm - mid) * 2 / pillar_length
    depths_pm_scale = np.clip(depths_pm_scale,-1,1)
    temp = (wopy3.forward.forward_calculators.S_einsum(cheb_order,ym,depths_pm_scale) * masses)
    mass_chebby = temp.sum((1))
    
    depths_chebby = ym[:,None,None,None,None] * pillar_length/2. + mid
    depths_chebby = np.tile(depths_chebby,(1,hsplit,hsplit,1,1))
    return x0,y0,depths_chebby,mass_chebby

def periodic_repeat(x,y,*grids):
    Lx = x[-1] + x[1] - x[0]
    Ly = y[-1] + y[1] - y[0]
    nx = len(x)
    ny = len(y)
    
    new_x = np.concatenate((x-Lx,x,x+Lx))
    new_y = np.concatenate((y-Ly,y,y+Ly))
    new_z = np.zeros((3*Ny,3*Nx))
    return (new_x,new_y,*tuple([np.tile(grid,(3,3)) for grid in grids]))

def K_mat(x,y,z):
    """Calculate Nagy (2000) kernels for the `gradient` of a prism.
    This functions needs to be evaluated at all corners of the prism. The
    origin of the coordinate system is in the station.
    """
    r = np.sqrt(x**2+y**2+z**2)
    K = np.zeros((x.shape)+(3,3))
    K[...,0,0] = -np.arctan2(y*z,(x*r))
    K[...,0,1] =  np.log(z+r)
    K[...,0,2] =  np.log(y+r)
    K[...,1,0] = K[...,0,1]
    K[...,1,1] = -np.arctan2(x*z,(y*r))
    K[...,1,2] = np.log(x+r)
    K[...,2,0] = K[...,0,2]
    K[...,2,1] = K[...,1,2]
    K[...,2,2] = -np.arctan2(x*y,(z*r))
    return K

def prism_sensitivity_matrix_mag(centers,sizes,magnetizations,stations):
    """Evaluate the prism magnetic kernel for a set of stations
    Returns the full sensitivity matrix for the susceptibility. Note that no units are assumed and you
    have to take care to use the right constants yourself. If `magnetizations` contains the core field
    in nT, and you want to have an output in nT again, it's enough to divide the results by `4 * pi`.
    Parameters
    ----------
    centers : np.ndarray
        Gives the 3-D coordinates of *centers* of prisms. Shape needs to be Nx3
    sizes : np.ndarray
        Gives the sizes of each prism in 3-D space. Shape needs to be Nx3
    magnetizations : np.ndarray
        Gives the magnetization vector in 3-D space at the center of the prism. Typically this will
        be equal to the core field.
    """
    sensitivity_matrix = np.zeros((stations.shape[0],3,centers.shape[0]))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Calculate correct corner of prisms
                p = centers.copy()
                p[:,0] += 0.5*sizes[:,0] * (-1)**i
                p[:,1] += 0.5*sizes[:,1] * (-1)**j
                p[:,2] += 0.5*sizes[:,2] * (-1)**k
                # Shift to coordinate system with station in the center
                dx = p[None,:,:] - stations[:,None,:]
                K = K_mat(dx[...,0],dx[...,1],dx[...,2])
                sensitivity_matrix += (-1)**(i+j+k) * np.einsum('ijkl,jl->ikj',K,magnetizations)
    return sensitivity_matrix

def K_z(x,y,z):
    """Nagy Kernel for vertical gravity component
    """
    r = np.sqrt(x**2+y**2+z**2)
    return x * np.log(y+r) + y * np.log(x+r) - z * np.arctan2(x*y,z*r)


def prism_sensitivity_matrix_grav(centers,sizes,stations):
    sensitivity_matrix = np.zeros((stations.shape[0],centers.shape[0]))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Calculate correct corner of prisms
                p = centers.copy()
                p[:,0] += 0.5*sizes[:,0] * (-1)**i
                p[:,1] += 0.5*sizes[:,1] * (-1)**j
                p[:,2] += 0.5*sizes[:,2] * (-1)**k
                # Shift to coordinate system with station in the center
                dx = p[None,:,:] - stations[:,None,:]
                K = K_z(dx[...,0],dx[...,1],dx[...,2])
                sensitivity_matrix += (-1)**(i+j+k) * K
    return sensitivity_matrix
