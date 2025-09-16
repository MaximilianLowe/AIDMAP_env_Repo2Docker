"""
Routines for two dimensional forward calculations in Cartesian coordinates
"""

import numpy as np

def approximate_gravity_kernel_2d(x,z,xs,zs):
    """
    Computes the 2-D gravity response kernel for a grid of pointmasses
    
    Parameters
    ----------
    x : np.array
        2-D grid with x locations of sources
    z : np.array
        2-D grid with z locations of sources
    xs : np.array
        1-D array with x locations of stations
    zs : np.array
        1-D array with z locations of stations
    
    Returns
    -------
    K : np.array
        3-D grid of shape (x.shape[0],x.shape[1],xs.shape[0]) giving the
        effect of each grid cell on each station. To get the actual gravity
        response you have to use something like this:
         `g = np.einsum('ijk,ij->k',K,dens_cube-dens_cube.mean(1)[:,None])`
    """
    dx = x[...,None] - xs
    dz = z[...,None] - zs
    K = 2 * dz/(dx**2+dz**2)
    return K

def prism_gravity_kernel_2d(x,z):
    """Evaluate kernel of prism infinite in y direction
    Coordinate system uses station as origin
    """
    r = np.sqrt(x**2+z**2)
    return x*(np.log(r**2)-2) + 2 * z * np.arctan2(x,z)