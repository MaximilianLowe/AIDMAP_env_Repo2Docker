"""
This contains helper functions for magnetic inversion.
There used to be more stuff in this script, but most of it got superseded by new things
in bacu.py and bacu_run.py
"""

from .SH_filter import *

import numpy as np
import subprocess

def run_magtess(tess_file,stat_file,comp='z'):
    """Run a single magtess component for a tess_file and all stations
    """
    prog_dict = dict()
    prog_dict["x"] = "tessbx"
    prog_dict["y"] = "tessby"
    prog_dict["z"] = "tessbz"
    
    with open(stat_file) as f:
        sub = subprocess.Popen([prog_dict[comp],tess_file],stdin=f,
                stdout=subprocess.PIPE,universal_newlines=True)  
        (stdoutdata,_) = sub.communicate()
    return np.genfromtxt(stdoutdata.split('\n'))

    
def magnetize_tesseroids_from_interface(lonGrid,latGrid,topGrid,bottomGrid,suscGrid,igrfDict=None,year=2014.0,magGrid=None):
    """Create magnetized tesseroids from two interfaces (bottom and top).
    
    Create magnetized tesseroids from two interfaces (bottom and top z positive DOWN). 
    Distances in KILOMETERS.
    The susceptibility grid is then used in combination with igrf or a specified magnetization grid to magnetize the
    tesseroids
    """
    assert len(np.unique(lonGrid))==lonGrid.shape[1]
    assert len(np.unique(latGrid))==lonGrid.shape[0]
    assert not (magGrid is None and igrfDict is None)
    tesses=[]
    N = len(lonGrid.ravel())
    dx = (lonGrid.max() - lonGrid.min())/(lonGrid.shape[1]-1)
    dy = (latGrid.max() - latGrid.min())/(lonGrid.shape[0]-1)
    
    if igrfDict is None:
        magField = magGrid.reshape((3,-1))
    else:
        igrfYear = interp_igrf(igrfDict,year)
        igrfYear = AB_to_coeff_vec(igrfYear[0],igrfYear[1])
        vCentr = - 0.5 * (topGrid + bottomGrid)
        design_matrix = get_field_design_matrix_grid( (np.unique(lonGrid),np.unique(latGrid)),13,6371.,6371. + vCentr)
        magField = design_matrix.dot(igrfYear)
        magField[2,:] = -magField[2,:]
    
    for i in range(N):
        lon0 = lonGrid.ravel()[i]
        lat0 = latGrid.ravel()[i]
        top = -1000.0 * topGrid.ravel()[i]
        bottom = -1000.0 * bottomGrid.ravel()[i]
        
        susc = suscGrid.ravel()[i]
        tessString = '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.4f %.2f %.2f %.2f\n' % (
                        lon0-0.5*dx,lon0+0.5*dx,lat0-0.5*dy,lat0+0.5*dy,top,
                        bottom,1.0,susc,magField[0,i],magField[1,i],magField[2,i])
        tesses.append(tessString)
        
    return tesses

def get_TFA(lonGrid,latGrid,heights,mag_grid,igrfDict,year=2014.0):
    """Calculate Total field anomaly from grid and igrf
    
    TFA = |mag_grid + igrf| - |igrf|
    Heights in m
    """
    ny,nx = lonGrid.shape[0],lonGrid.shape[1]
    igrfYear = interp_igrf(igrfDict,year)
    igrfYear = AB_to_coeff_vec(igrfYear[0],igrfYear[1])
    design_matrix = get_field_design_matrix_grid( (np.unique(lonGrid),np.unique(latGrid)),13,6371.,6371. + heights/1000.0)
    magField = design_matrix.dot(igrfYear)
    magField = magField.reshape((3,ny,nx))
    norm1 = np.sqrt( ((magField+mag_grid)**2).sum(0))
    norm2  = np.sqrt((magField**2).sum(0))
    return norm1 - norm2