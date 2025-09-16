"""
A collection of simplified seismological routines for use in joint
inversions in cartesian coordinates.
"""

import numpy as np
import subprocess
import os.path

GPDC_FOLDER = "geopsypack-win64-3.4.2/bin/"

def calc_rf(vp,vs,dens,dz,rayparameter=0.04*1e-3):
    """Calculate amplitude and runtime of P-S converted wave
    Parameters
    ----------
    vp : np.array
        First dimension should run over depth
    vs : np.array
    dens : np.array
    dz : float
        The vertical spacing. Units should be compatible with vp and vs
    rayparameter : float
        Describes the slowness of the incident P-wave. Should have units
        compatible with vp,vs and dz.

    Returns
    -------
    lagtime : np.array
        For each interface inside the model, it gives the travel time 
        difference between a P-wave and a S-wave originating from this
        interface.
    amplitude : np.array
        Amplitude of the converted S-wave, assuming an amplitude of one
        for the incident P-wave.

    """
    lagtime = np.cumsum(1.0/vs - 1.0/vp,axis=0) * dz
    lagtime = lagtime[1:]
    amplitudes_1 = 2*rayparameter * (vs[1:] - vs[:-1])
    amplitudes_2 = 2*(dens[1:]*vs[1:]**2 - dens[:-1]*vs[:-1]**2) * vp[:1]*vs[1:] /(dens[:-1]*vs[:-1]+dens[1:]*vs[1:]) * rayparameter**3
    amplitudes = amplitudes_1 + amplitudes_2
    return lagtime,amplitudes

def bin_rf(lagtime,amplitudes,dt=0.1):
    """Convert output of calc_rf into traces for visualization
    
    Parameters
    ----------
    lagtime : np.array
    amplitudes : np.array
    dt : float, optional
        Time interval of the time axis
    Returns
    -------
    t_ax : np.array
    A_binned : np.array
    """
    bin_ix = np.floor(lagtime / dt).astype(int)
    t_ax = np.arange(0,bin_ix.max()+1) * dt
    A_binned = np.zeros((bin_ix.max()+1,amplitudes.shape[1]))
    for j in range(amplitudes.shape[1]):
        for b in range(bin_ix.max()+1):
            A_binned[b,j] = amplitudes[bin_ix[:,j]==b,j].sum()
    return t_ax,A_binned

def write_geopsy_model(fname,thicknesses,vp,vs,density):
    """Write a layered model for calculation with gpdc to file
    Returns
    -------
    result: np.array
        First column contains frequency and second column 
        phase slowness (not velocity!)
    """
    M = np.vstack((thicknesses,vp,vs,density)).T
    np.savetxt(fname,M,header=str(thicknesses.shape[0]),comments='')

def run_geopsy(thicknesses,vp,vs,density,options_dict=dict(),tmp_file='oink.model',out_file='oink.out'):
    """Run GPDC dispersion calculation 
    Parameters
    ----------
    thicknesses : np.array
    vp : np.array
    vs : np.array
    density : np.array
    options_dict : dict, optional
        All keys will be passed to GPDC as options. Common options include:
        -R -> Number of Rayleigh wave modes to calculate
        -L -> Number of Love modes to calculate
        -min -> Minimum frequency
        -max -> Maximum frequency
        -step -> Frequency step
        -s -> Defines sampling type (can be period, frequency or log (default))
        -f -> Output only highest mode instead of all
    """
    write_geopsy_model(tmp_file,thicknesses,vp,vs,density)
    args = [tmp_file]
    for key in options_dict:
        args.append('-'+key)
        args.append(str(options_dict[key]))
    com = subprocess.Popen([os.path.join(GPDC_FOLDER,'gpdc.exe')] + args ,stdout=open(out_file,'w'))
    com.wait()
    result = np.loadtxt(out_file)
    return result

def scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Full Zoeppritz solution, considered the definitive solution.
    Calculates the angle dependent p-wave reflectivity of an interface
    between two mediums.
    Originally written by: Wes Hamlyn, vectorized by Agile.
    Returns the complex reflectivity.
    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
    Returns:
        ndarray. The exact Zoeppritz solution for all modes at the interface.
            A 4x4 array representing the scattering matrix at the incident
            angle theta1.
            
    elements = np.array(
        [
            ["PdPu", "SdPu", "PuPu", "SuPu"],
            ["PdSu", "SdSu", "PuSu", "SuSu"],
            ["PdPd", "SdPd", "PuPd", "SuPd"],
            ["PdSd", "SdSd", "PuSd", "SuSd"],
        ]
    )
    """
    # Taken from here: https://github.com/agilescientific/bruges/blob/main/bruges/reflection/reflection.py
    theta1 *= np.ones_like(vp1)
    p = np.sin(theta1) / vp1  # Ray parameter.
    theta2 = np.arcsin(p * vp2)  # Trans. angle of P-wave.
    phi1 = np.arcsin(p * vs1)    # Refl. angle of converted S-wave.
    phi2 = np.arcsin(p * vs2)    # Trans. angle of converted S-wave.

    # Matrix form of Zoeppritz equations... M & N are matrices.
    M = np.array([[-np.sin(theta1), -np.cos(phi1), np.sin(theta2), np.cos(phi2)],
                  [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                  [2 * rho1 * vs1 * np.sin(phi1) * np.cos(theta1),
                   rho1 * vs1 * (1 - 2 * np.sin(phi1) ** 2),
                   2 * rho2 * vs2 * np.sin(phi2) * np.cos(theta2),
                   rho2 * vs2 * (1 - 2 * np.sin(phi2) ** 2)],
                  [-rho1 * vp1 * (1 - 2 * np.sin(phi1) ** 2),
                   rho1 * vs1 * np.sin(2 * phi1),
                   rho2 * vp2 * (1 - 2 * np.sin(phi2) ** 2),
                   -rho2 * vs2 * np.sin(2 * phi2)]])

    N = np.array([[np.sin(theta1), np.cos(phi1), -np.sin(theta2), -np.cos(phi2)],
                  [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                  [2 * rho1 * vs1 * np.sin(phi1) * np.cos(theta1),
                   rho1 * vs1 * (1 - 2 * np.sin(phi1) ** 2),
                   2 * rho2 * vs2 * np.sin(phi2) * np.cos(theta2),
                   rho2 * vs2 * (1 - 2 * np.sin(phi2) ** 2)],
                  [rho1 * vp1 * (1 - 2 * np.sin(phi1) ** 2),
                   -rho1 * vs1 * np.sin(2 * phi1),
                   - rho2 * vp2 * (1 - 2 * np.sin(phi2) ** 2),
                   rho2 * vs2 * np.sin(2 * phi2)]])

    M_ = np.moveaxis(np.squeeze(M), [0, 1], [-2, -1])
    A = np.linalg.inv(M_)
    N_ = np.moveaxis(np.squeeze(N), [0, 1], [-2, -1])
    Z_ = np.matmul(A, N_)
    return np.transpose(Z_, axes=list(range(Z_.ndim - 2)) + [-1, -2])


def propagate_wave(model,incidence_angle,max_iterations=100,verbosity = 0,threshold=1e-4):
    """Propagate a geometric P-wave through a layered model

    At each interface in the model, the conversions and reflections are evaluated using
    Zoeppritz equation. Waves are tracked as long as their amplitude is above ``threshold``,
    where the incident P-wave has amplitude 1.

    TODO: There is no reflection at the top!
    TODO: A constant incidence angle is used, which is not quite correct

    Parameters
    ----------
    model : np.array
        Each row of model gives a layer as thickness, vp, vs, dens. Units should be all in SI
    incidence_angle : float
        Angle of incoming wave. It is assumed constant throughout propagation.
    max_iterations : int, optional
        Number of propgatations steps to perform. Note: You can set this to a relatively high
        value normally because the propgatation stops, once there are no more waves with
        amplitude above ``threshold``
    verbosity : int, optional
        1 -> print number of active waves, 2-> spam console with every wave
    threshold : float, optional
        Amplitude level below which waves are no longer tracked.
    """
    scattering_ix = np.zeros((2,2),dtype=int)
    scattering_ix[0,0] = 2 # in is P up
    scattering_ix[1,0] = 3 # in is S up
    scattering_ix[0,1] = 0 # in is P down
    scattering_ix[1,1] = 1 # in is S down
    
    # The order in the scattering matrix is different for rows and cols ... rolls eyes ...
    type_order =[[0,0],[1,0],[0,1],[1,1]] # Pu,Su,Pd,Sd
    
    Ms = scattering_matrix(model[1:,1],model[1:,2],model[1:,3],model[:-1,1],model[:-1,2],model[:-1,3],incidence_angle/180.0*np.pi)
    if Ms.ndim == 2:
        Ms = Ms[None,:,:]
    
    n_layers = model.shape[0]
    
    incoming_wave = [0,n_layers-1,0,1,[n_layers-1],[]] # Type (P/S), Layer, Direction (Up/Down), Amplitude, Traversed Layers (P), Traversed Layers (S)
    arrivals = []
    active_waves = [incoming_wave]
    for _ in range(max_iterations):
        new_waves = []
        for wave in active_waves:
            interface_ix = wave[1]-1+wave[2]
            if interface_ix >= n_layers-1:
                # Wave has left the model downwards
                continue
            elif interface_ix < 0:
                # Wave has arrived at the surface
                arrivals.append(wave)
                continue
            scatter_factors = Ms[interface_ix,:,scattering_ix[wave[0],wave[2]]]
            for i,(ps,direction) in enumerate(type_order):
                P_traversed = wave[4].copy()
                S_traversed = wave[5].copy()
                if ps == 0:
                    P_traversed.append(interface_ix+direction)
                else:
                    S_traversed.append(interface_ix+direction)
                new_wave = [ps,interface_ix + direction,direction,wave[3]*scatter_factors[i],P_traversed,S_traversed]
                if verbosity >= 2:
                    print('Input wave',wave)
                    print('Output wave',new_wave)
                if np.abs(new_wave[3])>threshold:
                   new_waves.append(new_wave) 
        active_waves = new_waves
        if verbosity >= 1:
            print(len(active_waves),end=",")
        if len(active_waves) == 0:
            break
    return arrivals

def arrivals_to_array(arrivals,model,unit_factor = 1000):
    # Convert arrivals to time
    arrival_array = np.zeros((len(arrivals),3))
    for i,wave in enumerate(arrivals):
        arrival_array[i,1] = wave[3]
        P_time = np.sum(model[wave[4],0]/model[wave[4],1]) * unit_factor
        S_time = np.sum(model[wave[5],0]/model[wave[5],2]) * unit_factor
        arrival_array[i,0] = S_time + P_time
        arrival_array[i,2] = wave[0]
    return arrival_array