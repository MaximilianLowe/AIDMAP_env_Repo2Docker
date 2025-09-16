"""Some elementary tests to see if the magnetic inversion code is working
Note that tessbz.exe needs to be present (and runnable) in the current folder.
"""

import numpy as np
import sys
sys.path.append("D:/repositories")
import os
import wopy3.func_dump
from wopy3.utils import ETA
import wopy3.maginv.bacu
import wopy3.maginv.bacu_run
import matplotlib.pyplot as plt


def init_grid(N,deg_size):
    lon = np.linspace(-deg_size,deg_size,N)
    lat = np.linspace(-deg_size,deg_size,lon.size)
    loni,lati = np.meshgrid(lon,lat)
    pd = wopy3.func_dump.get_pairwise_geo_distance(loni.flatten(),lati.flatten())
    return lon,lat,loni,lati,pd


def test_matern_approximator():
    N = 21
    deg_size = 30.0
    lon,lat,loni,lati,pd = init_grid(N,deg_size)

    approximator = wopy3.maginv.bacu.MaternApproximator(2000,'root')
    nus = np.arange(0.05,2+0.05,0.05)
    rhos = np.arange(100.0,1000.0,100.0)
    approximation_error = np.zeros((nus.size,rhos.size))
    timer = ETA(nus.size*rhos.size,wait_time=5)
    for i,nu in enumerate(nus):
        for j,rho in enumerate(rhos):
            k1 = approximator.interp1d(110.0*pd,(1.0,nu,rho))
            k2 = wopy3.func_dump.C_matern(110.0*pd,(1.0,nu,rho))
            approximation_error[i,j] = np.max(np.abs((k1-k2)/k2))
            timer()
    
    print('Maximum relative error %f'%approximation_error.max())

def test_sensitivity_store_lons():
    N = 5
    deg_size = 30.0
    lon,lat,loni,lati,pd = init_grid(N,deg_size)
    magGrid = np.zeros((3,N,N))
    magGrid[2,:,:] = 50000

    ranges = [(0.02,0.04),(10.0,50.0),(0.1,2.0),(1.0,3.0),(100,600),(300,800),(0.0,4e-5),(0.0,40)]
    n_steps = 3
    linspaces = [np.linspace(r[0],r[1],n_steps) for r in ranges]
    ref_pars = np.array([l[n_steps//2] for l in linspaces])
    chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot = ref_pars
    rho_chi = rho_chi/110.0
    rho_bot = rho_bot/110.0

    lons = np.random.random(10) * 2 * deg_size - deg_size
    lats = np.random.random(10) * 2 * deg_size - deg_size

    sens_matrix_store = wopy3.maginv.bacu.SensitivityMatrixStore(lon,lat,magGrid,linspaces[1],verbose=True,stat_file="ihateyou.txt",tess_file="ihateyou.tess",
                                                                        lons=lons,lats=lats)
    os.remove("ihateyou.txt")
    os.remove("ihateyou.tess")
    design_1 = sens_matrix_store.get_sens_matrix(0.03,30.0)
    design_2 = sens_matrix_store.get_sens_matrix(0.02,20.0)
    design_3 = sens_matrix_store.get_sens_matrix_grid(np.ones((N,N))*0.03,np.ones((N,N))*30.0)

    assert np.allclose(design_3[0],design_1[0]) and np.allclose(design_3[1],design_1[1])
    
    chi_grid = np.ones((N,N)) * 0.03
    chi_grid[:N//2,:N//2] = 0.02
    bot_grid = np.ones((N,N)) * 30.0
    bot_grid[:N//2,:N//2] = 20.0
    
    design_4 = sens_matrix_store.get_sens_matrix_grid(chi_grid,bot_grid)

    
    plt.scatter(lons,lats,25,design_2[0][:,3])
    plt.colorbar()

    plt.figure()
    plt.pcolormesh(design_4[0][0,:].reshape((N,N)))
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(design_4[0][0,:].reshape((N,N))-design_1[0][0,:].reshape((N,N)),vmin=-0.01,vmax=0.01)
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(design_4[0][0,:].reshape((N,N))-design_2[0][0,:].reshape((N,N)),vmin=-0.01,vmax=0.01)
    plt.colorbar()
    plt.show()

    assert np.allclose(design_4[0][0,:].reshape((N,N))[:N//2,:N//2],design_2[0][0,:].reshape((N,N))[:N//2,:N//2])
    assert np.allclose(design_4[0][0,:].reshape((N,N))[N//2:,N//2:],design_1[0][0,:].reshape((N,N))[N//2:,N//2:])

def test_probability_calculator(err_b = 0.0):
    """Test calculation of covariance matrices from BayesProbabilityCalculator
    """
    N = 5
    deg_size = 30.0
    lon,lat,loni,lati,pd = init_grid(N,deg_size)
    magGrid = np.zeros((3,N,N))
    magGrid[2,:,:] = 50000

    ranges = [(0.02,0.04),(10.0,50.0),(0.1,2.0),(1.0,3.0),(100,600),(300,800),(0.0,4e-5),(0.0,40)]
    n_steps = 3
    linspaces = [np.linspace(r[0],r[1],n_steps) for r in ranges]
    ref_pars = np.array([l[n_steps//2] for l in linspaces])
    repetitions = 3

    syn_bot = np.zeros((repetitions,N,N))
    syn_chi = np.zeros((repetitions,N,N))

    sens_matrix_store = wopy3.maginv.bacu.SensitivityMatrixStore(lon,lat,magGrid,linspaces[1],verbose=True,stat_file="ihateyou.txt",tess_file="ihateyou.tess")

    matern_approximator = wopy3.maginv.bacu.MaternApproximator(2000,'root')
    probability_calculator = wopy3.maginv.bacu.BayesProbabilityCalculator(pd,sens_matrix_store,matern_approximator)
    probability_calculator_2 = wopy3.maginv.bacu_run.make_probability_calculator(lon,lat,300e3,50000,linspaces[1],matern_approximator_N=2000)
    
    A_design,B_design = sens_matrix_store.get_sens_matrix(ref_pars[0],ref_pars[1])

    for k2 in range(repetitions):
        syn_chi[k2],syn_bot[k2] = wopy3.maginv.bacu.synthetic_generation(lon,lat,(A_design,B_design),ref_pars[0],ref_pars[1],
                                                                            (ref_pars[6],ref_pars[2],ref_pars[4]),
                                                                            (ref_pars[7],ref_pars[3],ref_pars[5]))

    for i in range(len(linspaces)):
        for k1 in range(n_steps):
            pars = ref_pars.copy()
            pars[i] = linspaces[i][k1]
            chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot = pars 
            rho_chi = rho_chi/110.0
            rho_bot = rho_bot/110.0
            sigma_chi_mat = probability_calculator_2.get_sigma_chi_mat(sigma_chi,nu_chi,rho_chi)
            sigma_bot_mat = probability_calculator_2.get_sigma_bot_mat(sigma_bot,nu_bot,rho_bot)
            sigma_B_mat,_ = probability_calculator_2.get_sigma_B_mat(
                        pars=(chi_0,bot_0,nu_chi,nu_bot,rho_chi,rho_bot,sigma_chi,sigma_bot),err_b=err_b)
            A_design,B_design = probability_calculator_2.sensitivity_calculator(pars[0],pars[1])
            
            A_design_2,B_design_2 = sens_matrix_store.get_sens_matrix(pars[0],pars[1])
            sigma_chi_mat_2 = matern_approximator.interp1d(pd,(sigma_chi,nu_chi,rho_chi))
            sigma_bot_mat_2 = matern_approximator.interp1d(pd,(sigma_bot,nu_bot,rho_bot))
            sigma_B_mat_2 = A_design_2.dot(sigma_chi_mat.dot(A_design_2.T)) + B_design_2.dot(sigma_bot_mat.dot(B_design_2.T))
            sigma_B_mat_2 += err_b * np.eye(sigma_B_mat_2.shape[0])

            assert np.allclose(sigma_chi_mat,sigma_chi_mat_2) and np.allclose(sigma_bot_mat,sigma_bot_mat_2) 
            assert np.allclose(A_design,A_design_2)
            assert np.allclose(B_design,B_design_2)
            assert np.allclose(sigma_B_mat,sigma_B_mat_2)


def test_hyperparameter_inversion():
    N = 5
    deg_size = 30.0
    lon,lat,loni,lati,pd = init_grid(N,deg_size)

    ranges = [(0.02,0.04),(10.0,50.0),(0.1,2.0),(1.0,3.0),(100,600),(300,800),(0.0,4e-5),(0.0,40),(0.0,1.0)]
    n_steps = 3
    linspaces = [np.linspace(r[0],r[1],n_steps) for r in ranges]
    ref_pars = np.array([l[n_steps//2] for l in linspaces])

    lons = loni.flatten()
    lats = lati.flatten()

    wopy3.maginv.bacu_run.hyperparameter_inversion(lon,lat,lons,lats,300e3,np.zeros(loni.size),50000,ranges,10,2,n_steps,PT_cold_fraction=1.0)
    wopy3.maginv.bacu_run.hyperparameter_inversion(lon,lat,lons,lats,300e3,np.zeros(loni.size),50000,ranges,10,2,n_steps,PT_cold_fraction=1.0,fixed_ix=[0,1],fixed_par_vals=[0.03,40.0])
    wopy3.maginv.bacu_run.hyperparameter_inversion(lon,lat,lons,lats,300e3,np.zeros(loni.size),50000,ranges,10,2,n_steps,PT_cold_fraction=1.0,
                                                    fixed_ix=[0,1,2,3,4,5,6],fixed_par_vals=[0.03,40.0,0.5,0.5,1000.0,1000.0,1.0e-5])

if __name__ == "__main__":
    #test_matern_approximator()
    test_sensitivity_store_lons()
    #test_hyperparameter_inversion()