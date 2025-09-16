# WoPy3
This is a collection of Python routines written by Wolfgang Szwillus, Kiel University. These routines were developed as part of my scientific work and I share them here, in the hope that they are useful for other researchers. I work mainly on Windows and cannot help you with Linux or Mac.

## Usage
There is no installation procedure. Just clone the repo somewhere on your machine and add the correct folder to path.

```
import sys
sys.path.append("some-place")
```

## File structure

```
WoPy3
|   func_dump.py: Helper routines, mainly concerning kriging
|   utils.py: Helper routines, mainly concerning plotting
|
└───MCMC: Module for running (transdimensional) MCMC
    |   MCMC.py: Main MCMC routines
    |   distributions.py: Contains common statistical distributions (could often be replaced by scipy.stats)
    |   transdim.py: Proposals and models for transdimensional MCMC
└───forward: Module with a collection of forward calculators
    |   Tpy.py: Bayesian estimation of lithospheric temperature (Lösing et al. 2020)
    |   cartesian.py: (Magnetic) forward calc in cartesian coords
    |   forward_calculators.py: Mainly gravity forward calculation in spherical coords with point masses
    |   mineos_link.py: An interface to call normal mode calculation software mineos from Python
└───funso: Discontinued attempt at modelling framework for lithospheric applications
└───kriging: Nonstationary kriging in spherical coordinates (Szwillus et al. 2019)
└───maginv: Susceptibility and Curie depth estimation in spherical coordinates
    |   SH_filter.py: Routines for Schmidt semi-normalized spherical harmonics (could be replaced by SHtools)
    |   mag_inv.py: Routines for magnetic inversion in spherical coordinates
    |   spectral_tools.py: Estimates spectra in 2-D cartesian coordinates
    |   bacu.py: Main routines for Curie depth estimation in spherical coordinates (Szwillus et al. subm.)
    |   bacu_run.py: Helper routines for the things in bacu.py

```

## References

*If you make use of our software, please cite the corresponding paper*

Lösing, M., Ebbing, J., & Szwillus, W. (2020). Geothermal heat flux in Antarctica: Assessing models and observations by Bayesian inversion. Frontiers in Earth Science, 8, 105.

Szwillus, W., Afonso, J. C., Ebbing, J., & Mooney, W. D. (2019). Global crustal thickness and velocity structure from geostatistical analysis of seismic data. Journal of Geophysical Research: Solid Earth, 124(2), 1626-1652.

Szwillus, W., Baykiev E., Dilixiati, Y., & Ebbing, J. (subm.) Linearized Bayesian estimation of magnetization and depth to magnetic bottom from satellite data. Submitted to Geophysical Journal International.
