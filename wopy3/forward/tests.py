import wopy3.forward.seismo as seismo
import numpy as np
import wopy3.forward.cartesian as cartesian
import wopy3.forward.two_D as two_D

def test_gpdc():
    thicknesses = np.array([15.0,15.0,50.0]) * 1e3
    vs = np.array([3200,3900,4500])
    vp = 1.8 * vs
    density = np.array([2600,2900,3200])

    result = seismo.run_geopsy(thicknesses,vp,vs,density,dict(min=0.001,max=0.25,verbosity=2))
    assert result.size > 0

def test_2d_kernel():
    x0,x1 = 1,10
    z0,z1 = 1,10

    multi = lambda f:f(x1,z1) - f(x0,z1) - f(x1,z0) + f(x0,z0)

    yv = np.linspace(1,1000,20)
    for y in yv:
        partial = lambda x,z: cartesian.K_z(x,y,z) - cartesian.K_z(x,-y,z)
        # print(multi(partial),multi(two_D.kernel_2d))
    result1,result2 = multi(partial),multi(two_D.kernel_2d)
    assert np.abs(result1-result2) < 0.001