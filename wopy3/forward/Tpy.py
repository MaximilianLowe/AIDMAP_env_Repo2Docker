import numpy as np

class TempCalc:
    """1-D heat conduction simulation
    """
    def __init__(self,moho,isotherm_depths,surface_temp):
        self.isotherm_depths = isotherm_depths
        self.moho = moho
        self.T0 = surface_temp
    def __call__(self,args):
        args = np.asarray(args)
        k1,k2,A,qD = args.T
        T = np.zeros(k1.shape+(len(self.isotherm_depths),))
        moho = self.moho
        for i,z in enumerate(self.isotherm_depths):
            if z<=moho:
                T[...,i] = self.T0 + (qD+A*moho)/k1 * z - A/(2.0*k1) * z**2
            else:
                T[...,i] = self.T0 + moho*(qD/k1+0.5*A*moho/k1-qD/k2) + qD/k2 * z
        return T
    
    def get_temp(self,args,depths):
        args = np.asarray(args)
        k1,k2,A,qD = args.T
        T = np.zeros(k1.shape+(len(depths),))
        moho = self.moho
        for i,z in enumerate(depths):
            if z<=moho:
                T[...,i] = self.T0 + (qD+A*moho)/k1 * z - A/(2.0*k1) * z**2
            else:
                T[...,i] = self.T0 + moho*(qD/k1+0.5*A*moho/k1-qD/k2) + qD/k2 * z
        return T
        
    def get_surface_heat_flow(self,args):
        k1,k2,A,qD = args.T
        return qD + self.moho * A
        
    def get_isotherms(self,args,Ts):
        """
        Theoretically there should be no decrease of temperature with depth,
        so that the temperature curve is monotone and there should only ever be
        one valid solution ...
        """
        args = np.asarray(args)
        k1,k2,A,qD = args.T
        q0 = qD + self.moho * A
        Tm = self.T0 + (qD+A*self.moho)/k1 * self.moho - A/(2.0*k1) * self.moho**2
        depths = np.zeros(k1.shape+(len(Ts),))
        for i,T in enumerate(Ts):
            sol_2 = 1.0/A * (q0 - np.sqrt(q0**2 - 2.0*A*k1*(T-self.T0)))
            depths[T<=Tm,i] = sol_2[T<=Tm]
            depths[T>Tm,i] = (self.moho + (T-Tm)/qD * k2)[T>Tm]
        return depths