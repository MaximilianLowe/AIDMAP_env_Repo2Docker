import sympy
import numpy as np
import matplotlib.pyplot as plt
import collections

def SHA_integration(lon,lat,grid,expansion_matrix):
    """Carry out Spherical Harmonic Analysis using numerical integration
    """
    theta = (90-lat)/180.0*np.pi
    #coeffs = 0.5 * (expansion_matrix * np.sin(theta[:,None,None,None]) * grid[:,:,None,None]).sum((0,1)) * (lat[1]-lat[0]) * (lon[1]-lon[0]) / (180.0**2) * np.pi**2
    coeffs = np.einsum('ijlm,ij,i->lm',expansion_matrix,grid,np.sin(theta)) * (lat[1]-lat[0]) * (lon[1]-lon[0]) / (180.0**2) * np.pi**2 / 2
    coeffs[:,0] = coeffs[:,0] * 2
    return coeffs


class Fraction:
    def __init__(self,num,den):
        self.num=num
        self.den=den
        
    def mult_num(self,num):
        self.num *= num
        return self
    
    def mult_den(self,den):
        self.den *= den
        return self
        
    def val(self):
        return self.num/self.den
    
    def mult(self,other):
        self.num *= other.num
        self.den *= other.den
        return self
    
    def add(self,other):
        self.num = self.num * other.den + other.num * self.den
        self.den = self.den * other.den
        return self
        
    def copy(self):
        return Fraction(self.num,self.den)
    
    def shorten(self):
        obj = sympy.Rational(self.num/self.den)
        self.num = sympy.Integer(obj.p)
        self.den = sympy.Integer(obj.q)

def binomial_fracs(n,k):
    return sympy.factorial(n),sympy.factorial(k)*sympy.factorial(n-k)
    
def power_of_sine_fraction(n):
    fractions = []
    cos_pow = []
    if n%2 == 0:
        for k in range(n//2):
            F = Fraction(*binomial_fracs(n,k))
            F.mult_num((-1)**(n//2-k))
            F.mult_den(2**(n-1))
            fractions.append(F)
            cos_pow.append(n-2*k)
        F = Fraction(*binomial_fracs(n,n//2))
        F.mult_den(2**n)
        fractions.append(F)
        cos_pow.append(0)
        kind = 'c'
    else:
        for k in range((n-1)//2+1):
            F = Fraction(*binomial_fracs(n,k))
            F.mult_num((-1)**((n-1)//2-k))
            F.mult_den(2**(n-1))
            fractions.append(F)
            cos_pow.append(n-2*k)
            kind = 's'
    return fractions,cos_pow,kind

def multiply_trig_series(fractions,cos_pow,kind):
    if cos_pow[-1] == 0:
        new_fractions = [Fraction(0,1) for k in range(len(fractions))]
    else:
        new_fractions = [Fraction(0,1) for k in range(len(fractions)+1)]
    new_pow = []
    new_pow.append(cos_pow[0]+1)
    for k,p in enumerate(cos_pow):
        F = fractions[k].copy().mult_den(2)
        if p>0:
            new_fractions[k].add(F)
            new_fractions[k+1].add(F)
            new_pow.append(p-1)
        else:
            new_fractions[k].add(F).add(F)
    if kind == 's' and new_pow[-1]==0:
        new_fractions = new_fractions[:-1]
        new_pow = new_pow[:-1]
    return new_fractions,new_pow

class TrigSeries:
    @staticmethod
    def make_Pnn(n):
        factor = Fraction(sympy.factorial(2*n),sympy.factorial(n)*2**n)
        fractions,cos_pow,kind = power_of_sine_fraction(n)
        new_fractions = [Fraction(factor.num,factor.den).mult(F) for F in fractions]
        return TrigSeries(new_fractions,cos_pow,kind)
    
    def __init__(self,fractions,cos_pow,kind):
        """This class should be immutable (unlike fraction...)
        """
        self.fractions = fractions
        self.cos_pow = cos_pow
        self.kind = kind
        if self.kind == 'c':
            self.func = sympy.cos
        elif self.kind =='s':
            self.func = sympy.sin
        else:
            raise ValueError('Incorrect kind')
    
    @property
    def max_n(self):    
        return self.cos_pow[0]
        
    def multiply_trig_series(self):
        # Multiply with cos(x) and return a NEW object
        new_fractions,cos_pow = multiply_trig_series(self.fractions,self.cos_pow,self.kind)
        return TrigSeries(new_fractions,cos_pow,self.kind)
    
    def mult(self,a,b=1):
        # Multiply with a scalar
        new_fractions = [F.copy().mult_num(a).mult_den(b) for F in self.fractions]
        return TrigSeries(new_fractions,self.cos_pow,self.kind)
    
    def add(self,other):
        assert other.kind == self.kind
        new_cos_pow = np.sort(np.union1d(self.cos_pow,other.cos_pow))[::-1]
        new_fractions = dict()
        for p in new_cos_pow:
            new_fractions[p] = Fraction(0,1)
        
        for k,p in enumerate(self.cos_pow):
            new_fractions[p].add(self.fractions[k])
        for k,p in enumerate(other.cos_pow):
            new_fractions[p].add(other.fractions[k])
        # Turn into list
        new_fractions = [new_fractions[p] for p in new_cos_pow]
        return TrigSeries(new_fractions,new_cos_pow,self.kind)
    
    def to_symbolic(self,x):
        expr = 0
        for i,p in enumerate(self.cos_pow):
            expr = expr + self.fractions[i].val() * self.func(p*x)
        return expr
    
    def shorten(self):
        for F in self.fractions:
            F.shorten()
        return self

class LegCollection:
    def __getitem__(self,i):
        return self.dict[i]
    
    def __setitem__(self,i,val):
        self.dict[i] = val
    
    def __init__(self,max_n):
        self.max_n = max_n
        self.dict = dict()
        for n in range(max_n+1):
            self[n,n] = TrigSeries.make_Pnn(n).shorten()
            
            
        # Calculate p_n_n-1
        for n in range(1,max_n+1):
            self[n,n-1] = self[n-1,n-1].mult(2*n-1)
            self[n,n-1] = self[n,n-1].multiply_trig_series().shorten()
            
    def _recursion(self):
        max_n = self.max_n
        for m in range(0,max_n):
            for n in range(2+m,max_n+1):
                new_series = self[n-1,m].mult(2*n-1,n-m).multiply_trig_series()
                new_series = new_series.add(self[n-2,m].mult(-(n+m-1),n-m))
                self[n,m] = new_series.shorten()
    
    def normalize_coeffs(self,n,m):
        # Apply normalization
        # Since there is a square root involved, we first
        # square everything, then multiply as integers,
        # and then take the square root
        series = self[n,m]
        vals = np.zeros(len(series.cos_pow))
        pre_num = (2*n+1) * sympy.factorial(n-m)
        pre_den = 2 * sympy.factorial(n+m)
        for i,F in enumerate(series.fractions):
            num2,den2 = F.num**2,F.den**2
            F2 = sympy.Rational(pre_num * num2,pre_den*den2)
            vals[i] = sympy.sqrt(F2) * np.sign(F.num) * np.sign(F.den)
        return vals
        
    def to_coeff_matrix(self,normalize=False):
        max_n = self.max_n
        A = np.zeros((max_n+1,max_n+1,max_n+1))
        for n in range(max_n+1):
            for m in range(n+1):
                series = self[n,m]
                if normalize:
                    A[n,m,series.cos_pow] = self.normalize_coeffs(n,m)
                else:
                    A[n,m,series.cos_pow] = [F.val() for F in series.fractions]
        return A
    
    def fourier_series_tesseroid(self,W,E,lmax=None):
        if lmax is None:
            lmax = self.max_n
        m_vec = np.arange(0,lmax+1)
        phi1,phi2 = W/180.0*np.pi,E/180.0*np.pi
        result = np.zeros((lmax+1,2))
        result[m_vec!=0,0] = 2.0/m_vec[m_vec!=0] * (np.sin(m_vec[m_vec!=0] * phi2) - np.sin(m_vec[m_vec!=0] * phi1))# Cosinus part integrated
        result[m_vec==0,0] = phi2-phi1
        result[m_vec!=0,1] = -2.0/m_vec[m_vec!=0] * (np.cos(m_vec[m_vec!=0] * phi2) - np.cos(m_vec[m_vec!=0] * phi1)) #Sinus part integrated
        return result/(2.0*np.pi)

def normalization(n,m):
    return sympy.sqrt((2*n+1)/2 * sympy.factorial(n-m)/sympy.factorial(n+m))

def assoc_legendre_norm(n,m,x):
    return (normalization(n,m) * sympy.assoc_legendre(n,m,x))

class Normalizor:    
    symbol_n,symbol_m = sympy.symbols('n m',integer=True,positive=True)    
    
    def __init__(self):
        self.expr1 = (normalization(Normalizor.symbol_n,Normalizor.symbol_m)/normalization(Normalizor.symbol_n-1,Normalizor.symbol_m))
        self.expr1 = sympy.sqrt(self.expr1**2).simplify()
        self.expr2 = (normalization(Normalizor.symbol_n,Normalizor.symbol_m)/normalization(Normalizor.symbol_n-2,Normalizor.symbol_m))
        self.expr2 = sympy.sqrt(self.expr2**2).simplify()
        self.func1 = sympy.lambdify((Normalizor.symbol_n,Normalizor.symbol_m),self.expr1)
        self.func2 = sympy.lambdify((Normalizor.symbol_n,Normalizor.symbol_m),self.expr2)

    
    def normalize_ratios(self,n,m):
        # Return normalization ratios for (n,m), (n-1,m) and (n-2,m)
        return self.func1(n,m),self.func2(n,m)
        
class TrigSeriesFloat(TrigSeries):
    @staticmethod
    def from_TrigSeries(other):
        factors = [float(F.val()) for F in other.fractions]
        return TrigSeriesFloat(factors,other.cos_pow,other.kind)
    
    @staticmethod
    def make_Pnn_normalized(n):
        factor = sympy.factorial(2*n)/(sympy.factorial(n)*2**n) * normalization(n,n)
        fractions,cos_pow,kind = power_of_sine_fraction(n)
        new_factors = [float(F.val() * factor) for F in fractions]
        return TrigSeriesFloat(new_factors,cos_pow,kind)
    
    def __init__(self,factors,cos_pow,kind):
        """This class should be immutable (unlike fraction...)
        """
        self.factors = np.asarray(factors)
        self.cos_pow = cos_pow
        self.kind = kind
        if self.kind == 'c':
            self.func = sympy.cos
        elif self.kind =='s':
            self.func = sympy.sin
        else:
            raise ValueError('Incorrect kind')
        
    def multiply_trig_series(self):
        # Multiply self with cos(x) and return a NEW object    
        if self.cos_pow[-1] == 0:
            new_factors = [0 for k in range(len(self.factors))]
        else:
            new_factors = [0 for k in range(len(self.factors)+1)]
        
        new_pow = []
        new_pow.append(self.cos_pow[0]+1)
        for k,p in enumerate(self.cos_pow):
            F = self.factors[k]/2
            if p>0:
                new_factors[k] += F
                new_factors[k+1] += F
                new_pow.append(p-1)
            else:
                new_factors[k] += 2 * F
        if self.kind == 's' and new_pow[-1]==0:
            new_factors = new_factors[:-1]
            new_pow = new_pow[:-1]
        
        return TrigSeriesFloat(new_factors,new_pow,self.kind)

    def multiply_trig_series_sin(self):
        # Multiply self with sin(x) and return a NEW object    
        temp_pow,temp_factors = self._full_vectors()
        new_pow = np.arange(self.max_n+2,dtype=int)
        new_factors = np.zeros(len(new_pow))
        if self.kind == 'c':
            new_factors[1] = temp_factors[0] - 0.5 * temp_factors[2]
            new_factors[2:] = -0.5 * (temp_factors[3:] - temp_factors[1:-2])
            new_kind  = 's'
        elif self.kind == 's':
            new_factors[0] = 0.5 * temp_factors[1]
            new_factors[1] = 0.5 * temp_factors[2]
            new_factors[2:] = 0.5 * (temp_factors[3:] - temp_factors[1:-2])
            new_kind = 'c'
        
        sel = ~(new_factors==0)
        new_pow = new_pow[sel][::-1]
        new_factors = new_factors[sel][::-1]
                
        return TrigSeriesFloat(new_factors,new_pow,new_kind)
    
    
    def _full_vectors(self):
        # Extend to higher powers (which are all zero)
        temp_pow = np.arange(self.max_n+3,dtype=int)
        temp_factors = np.zeros(len(temp_pow))
        temp_factors[self.cos_pow] = self.factors
        return temp_pow,temp_factors
    
    def __mul__(self,a):
        return TrigSeriesFloat(np.array(self.factors)*a,self.cos_pow,self.kind)
    
    def __truediv__(self,a):
        return TrigSeriesFloat(np.array(self.factors)/a,self.cos_pow,self.kind) 
    
    def __add__(self,other):
        if not other.kind == self.kind:
            raise ValueError('Cannot add different kinds')
        
        new_cos_pow = np.sort(np.union1d(self.cos_pow,other.cos_pow))[::-1]
        new_factors = collections.defaultdict(int)
        
        for k,p in enumerate(self.cos_pow):
            new_factors[p] += self.factors[k]
        for k,p in enumerate(other.cos_pow):
            new_factors[p] += other.factors[k]
        new_factors = [new_factors[p] for p in new_cos_pow]
        return TrigSeriesFloat(new_factors,new_cos_pow,self.kind)
    
    def __sub__(self,other):
        return self.__add__(-other)
    
    def __neg__(self):
        return TrigSeriesFloat(-self.factors.copy(),self.cos_pow,self.kind)
    
    def to_symbolic(self,x):
        expr = 0
        for i,p in enumerate(self.cos_pow):
            expr = expr + self.factors[i] * self.func(p*x)
        return expr

class FloatLegCollection(LegCollection):
    """This uses float representation instead of rational and only uses re-normalized polynomials
    """        
    
    def __init__(self,max_n):
        self.max_n = max_n
        self.dict = dict()
        self.float_dict = dict()
        self.A = np.zeros((max_n+1,max_n+1,max_n+1))
        self.normalizor = Normalizor()
        self.shifted_A = None
        # The initial objects are calculated as fraction objects - because of normalization accuracy
        # The recursion then only uses the rational ones
        for m in range(max_n+1):            
            for n in range(m,max_n+1,1):
                if n==m:
                    self[n,m] = TrigSeries.make_Pnn(n).shorten()
                    self.float_dict[n,m] = TrigSeriesFloat.make_Pnn_normalized(n)
                elif n==m+1:
                    self.float_dict[n,m] = self.float_dict[n-1,m].multiply_trig_series() * (2*n-1) * float((normalization(n,m) / normalization(n-1,m)))
                else:
                    ratios = self.normalizor.normalize_ratios(n,m)
                    self.float_dict[n,m] = self.float_dict[n-1,m].multiply_trig_series() * (2*n-1)/(n-m)*ratios[0] - self.float_dict[n-2,m] * (n+m-1)/(n-m)*ratios[1]
    
        for n in range(max_n+1):
            for m in range(n+1):
                obj = self.float_dict[n,m]
                self.A[n,m,obj.cos_pow] = obj.factors
                
    
    def get_shifted_coeffs(self):
        if self.shifted_A is None:
            self.shifted_A = np.zeros((self.max_n+1,self.max_n+1,self.max_n+2))
            for n in range(self.max_n+1):
                for m in range(n+1):
                    temp = self.float_dict[n,m].multiply_trig_series_sin()
                    self.shifted_A[n,m,temp.cos_pow] = temp.factors
        else:
            return self.shifted_A
    
    def evaluate(self,theta):
        n_vec = np.arange(self.max_n+1)
        sin = np.sin(theta*n_vec)
        cos = np.cos(theta*n_vec)
        result = np.zeros((self.max_n+1,self.max_n+1))
        result[:,::2] = np.einsum('i,lmi->lm',cos,self.A[:,::2])
        result[:,1::2] = np.einsum('i,lmi->lm',sin,self.A[:,1::2])
        return result
    
    def integrate(self,a,b):
        n_vec = np.arange(self.max_n+1)
        sin = np.sin(b*n_vec)/n_vec
        sin[n_vec==0] = b
        cos = np.cos(b*n_vec)/n_vec
        cos[n_vec==0] = 0
        result = np.zeros((self.max_n+1,self.max_n+1))
        result[:,::2] = np.einsum('i,lmi->lm',sin,self.A[:,::2])
        result[:,1::2] = np.einsum('i,lmi->lm',-cos,self.A[:,1::2])
        
        sin = np.sin(a*n_vec)/n_vec
        sin[n_vec==0] = a        
        cos = np.cos(a*n_vec)/n_vec
        cos[n_vec==0] = 0
        result[:,::2] -= np.einsum('i,lmi->lm',sin,self.A[:,::2])
        result[:,1::2] -= np.einsum('i,lmi->lm',-cos,self.A[:,1::2])
        return result
    
    def integrate_sin(self,a,b):
        # Integrate after multiplying with sin(x)
        # Note that now the even m correspond to sinus-series
        n_vec = np.arange(self.max_n+2)
        sin = np.sin(b*n_vec)/n_vec
        sin[n_vec==0] = b
        cos = np.cos(b*n_vec)/n_vec
        cos[n_vec==0] = 0
        result = np.zeros((self.max_n+1,self.max_n+1))
        result[:,::2] = np.einsum('i,lmi->lm',-cos,self.shifted_A[:,::2])
        result[:,1::2] = np.einsum('i,lmi->lm',sin,self.shifted_A[:,1::2])
        
        sin = np.sin(a*n_vec)/n_vec
        sin[n_vec==0] = a
        cos = np.cos(a*n_vec)/n_vec
        cos[n_vec==0] = 0
        result[:,::2] -= np.einsum('i,lmi->lm',-cos,self.shifted_A[:,::2])
        result[:,1::2] -= np.einsum('i,lmi->lm',sin,self.shifted_A[:,1::2])
        
        return result
        