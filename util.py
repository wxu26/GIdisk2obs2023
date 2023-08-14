import numpy as np

"""
=======================================================================
Constants
=======================================================================
"""
import astropy.constants as const
c_light = const.c.cgs.value
mp = const.m_p.cgs.value
kB = const.k_B.cgs.value
h = const.h.cgs.value
sigma_SB = const.sigma_sb.cgs.value
G = const.G.cgs.value
au = const.au.cgs.value
pc = const.pc.cgs.value
Msun = const.M_sun.cgs.value
yr = 365*24*3600
pi = np.pi

small_number = 1.e-8

"""
=======================================================================
EoS
=======================================================================
"""
# piecewise-linear interpolation based on Kunz & Mouschovias 2009
def get_gamma_scalar(T):
    T_to_cgs = 63.52144902566
    x = T/T_to_cgs
    if (x< 0.15742713923228946 ): y= 1.6666666666666674 ;
    elif (x< 0.7840818852092962 ): y= -0.029510344497522694 *x + 1.6713123957786717 ;
    elif (x< 1.2692464735582838 ): y= -0.30536198503015355 *x + 1.8876026701255741 ;
    elif (x< 2.412452061925896 ): y= -0.137250136180486 *x + 1.6742272988097704 ;
    elif (x< 3.5087932817520255 ): y= 0.00045928128751677744 *x + 1.3420099306924729 ;
    elif (x< 6.321611207800625 ): y= 0.01766887702267927 *x + 1.2816250167952665 ;
    elif (x< 9.19447365850311 ): y= 0.0006180548997382237 *x + 1.3894136850298655 ;
    elif (x< 18.436663403919912 ): y= -0.008279871569026645 *x + 1.4712254355622216 ;
    elif (x< 31.485427846457895 ): y= -0.0033079527177521406 *x + 1.3795598412296695 ;
    else: y= 1.2754075346153901 ;
    return y;
get_gamma = np.vectorize(get_gamma_scalar)
def get_cs_scalar(T):
    T_to_cgs = 63.52144902566
    x = T/T_to_cgs
    if (x< 0.15742713923228946 ): y= 1.6666666666666674 ;
    elif (x< 0.7840818852092962 ): y= -0.029510344497522694 *x + 1.6713123957786717 ;
    elif (x< 1.2692464735582838 ): y= -0.30536198503015355 *x + 1.8876026701255741 ;
    elif (x< 2.412452061925896 ): y= -0.137250136180486 *x + 1.6742272988097704 ;
    elif (x< 3.5087932817520255 ): y= 0.00045928128751677744 *x + 1.3420099306924729 ;
    elif (x< 6.321611207800625 ): y= 0.01766887702267927 *x + 1.2816250167952665 ;
    elif (x< 9.19447365850311 ): y= 0.0006180548997382237 *x + 1.3894136850298655 ;
    elif (x< 18.436663403919912 ): y= -0.008279871569026645 *x + 1.4712254355622216 ;
    elif (x< 31.485427846457895 ): y= -0.0033079527177521406 *x + 1.3795598412296695 ;
    else: y= 1.2754075346153901 ;
    return np.sqrt(y*x)*100*au/(1e3*yr);
get_cs = np.vectorize(get_cs_scalar)
# cs^2 = gamma*p/rho
# u = p/rho/(gamma-1)
def get_u(T):
    gamma = get_gamma(T)
    return get_cs(T)**2/gamma/(gamma-1)


"""
=======================================================================
Misc functions
=======================================================================
"""
# planck function
def B(T,nu):
    #return 2*h*nu**3/c_light**2 * 1/(np.exp(h*nu/(kB*T))-1)
    x = np.exp(-h*nu/(kB*T))
    return 2*h*nu**3/c_light**2 * x/(1-x)
# interpolation
# these functions are written as classes so that they can be pickled
# inside other classes
import scipy.interpolate
class interp1d_log:
    def __init__(self, x, y):
        self.f_log = scipy.interpolate.interp1d(np.log(x),np.log(y), bounds_error=False, fill_value='extrapolate')
    def __call__(self, x):
        return np.exp(self.f_log(np.log(x)))