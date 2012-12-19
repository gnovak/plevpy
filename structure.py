# General notes
#
# 1) All operations are element-by-element unless otherwise indicated.
#    So a*b does _not_ do matrix multiplication.
# 2) There _is_ a broadcasting mechanism, whereby 
#    array([1,2,3]) * array([[4, 5, 6],   = array([[ 4, 10, 18], 
#                            [7, 8, 9]])           [ 7, 16, 27]])
#    Note that what was done here was not matrix multiplication, but
#    one application of element-by-element multiplication for each row
#    of the 2x3 array.
# 3) A few words on data types.  Brackets make a _list_ of things:
#    [1,2,3].  Parenthesis make a _tuple_ of things: (1,2,3).  Tuples
#    are immutable lists and are sometimes used in place of lists for
#    efficiency.  Passing either a list or a tuple to the function
#    numpy.array() makes an _array_ of those things.  An array is like
#    a list except: a) It can only hold integers, floating point
#    numbers, etc, instead of arbitrary python objects and b) It's
#    much faster for computation and c) It has the element-by-element
#    muliplication, addition, etc, operators defined for it.  Braces
#    make a _dictionary_: {'a': 1, 'b': 2}.  Dictionaries are like
#    lists, except the elements are named with strings rather than
#    being referred to by number.  So you can do things like dd['a']
#    or dd['the-named-thing'].
# 4) Python inherits C's strange behavior with respect to integer
#    arithemtic.  7/3 = 2, but 7.0/3, 7/3.0, and 7.0/3.0 all equal
#    2.333
# 5) The power operator is **, not ^
# 6) loadtxt(filename) to load tables from text files.

import hashlib
import os, os.path
import cPickle
import types

# Get access to scipy stuff
import scipy
import scipy.interpolate
import scipy.integrate
import scipy.optimize

### I usally do this: 
# from numpy import *
### which gets access to all numpy stuff in the present namespace so
### we can type array() instead of numpy.array().  But for clarity I'm
### going to do something else so that you can easily see what's
### coming from numpy.  However, numpy is still too long to type, so
### I'm going to rename it to np.  Unfortunately basic things like exp
### and log and pi are not in the default Python namespace.  That's
### why I usually do 'from numpy import *'

import numpy as np

# Get access to pylab, a matlab clone.  I usually just use the
# plotting stuff, but it some of the definitions of the functions you
# used are in here, too.

import pylab

##############################
# EOS
# Choices are 'polytrop', 'scvh', 'gsn'
# eos = 'polytrope'
# eos_parameters = dict(gamma=(5,3))

#eos = 'gsn'
eos = 'mesa'
eos_parameters = dict()

# Make sure the analytic "toy" EOS always exists.
gsn_natural_parameters = dict()

##############################
# Physical parameters
BAR = 1e6 # cgs
hbar = 1.05e-27 # cgs
mp = 1.67e-24  # cgs
me = 9.1e-28 # cgs
G = 6.673e-8    # cgs
kB = 1.38e-16  # cgs
Mjup = 1.899e30 # cgs
Rjup = 7.1492e9 # cgs
year = 3.15e7 # cgs
Lsun = 4e33   # cgs
AU = 1.5e13   # cgs
bar = 1e6   # cgs
cc = 3e10 # cgs

exts = ['pdf', 'png', 'eps']

##############################
## Interpolation
##############################
# Scipy has many functions to do 2d interpolation.  The all have
# different calling conventions.  Some of them aren't very good, some
# aren't robust, some don't respect all of the arguments you give
# them.  I want to feed three 2D arrays of x, y, and z values.  I want
# to optionally do the interpolation in linear or log space for each
# of the variables.  For the resulting function, I want the inputs and
# outputs to be in linear space regardless of how it does ithe
# interpolation.  Finally, I want it to apply pointwise to its
# arguments for scalars, 1d, and 2d arrays, rather than the wacky
# things that some of the scipy functions try to do.  This functions
# provide a uniform interface to the interpolation routines that I
# have found to be reasonably robust.


def interp_rbf(X,Y,Z,logx=True, logy=True, logz=True):
    """Unstructured interplation using radial basis functions"""
    # X,Y,Z should be physical values.  logx, etc, say whether you
    # want to do the interpolation in log space or not.
    X = np.log(X) if logx else X
    Y = np.log(Y) if logy else Y
    Z = np.log(Z) if logz else Z

    internal = scipy.interpolate.Rbf(X.ravel(), Y.ravel(), Z.ravel())

    def interpolator(xx, yy):
        xx = np.log(xx) if logx else xx
        yy = np.log(yy) if logy else yy            
        zz = internal(xx, yy)
        return np.exp(zz) if logz else zz

    return interpolator


def interp_griddata(X,Y,Z,logx=True, logy=True, logz=True):
    """Unstructured interplation using nearest neighbors for now."""
    X = np.log(X) if logx else X
    Y = np.log(Y) if logy else Y
    Z = np.log(Z) if logz else Z

    coords = grid_to_points(X,Y)
    values = grid_to_points(Z)

    def interpolator(xx,yy):
        xx,yy = np.asarray(xx), np.asarray(yy)
        if xx.shape != yy.shape: raise RuntimeError

        xx = np.log(xx) if logx else xx
        yy = np.log(yy) if logy else yy            

        desired_coords = grid_to_points(xx,yy)

        zz = scipy.interpolate.griddata(coords, values, desired_coords, method='nearest')
        zz = zz.reshape(xx.shape)            

        return np.exp(zz) if logz else zz
    return interpolator


def interp_spline(X,Y,Z,logx=True, logy=True, logz=True):
    """Unstructured interplation using splines."""
    X = np.log(X) if logx else X
    Y = np.log(Y) if logy else Y
    Z = np.log(Z) if logz else Z

    internal = scipy.interpolate.SmoothBivariateSpline(
        X.ravel(), Y.ravel(), Z.ravel())

    def interpolator(xx,yy):
        xx,yy = np.asarray(xx), np.asarray(yy)
        if xx.shape != yy.shape: raise RuntimeError

        # interpolating function operates  on the cross product of
        # its  arguments, and  I  want it  to  operate element  by
        # elemnt.   This complicates  things...  Handle  the three
        # common cases: scalars, arrays, and 2d arrays
        if len(xx.shape) == 0:
            xx = np.log(xx) if logx else xx
            yy = np.log(yy) if logy else yy            
            zz = internal(xx,yy)[0,0]
            result = np.exp(zz) if logz else zz
        elif len(xx.shape) == 1: 
            xx = np.log(xx) if logx else xx
            yy = np.log(yy) if logy else yy            
            zz = np.array([internal(the_x, the_y)[0,0]
                           for the_x, the_y in zip(xx,yy)])
            result = np.exp(zz) if logz else zz
        elif len(xx.shape) == 2: 
            # now we're getting tricky
            result = interpolator(xx.ravel(), yy.ravel()).reshape(xx.shape)
        else: 
            raise RuntimeError 
        return result
    return interpolator


def interp_1d(xx,yy,logx=True, logy=True, kind='linear'):

    xx = np.log(xx) if logx else xx
    yy = np.log(yy) if logy else yy

    internal = scipy.interpolate.interp1d(xx, yy, kind=kind)
    
    def interpolator(uu):
        uu = np.log(uu) if logx else uu
        ww = internal(uu)
        result = np.exp(ww) if logy else ww
        return result

    return interpolator
    

def interp_rect_spline(xx,yy,Z,logx=True, logy=True, logz=True, kx=5, ky=5):
    """interplation using splines on a rectangular grid."""
    # this one is different:  needs rectangular grid, so enforce this in the arguments.
    xx = np.log(xx) if logx else xx
    yy = np.log(yy) if logy else yy
    Z = np.log(Z) if logz else Z

    internal = scipy.interpolate.RectBivariateSpline(xx, yy, Z, kx=kx, ky=ky)

    def interpolator(uu,vv):
        uu,vv = np.asarray(uu), np.asarray(vv)
        if uu.shape != vv.shape: raise RuntimeError

        # interpolating function operates  on the cross product of
        # its  arguments, and  I  want it  to  operate element  by
        # elemnt.   This complicates  things...  Handle  the three
        # common cases: scalars, arrays, and 2d arrays
        if len(uu.shape) == 0:
            uu = np.log(uu) if logx else uu
            vv = np.log(vv) if logy else vv            
            zz = internal(uu,vv)[0,0]
            result = np.exp(zz) if logz else zz
        elif len(uu.shape) == 1: 
            uu = np.log(uu) if logx else uu
            vv = np.log(vv) if logy else vv            
            zz = np.array([internal(the_x, the_y)[0,0]
                           for the_x, the_y in zip(uu,vv)])
            result = np.exp(zz) if logz else zz
        elif len(uu.shape) == 2: 
            # now we're getting tricky
            result = interpolator(uu.ravel(), vv.ravel()).reshape(uu.shape)
        else: 
            raise RuntimeError 
        return result
    return interpolator

##############################
### Utilities for dealing with storing python objects in files.
##############################
def hashKey(*a, **kw):
    """Returns a unique key given a bunch of python objects as arguments"""
    return hashlib.sha256(cPickle.dumps((a, kw), protocol=-1)).digest()


def can(obj, file, protocol=2):
    """More convenient interactive syntax for pickle"""
    if type(file) is str: f=open(file,'wb')
    else: f=file

    cPickle.dump(obj, f, protocol=protocol)

    if type(file) is str: f.close()


def uncan(file):
    """More convenient interactive syntax for pickle"""
    # If filename, should this read until all exhausted?
    if type(file) is str: f=open(file, 'rb')
    else: f=file    

    obj = cPickle.load(f)

    if type(file) is str: f.close()

    return obj


def memoize(f, withFile=True, keyf=hashKey):
    """Return a 'fast' version of long-running function f If called
    with the same arguments, it just returns the previous return-value
    instead of recomputing.  It does this by converting all positional
    and keyword arguments to strings and indexing into a dictionary.
    A different method of generating keys can be used if you specify
    the keyf funciton.  If withFile is true, pickle the resulting
    dictionary to a file with name f.func_name.memo.  Be careful of
    using this feature with several instances of Python
    running... they will each overwrite the file.  It should not lead
    to corruption, though since the dict is only loaded from the file
    when memoize itself is called."""
    def g(*args, **kw):
        key = keyf(*args, **kw)
        if not key in results:
            results[key] = f(*args, **kw)            
            if withFile: can(results, fname)
        return results[key]
    
    fname = f.func_name + '.memo'
    if withFile and os.path.isfile(fname): results = uncan(fname)
    else: results = {}
    
    g.func_doc, g.func_name = f.func_doc, f.func_name
    return g

##############################
### Hypothetical main() function
##############################
def main_function():
    pressure = pressure_scvh('protons.txt')
    
    print pressure(1.1, 2.2)

    hse(pressure)

##############################
### Atmospheric boundary conditions
##############################

def read_atmospheric_bc_table(filename="../bound_py_isotropic_default.dat"):
    bc = np.loadtxt(filename)
    # If 8 columns, it's an "expanded" boundary condition where we
    # only need columns 1 2 and 4
    if bc.shape[1] == 8:
        bc = bc[:,[0,1,3]]
    xax = np.unique(bc[:,0])
    yax = np.unique(bc[:,1])
    nx = len(xax)
    ny = len(yax)
    assert bc.shape[0] == nx*ny
    Teff = bc[:,0].reshape(nx,ny)
    logg = bc[:,1].reshape(nx,ny)
    sigma = bc[:,2].reshape(nx,ny)    
    # make sure that the table is oriented the way it should be    
    assert all(Teff[0,:] == Teff[0,0])
    assert all(logg[:,0] == logg[0,0])    
    return Teff, logg, sigma


def atmospheric_bc(filename="../bound_py_isotropic_default.dat", 
                   s_axis=np.linspace(5,11,100),
                   logg_axis=np.linspace(1,4,100)):

    Teff, logg, sigma = read_atmospheric_bc_table(filename)
    Teff_grid = pylab.griddata(sigma.ravel(), logg.ravel(), Teff.ravel(), 
                               s_axis, logg_axis, interp='nn')    
    return interp_rect_spline(s_axis, logg_axis, Teff_grid)

##############################
### Equations of state
##############################
def pressure_scvh(other_arg, filename='table'):
    def pressure(rho, sigma):
        #interplote(rho, sigma, rho_table, sigma_table)
        pass        

    #rho_table = np.loadtxt(filename)
    #sigma_table = np.loadtxt(filename)
    return pressure

# def pressure_temperature_mesa(filename='../eos/MESA_EOS/rho_sig_grid'):
#     logrho = np.loadtxt(filename + '_logrho.txt')
#     sigma = np.loadtxt(filename + '_sigma.txt')
#     logpressure = np.loadtxt(filename + '_logP.txt')
#     logtemp = np.loadtxt(filename + '_logT.txt')
#     mu = np.loadtxt(filename + '_mu.txt')
#     rho = 10**logrho
#     pressure = 10**logpressure
#     temp = 10**logtemp
#     rho = rho[:,0]
#     sigma = sigma[0,:]
#     clean_array(temp)
#     clean_array(rho)
#     clean_array(mu)    
#     rho_func = interp_rect_spline(rho,sigma,pressure,
#                                        logx=True, logy=True, logz=True)
#     temp_func = interp_rect_spline(rho,sigma,temp,
#                                    logx=True, logy=True, logz=True)
#     mu_func = interp_rect_spline(rho,sigma,mu,
#                                    logx=True, logy=True, logz=True)
#     return rho_func, temp_func, mu_func
    
def density_temperature_mesa(filename='../eos/MESA_EOS/P_sig_grid'):
    def read_mesa_tables_slow(filename_base):
        return [np.loadtxt(filename_base + '_logP.txt'),
                np.loadtxt(filename_base + '_sigma.txt'),
                np.loadtxt(filename_base + '_logrho.txt'),
                np.loadtxt(filename_base + '_logT.txt'),
                np.loadtxt(filename_base + '_mu.txt'),
                np.loadtxt(filename_base + '_logE.txt')]

    def clean_array(xx):
        idx_good = ( np.logical_not(np.isnan(xx)) &  
                     np.logical_not(np.isinf(xx)) )
        idx_bad = np.isnan(xx) | np.isinf(xx)
        xx[idx_bad] = (xx[idx_good]).mean()

    read_mesa_tables = memoize(read_mesa_tables_slow)
    
    logpressure, sigma, logrho, logtemp, mu, loge = read_mesa_tables(filename)

    # get rid of log quantities
    rho,pressure,temp,en = [10**arr for arr in (logrho, logpressure, logtemp, loge)]
    # get rid of nans, etc for now
    for arr in rho, temp, mu, en:
        clean_array(arr)
    # make the axes 1d
    pressure = pressure[:,0]
    sigma = sigma[0,:]
    rho_func = interp_rect_spline(pressure,sigma,rho, logx=True, logz=True)
    temp_func = interp_rect_spline(pressure,sigma,temp, logx=True, logz=True)
    mu_func = interp_rect_spline(pressure,sigma,mu, logx=True, logz=True)
    en_func = interp_rect_spline(pressure,sigma,en, logx=True, logz=True)
    return rho_func, temp_func, mu_func, en_func


def density_mesa(filename='../eos/MESA_EOS/P_sig_grid'):
    return density_temperature_mesa(filename)[0]


def temperature_mesa(filename='../eos/MESA_EOS/P_sig_grid'):
    return density_temperature_mesa(filename)[1]


def mu_mesa(filename='../eos/MESA_EOS/P_sig_grid'):
    return density_temperature_mesa(filename)[2]


def energy_mesa(filename='../eos/MESA_EOS/P_sig_grid'):
    return density_temperature_mesa(filename)[3]


######################################
### Functions for reading the SCvH EoS
######################################
def read_the_stuff(the_eos_directory):
    result = {}
    for root, dirs, files in os.walk(the_eos_directory):
        for f in files:
            fullpath = os.path.join(root, f)
            name = canonize(fullpath)
            result[name] = np.loadtxt(fullpath)        
            print fullpath
    return result


def canonize(name):
    return '.'.join((name.split('/')[-1]).split('.')[:-1])


def getS(p_values, t_values, S_grid):
    return interp_rect_spline(p_values,t_values,S_grid, False, False, False)


def make_the_tables(Y = 0.25, dirpath = 'reformated_eos_tables/i/'):
    """Y = helium mass fraction"""
    denom_small = 1e-2
    
    result = read_the_stuff(dirpath)

    # GSN -- Interpolation routines unfortunately get confused if
    # there are nans anywhere in the array, so purge them here.
    def good(aa):
        bb = np.array(aa)
        fill = bb[bb==bb].mean()
        bb[bb!=bb] = fill
        return bb

    for kk in result.keys():
        result[kk] = good(result[kk])
    
    logP_array = result['h_tab_i_logP_array']
    logT_array = result['h_tab_i_logT_array']
    Scgs_array_H  = 10**result['h_tab_i_logScgs']
    Scgs_array_He = 10**result['he_tab_i_logScgs']
    # GSN -- changing to logrho in following two lines
    logrho_array_H  = result['h_tab_i_logrho']
    logrho_array_He = result['he_tab_i_logrho']

    # GSN -- adding choice about degree of spline interpolation to
    # calls to interp_rect_spline
    kx, ky = 1, 1

    # S and rho functions
    Scgs_of_logPlogT_H  = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           Scgs_array_H.transpose(),False, False, False, 
                                          kx=kx, ky=ky)
    Scgs_of_logPlogT_He = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           Scgs_array_He.transpose(),False, False, False, 
                                          kx=kx, ky=ky)
    # GSN -- looking at values, think this has to be log rho
    # GSN -- changing rho to logrho in following two lines
    logrho_of_logPlogT_H  = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           logrho_array_H.transpose(),False, False, False, 
                                               kx=kx, ky=ky)
    logrho_of_logPlogT_He = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           logrho_array_He.transpose(),False, False, False, 
                                               kx=kx, ky=ky)
    # Mixing ratio functions
    XH2_of_logPlogT = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           result['h_tab_i_Xwhole'].transpose(),False, False, False, 
                                         kx=kx, ky=ky)
    XHe_of_logPlogT = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           result['he_tab_i_Xwhole'].transpose(),False, False, False, 
                                         kx=kx, ky=ky)
    XH_of_logPlogT = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           result['h_tab_i_Xsplit'].transpose(),False, False, False, 
                                        kx=kx, ky=ky)
    XHeplus_of_logPlogT = interp_rect_spline(logP_array[0,:],logT_array[:,0],
                           result['he_tab_i_Xsplit'].transpose(),False, False, False, 
                                             kx=kx, ky=ky)

    def S_of_PT(P,T):
        Scgs_H  = Scgs_of_logPlogT_H(np.log10(P),np.log10(T))
        Scgs_He = Scgs_of_logPlogT_He(np.log10(P),np.log10(T))
        # for Smix, see Eqs 48-56 of SCvH 1996
        XH      = XH_of_logPlogT(np.log10(P),np.log10(T))
        XHe     = XHe_of_logPlogT(np.log10(P),np.log10(T))
        XH2     = XH2_of_logPlogT(np.log10(P),np.log10(T))
        XHeplus = XHeplus_of_logPlogT(np.log10(P),np.log10(T))
        XeH     = 0.5*(1 - XH2 - XH)
        XeHe    = (1./3.)*(2 - 2*XHe - XHeplus)
        mH  = mp
        mHe = 4*mp
        beta = (mH/mHe) * Y/(1-Y)
        gamma = (3./2.) * (1. + XH + 3*XH2)/(1. + 2*XHe + XHeplus)
        # GSN -- Also, sometimes results in division by zero
        delta_const = (2./3.) * beta*gamma
        delta_num =  (2. - 2.*XHe - XHeplus) 
        delta_denom = (1. - XH2 - XH)
        delta = delta_const*delta_num / delta_denom
        delta[abs(delta_denom) < denom_small] = 0

        # SCvH eq 53
        Smix_per_kB_prefactor = (1.-Y)/mp * 2./(1 + XH + 3*XH2) 
        Smix_per_kB_part1 = np.log(1+beta*gamma) - XeH*np.log(1+delta) + beta*gamma * np.log(1+1./(beta*gamma))
        Smix_per_kB_part2 = -beta*gamma*XeHe*np.log(1+1./delta)
        Smix_per_kB_part2[abs(delta_denom) < denom_small] = 0
        Smix_per_kB = Smix_per_kB_prefactor*(Smix_per_kB_part1 + Smix_per_kB_part2)
        
        # Algebra for part 2 of above formula showing that that part -> 0
        # (1/3)*num * log ( (const*num + denom) / (const*num))
        # (1/3)*num * log (const*num + denom)  - log(const*num))
        # (1/3)*num * log (const*num + denom)  - (1/3)*num * log(const*num))
        
        Smix_cgs = kB*Smix_per_kB
        # GSN -- might want to verify cgs/unitless conversion here.
        # GSN -- thought that entropy of mixing should be order unity, but this gives numbers ~ 10^-5
        Scgs = (1.-Y)*Scgs_H + Y*Scgs_He + Smix_cgs
        S = Scgs*mp/kB 
        return S

    def rho_of_PT(P,T):
        # GSN changing PT to logPlogT in function calls on next two lines
        # GSN changing rho to logrho on following two lines
        rho_H  = 10**logrho_of_logPlogT_H(np.log10(P),np.log10(T))
        rho_He = 10**logrho_of_logPlogT_He(np.log10(P),np.log10(T))
        # GSN -- sure I'm being dumb but this doesn't look right to
        # me.  Convince myself that it's right.
        one_over_rho = (1.-Y)/rho_H + Y/rho_He
        rho = 1./one_over_rho
        return rho

    return S_of_PT, rho_of_PT, logP_array, logT_array

#def rho_of_logT_logP(log)
   
#def getS(P_grid,T_grid,Scgs_grid): # function that interpolates S(P,T)
#    P_vec = np.unique(P_grid)
#    T_vec = np.unique(T_grid)
#    Scgs_fn = scipy.interpolate.interp2d(P_vec,T_vec,S_grid)
#    return Scgs_fn

#def S_of_TP(P_grid,T_grid,Scgs_grid,P,T): # function that interpolates S(P,T)
#    P_vec = np.unique(P_grid)
#    T_vec = np.unique(T_grid)
#    Scgs_fn = interp_rect_spline(P_vec,T_vec,tt,logx=True, logy=True, logz=True)
#    return Scgs_fn

#def getrho(P_grid,T_grid,rho_grid): # function that interpolates rho(P,T)
#    P_vec = np.unique(P_grid)
#    T_vec = np.unique(T_grid)
#    rho_fn = scipy.interpolate.interp2d(P_vec,T_vec,rho_grid)
#    return rho_fn


def pressure_polytrope(gamma=(5,3), rho_scale=1.0):
    """rho_scale in cgs"""
    # assuming entropy comes from protons, pressure is provided by
    # ideal gas law, assuming no ionization.    

    def pressure(rho, sigma):
        return factor * np.exp(2*sigma/3.0) * (rho*rho_scale**scale_exp)**gamma 

    # somewhat tortured definition to allow exact arithmetic
    if np.iterable(gamma):
        g_num, g_denom = gamma
    else:
        g_num, g_denom = gamma, 1.0
    gamma = (1.0*g_num)/(1.0*g_denom)

    scale_exp = (5*g_denom - 3*g_num) / (3.0*g_num)
    factor = 2*np.pi*hbar**2*np.exp(-5/3.0) / mp**(8/3.0)
    return pressure

##############################
### Equation of state based on numerically finding temperature that
### gives desired entropy for a given density, then using that
### temperature to find the pressure.
##############################
def dominates(xx,yy,ff=1.0, nexp=2.0):
    """Smoothly returns 1 if xx/yy is greater than ff, zero
    otherwise.  nexp controls the steepness of the switch, with
    larger values indicating steeper switches."""        
    return 1/(1.0+(xx/(ff*yy))**-nexp)


def pressure_gsn(**kw):
    """Return a function that computes pressure based on interpolation
on a grid that's rectangular in density and entropy.

rhos = rho values for interpolation grid
sigmas = sigma values for interpolation grid
_mp = proton mass
_me = electron mass
_hbar = planck constant
_cc = speed of light
ff_e, nexp_e, ff_p, nexp_p = controls where and how steeply the
   physics changes for protons and electrons.  See
   pressure_gsn_one_nn_kt()
Gamma_e, Gamma_p = number of degrees of freedom for electrons and
   protons as they become relativistic.  This could be a function, but
   for now is a constant.
method = scheme for finding desired value of entropy.  Can be 'bisect' or 'newton' 
"""
    rhos, sigmas, pp, tt = eos_gsn_tables(**kw)
    return interp_rect_spline(rhos,sigmas,pp,logx=True, logy=True, logz=True)


def temperature_gsn(**kw):
    """Return a function that computes temperature based on interpolation
on a grid that's rectangular in density and entropy.

rhos = rho values for interpolation grid
sigmas = sigma values for interpolation grid
_mp = proton mass
_me = electron mass
_hbar = planck constant
_cc = speed of light
ff_e, nexp_e, ff_p, nexp_p = controls where and how steeply the
   physics changes for protons and electrons.  See
   pressure_gsn_one_nn_kt()
Gamma_e, Gamma_p = number of degrees of freedom for electrons and
   protons as they become relativistic.  This could be a function, but
   for now is a constant.
method = scheme for finding desired value of entropy.  Can be 'bisect' or 'newton' 
"""
    rhos, sigmas, pp, tt = eos_gsn_tables(**kw)
    return interp_rect_spline(rhos,sigmas,tt,logx=True, logy=True, logz=True)


# FIXME -- this is not kt, should be just tt
def pressure_gsn_one_nn_kt(nn, tt, ffree, mm, ff, nexp, Gamma, _hbar, _cc):
    """Compute pressure for one species of particle.  

    nn = total number density
    tt = temperature
    ffree = fraction of given particles that are free.

    Gamma = number of degrees of freedom as fn of energy for rel. gas.

    Make transition:
    from ideal gas to degenerate NR gas when ef/kt > ff[0]
    from degenerate NR gas to degenerate R gas when ef/rm > ff[1]
    from degenerate R gas to degenerate R e+e- gas when kt/rm > ff[2]
    from R gas to degenerate R e+e- gas when ef/kt > ff[3]
    from ideal gas to R gas when kt/rm > ff[4]"""

    nn, tt = np.asarray(nn), np.asarray(tt)

    if not np.iterable(ff): ff = 5*[ff]
    if not np.iterable(nexp): nexp = 5*[nexp]

    compton = _hbar/(mm*_cc)    
    rm = mm*_cc**2
    kt = kB*tt

    # Find fermi energy, smoothing between asymptotic limits at
    # relativistic transition.
    kf = (np.pi/2.0)**(2.0/3.0) * nn**(1/3.0)

    ef = (dominates(kf, 1.0/compton)*(_hbar*kf*_cc) + 
          dominates(1.0/compton, kf)*(_hbar**2*kf**2/(2.0*mm)))
    
    pp = []
    # ideal gas pressure
    pp.append(dominates(rm, kt, 1/ff[4], nexp[4])*dominates(kt, ef, 1/ff[0], nexp[0]) * ffree*nn*kt)
    
    # non-relativisitic degenerate pressure
    # factor = 3**(2/3.0)*np.pi**(4/3.0)/40.0
    # FIXME -- factor of 8 in comparison to actual value, check notes.
    # degeneracy pressure relevant whether or not particles are free
    factor = 3**(2/3.0)*np.pi**(4/3.0)/5.0
    pp.append(dominates(rm, ef, 1/ff[1], nexp[1])*dominates(ef, kt, ff[0], nexp[0]) * factor * (_hbar**2/mm)*nn**(5/3.0))

    # relativistic degenerate pressure
    # FIXME -- factor 1.5 between original value and actual value, check notes.
    # factor = 3**(4/3.0)*np.pi**(2/3.0)/8.0
    # degeneracy pressure relevant whether or not particles are free
    factor = 3**(1/3.0)*np.pi**(2/3.0)/4.0
    pp.append(dominates(ef, rm, ff[1], nexp[1])*dominates(rm, kt, 1/ff[2], nexp[2]) * factor * (_hbar*_cc)*nn**(4/3.0))

    # relativistic degenerate pressure when thermal en is greater than
    # rest mass.  Excite additional degrees of freedom, and the
    # pressure scales as g^(-1/3) where g is the degeneracy.  So put
    # in that (extremely minor) correction.
    # FIXME -- factor 1.5 here, too, see above
    # factor = 3**(4/3.0)*np.pi**(2/3.0)/8.0 
    # Can't have binding energy greater than rest mass energy, so ffree not needed
    factor = 3**(1/3.0)*np.pi**(2/3.0)/4.0 
    factor = factor * (Gamma/2.0)**(-1/3.0)
    pp.append(dominates(ef, kt, ff[3], nexp[3])*dominates(kt, rm, ff[2], nexp[2]) * factor * (_hbar*_cc)*nn**(4/3.0))
    
    # Relativistic gas.  in this case, fermi energy is meaningless since kt >> ef
    # Can't have binding energy greater than rest mass energy, so ffree not needed
    factor = Gamma*np.pi**2/(45*_cc**3*_hbar**3)
    pp.append((dominates(kt, ef, 1/ff[3], nexp[3])*dominates(ef, rm) + 
               dominates(kt, rm, ff[4], nexp[4])*dominates(rm, ef)) * factor * kt**4)
              
    return sum(pp)


# FIXME -- this is not kt, should be just tt
def entropy_gsn_one_nn_kt(nn, tt, ffree, mm, ff, nexp, Gamma, _hbar, _cc):
    """Compute entropy for one species of particle.  

    Gamma = number of degrees of freedom as fn of energy for rel. gas.

    Make transition:
    from ideal gas to degenerate NR gas when ef/kt > ff[0]
    from degenerate NR gas to degenerate R gas when ef/rm > ff[1]
    from degenerate R gas to degenerate R e+e- gas when kt/rm > ff[2]
    from R gas to degenerate R e+e- gas when ef/kt > ff[3]
    from ideal gas to R gas when kt/rm > ff[4]"""

    nn, tt = np.asarray(nn), np.asarray(tt)

    if not np.iterable(ff): ff = 5*[ff]
    if not np.iterable(nexp): nexp = 5*[nexp]

    compton = _hbar/(mm*_cc)    
    rm = mm*_cc**2
    kt = kB*tt

    nq = ((mm*kt)/(2*np.pi*_hbar**2))**(3/2.0)  # NR expression

    # Find fermi energy, smoothing between asymptotic limits at
    # relativistic transition.
    kf = (np.pi/2.0)**(2.0/3.0) * nn**(1/3.0)

    ef = (dominates(kf, 1.0/compton)*(_hbar*kf*_cc) + 
          dominates(1.0/compton, kf)*(_hbar**2*kf**2/(2.0*mm)))
    
    sigma = []
    # ideal gas
    sigma.append(dominates(rm, kt, 1/ff[4], nexp[4])*dominates(kt, ef, 1/ff[0], nexp[0]) * ffree * (np.log(nq/nn) + 2.5))

    # degenerate non-relativisitic 
    factor = 2*np.pi**(5/3.0)/3**(2/3.0)
    sigma.append(dominates(rm, ef, 1/ff[1], nexp[1])*dominates(ef, kt, ff[0], nexp[0]) * factor*(nq/nn)**(2/3.0))

    # degenerate, relativistic
    factor = 8/(np.pi**2*_hbar*_cc)
    sigma.append(dominates(ef, rm, ff[1], nexp[1])*dominates(rm, kt, 1/ff[2], nexp[2]) * factor*kt/nn**(1/3.0))

    # relativistic degenerate pressure when thermal en is greater than
    # rest mass.    
    factor = 8/(np.pi**2*_hbar*_cc)
    sigma.append(dominates(ef, kt, ff[3], nexp[3])*dominates(kt, rm, ff[2], nexp[2]) * factor*Gamma*kt/nn**(1/3.0))
    
    # Relativistic gas.  in this case, fermi energy is meaningless since kt >> ef
    factor = 4*np.pi**2/(45*_hbar**3*_cc**3)
    sigma.append((dominates(kt, ef, 1/ff[3], nexp[3])*dominates(ef, rm) + 
                  dominates(kt, rm, ff[4], nexp[4])*dominates(rm, ef)) * factor * kt**3 / nn)
              
    return sum(sigma)


# FIXME -- this is not kt, should be just tt
def energy_gsn_one_nn_kt(nn, tt, ffree, mm, ff, nexp, Gamma, _hbar, _cc):
    """Compute the thermal energy density for one species of particle.  

    Gamma = number of degrees of freedom as fn of energy for rel. gas.

    Make transition:
    from ideal gas to degenerate NR gas when ef/kt > ff[0]
    from degenerate NR gas to degenerate R gas when ef/rm > ff[1]
    from degenerate R gas to degenerate R e+e- gas when kt/rm > ff[2]
    from R gas to degenerate R e+e- gas when ef/kt > ff[3]
    from ideal gas to R gas when kt/rm > ff[4]"""

    nn, tt = np.asarray(nn), np.asarray(tt)

    if not np.iterable(ff): ff = 5*[ff]
    if not np.iterable(nexp): nexp = 5*[nexp]

    compton = _hbar/(mm*_cc)    
    rm = mm*_cc**2
    kt = kB*tt

    # Find fermi energy, smoothing between asymptotic limits at
    # relativistic transition.
    kf = (np.pi/2.0)**(2.0/3.0) * nn**(1/3.0)

    ef = (dominates(kf, 1.0/compton)*(_hbar*kf*_cc) + 
          dominates(1.0/compton, kf)*(_hbar**2*kf**2/(2.0*mm)))
    
    pp = []
    # ideal gas 
    pp.append(dominates(rm, kt, 1/ff[4], nexp[4])*dominates(kt, ef, 1/ff[0], nexp[0]) * ffree*3* nn*kt / 2.0)
    
    # non-relativisitic degenerate 
    pp.append(dominates(rm, ef, 1/ff[1], nexp[1])*dominates(ef, kt, ff[0], nexp[0]) * 4.5*(kB*tt)**2*mm*nn**(5/3.0)/_hbar**2)

    # relativistic degenerate 
    pp.append(dominates(ef, rm, ff[1], nexp[1])*dominates(rm, kt, 1/ff[2], nexp[2]) * 4.5*(kB*tt)**2*nn*(2/3.0)/(_hbar*_cc))

    # relativistic degenerate when thermal en is greater than rest
    # mass.  Haven't thought carefully about this yet.  Think I just
    # pick up factor Gamma
    pp.append(dominates(ef, kt, ff[3], nexp[3])*dominates(kt, rm, ff[2], nexp[2]) * 4.5*Gamma*(kB*tt)**2*nn*(2/3.0)/(_hbar*_cc))
    
    # Relativistic gas.  in this case, fermi energy is meaningless since kt >> ef
    pp.append((dominates(kt, ef, 1/ff[3], nexp[3])*dominates(ef, rm) + 
               dominates(kt, rm, ff[4], nexp[4])*dominates(rm, ef)) * Gamma * kt**4)
              
    return sum(pp)


def pressure_gsn_nn_kt(tion=1e4, nexp_ion=2, 
                       _me=me, ff_e=1.0, nexp_e=4.0, Gamma_e=4.0, 
                       _mp=mp, ff_p=1.0, nexp_p=4.0, Gamma_p=4.0,                       
                       _hbar=hbar, _cc=cc):
    """Make a function that combines the proton and electron
    contribution to the pressure."""
    # serves to fix constants, electron mass, etc.
    def pressure_gsn_explicit(nn,tt):
        ffree = dominates(tt, tion, 1.0, nexp_ion)
        return (pressure_gsn_one_nn_kt(nn, tt, ffree, mm=_me, ff=ff_e, nexp=nexp_e, Gamma=Gamma_e, _hbar=_hbar, _cc=_cc)
                + pressure_gsn_one_nn_kt(nn, tt, 1.0, mm=_mp, ff=ff_p, nexp=nexp_p, Gamma=Gamma_p, _hbar=_hbar, _cc=_cc))
    return pressure_gsn_explicit


def entropy_gsn_nn_kt(tion=1e4, nexp_ion=2, 
                      _me=me, ff_e=1.0, nexp_e=4.0, Gamma_e=4.0, 
                      _mp=mp, ff_p=1.0, nexp_p=4.0, Gamma_p=4.0,
                      _hbar=hbar, _cc=cc):
    """Make a function that combines the proton and electron
    contribution to the entropy."""
    # serves to fix constants, electron mass, etc.
    def entropy_gsn_explicit(nn,tt):
        ffree = dominates(tt, tion, 1.0, nexp_ion)
        return (entropy_gsn_one_nn_kt(nn, tt, ffree, mm=_me, ff=ff_e, nexp=nexp_e, Gamma=Gamma_e, _hbar=_hbar, _cc=_cc) + 
                entropy_gsn_one_nn_kt(nn, tt, 1.0, mm=_mp, ff=ff_p, nexp=nexp_p, Gamma=Gamma_p, _hbar=_hbar, _cc=_cc))
    return entropy_gsn_explicit


def energy_gsn_nn_kt(tion=1e4, nexp_ion=2, 
                     _me=me, ff_e=1.0, nexp_e=4.0, Gamma_e=4.0, 
                     _mp=mp, ff_p=1.0, nexp_p=4.0, Gamma_p=4.0,
                     _hbar=hbar, _cc=cc):
    """Make a function that combines the proton and electron
    contribution to the energy."""
    # serves to fix constants, electron mass, etc.
    def energy_gsn_explicit(nn,tt):
        ffree = dominates(tt, tion, 1.0, nexp_ion)
        return (energy_gsn_one_nn_kt(nn, tt, ffree, mm=_me, ff=ff_e, nexp=nexp_e, Gamma=Gamma_e, _hbar=_hbar, _cc=_cc) + 
                energy_gsn_one_nn_kt(nn, tt, 1.0, mm=_mp, ff=ff_p, nexp=nexp_p, Gamma=Gamma_p, _hbar=_hbar, _cc=_cc))
    return energy_gsn_explicit
     

# def eos_gsn_tables_internal(rhos=np.logspace(-5,20,50), 
#                                     sigmas=np.logspace(-6,20,50), 
#                                     _mp=mp, _me=me, _cc=cc, _hbar=hbar,
#                                     method='bisect', 
#                                     **kw):


def eos_gsn_tables_internal(rhos=np.logspace(-12,4,50), 
                            sigmas=np.logspace(0,2,50), 
                            _mp=mp, _me=me, _cc=cc, _hbar=hbar,
                            method='bisect', 
                            **kw):
    """Equation of state taking into account ideal gas pressure,
    degeneracy pressure, relativistic degeneracy pressure, and
    relativistic gas pressure.  You can move around the boundaries
    where the physics switches between one expression and another via
    ff_e and ff_p.  The idea is to say things like 'how would it
    change the radius/mass relation if I turn off electron degneracy
    pressure?'  This set of functions uses an entirely separate set of
    physics constants _me, _mp, _cc, and _hbar, and tries to pass them
    around appropriately so that you can change the physics constants
    and draw plots of things like what happens when you change the
    speed of light (to see what relativistic effects are
    contributing.

    This draws a regular grid in density and entropy and then uses
    newton's method or bisection to find the temperature that gives
    the desired entropy.  This make interpolation on the resulting
    grid trivial.

    This function just makes the interpolation tables to facilitate
    storing them since it takes a little time to generate a fine grid.

    Extra args passed to entropy_gsn_nn_kt(), pressure_gsn_nn_kt()"""

    def the_ff(logt):
        """The function of which we're finding a root"""
        result = np.log(the_ss(rho/_mp, np.exp(logt))) - np.log(desired_entropy)
        return result

    def init(nn, sigma):
        """Make an initial guess for the temperature"""
        t_degen = (2*np.pi*_hbar**2/(_mp*kB))*nn**(2/3.0)*sigma
        t_ideal_gas = (2*np.pi*_hbar**2/(_mp*kB))*np.exp(2*sigma/3.0 - 5/3.0)
        t_rel = _hbar*_cc*nn**(1/3.0)*sigma**(1/3.0)/kB
        # entropy is in the protons, energy is in the degenerate electrons
        if sigma <= 1.5*np.log(_mp/_me):  
            return np.log(t_degen)
        elif kB*t_ideal_gas > _me*_cc**2:
            return np.log(t_rel)
        else:
            return np.log(t_ideal_gas)

    def find_root_newton(logt0, rho):
        """Find the temperature using newton's method."""
        try: 
            logt = scipy.optimize.newton(the_ff, logt0)
            tt = np.exp(logt)
            pp = the_pp(rho/_mp, tt)            
        except RuntimeError: 
            tt = np.nan
            pp = np.nan
        return pp, tt

    def find_root_bisect(logt0, rho):        
        """Find the temperature using bisection"""
        delta = 1.0
        # initial guess at range
        logtl, logth = logt0, logt0+delta

        # expand range until it includes the root
        nplus, nminus = 0,0
        while(the_ff(logth) < 0):
            logth += delta
            nplus += 1
        while(the_ff(logtl) > 0):
            logtl -= delta
            nminus += 1

        #print nplus, nminus
        # find the root
        logt = scipy.optimize.bisect(the_ff, logtl, logth, xtol=1e-5, rtol=1e-5)
        tt = np.exp(logt)
        pp = the_pp(rho/_mp, tt)            
        return pp, tt
    
    def clean_nans(aa):         
        """Newton's method sometimes fails and I've put nans into the
        array in that case.  Try to interpolate from neighboring
        values to remove them."""
        # nan is the only thing that's not equal to itself
        # This is a pain... edges/corners require special handling, can have bad cells next to each other
        ii,jj = np.nonzero(aa!=aa)
        nn = len(ii)
        if nn != 0: 
            print "Warning: structure.py: Fixing", nn, "failed points"
            if ((ii==0) | (jj==0) | (ii==aa.shape[0]-1) | (jj==aa.shape[1]-1)).any():
                raise RuntimeError, "structure.py: Don't know how to fix failed edge points"
            # arithmetic mean
            # aa[ii,jj] = 0.25*(aa[ii+1,jj] + aa[ii-1,jj] + aa[ii,jj+1] + aa[ii,jj-1])
            # geometric mean
            aa[ii,jj] = (aa[ii+1,jj]*aa[ii-1,jj]*aa[ii,jj+1]*aa[ii,jj-1])**0.25
            if (aa!=aa).any():
                raise RuntimeError, "structure.py: Don't know how to fix adjacent failed points"
        return aa
    
    if method.lower()=='newton': find_root = find_root_newton
    elif method.lower()=='bisect': find_root = find_root_bisect
    else: raise RuntimeError, "Unknown root finding method!"

    # Do a copy here to prevent a mutable copy of a default arg from
    # escaping into the wild.
    rhos, sigmas = np.array(rhos), np.array(sigmas)

    the_pp = pressure_gsn_nn_kt(_mp=_mp, _me=_me, _cc=_cc, _hbar=_hbar, **kw)    
    the_ss = entropy_gsn_nn_kt(_mp=_mp, _me=_me, _cc=_cc, _hbar=_hbar, **kw)

    # construct tables
    pp, tt = [], []
    for rho in rhos:
        prow, trow = [], []
        for desired_entropy in sigmas:
            logt0 = init(rho/_mp, desired_entropy)
            pval, tval = find_root(logt0, rho)            
            prow.append(pval)
            trow.append(tval)
        pp.append(prow)
        tt.append(trow)

    pp, tt = np.array(pp), np.array(tt)
    pp,tt = clean_nans(pp), clean_nans(tt)

    return rhos, sigmas, pp, tt

eos_gsn_tables = memoize(eos_gsn_tables_internal)


def eos_gsn_pt(**kw):
    """Return a function that computes density and a function that
computes entropy, both as a function of pressure and temperature based
on interpolation on a grid that's rectangular in pressure and
temperature.

ps = pressure values for interpolation grid
ts = temperaturevalues for interpolation grid
_mp = proton mass
_me = electron mass
_hbar = planck constant
_cc = speed of light
ff_e, nexp_e, ff_p, nexp_p = controls where and how steeply the
   physics changes for protons and electrons.  See
   pressure_gsn_one_nn_kt()
Gamma_e, Gamma_p = number of degrees of freedom for electrons and
   protons as they become relativistic.  This could be a function, but
   for now is a constant.
method = scheme for finding desired value of density.  Can be 'bisect' or 'newton' 
"""

    pp, tt, nn, sigmas  = eos_gsn_pt_tables(**kw)
    nn_pt = interp_rect_spline(pp,tt,nn,logx=True, logy=True, logz=True)
    sig_pt = interp_rect_spline(pp,tt,sigmas,logx=True, logy=True, logz=False)
    return nn_pt, sig_pt


# Excluding where electrons become relativistic (b/c pressure doesn't
# depend on density anymore) and setting min and max pressures by
# pressure when electrons go relativistic (10^23) and at cosmic
# density at the min temperature (10^-21) gives the limits: 
#   ps=np.logspace(-21,23,8), 
#   ts=np.logspace(0,9,6),       
# My rather wide limits
#   ps=np.logspace(-5,15,30), 
#   ts=np.logspace(0,6,30),                                
# SCvH Limits:
#   ps=np.logspace(4,19,10), 
#   ts=np.logspace(2,7,10),                                                         
def eos_gsn_pt_tables_internal(ps=np.logspace(4,19,60), 
                               ts=np.logspace(2,7,60), 
                               _mp=mp, _me=me, _cc=cc, _hbar=hbar,
                               method='bisect', 
                               **kw):
    """As eos_gsn_tables_internal, except taking pressure and
    temperature to be the independent variables rather than density
    and entropy.

    Equation of state taking into account ideal gas pressure,
    degeneracy pressure, relativistic degeneracy pressure, and
    relativistic gas pressure.  You can move around the boundaries
    where the physics switches between one expression and another via
    ff_e and ff_p.  The idea is to say things like 'how would it
    change the radius/mass relation if I turn off electron degneracy
    pressure?'  This set of functions uses an entirely separate set of
    physics constants _me, _mp, _cc, and _hbar, and tries to pass them
    around appropriately so that you can change the physics constants
    and draw plots of things like what happens when you change the
    speed of light (to see what relativistic effects are
    contributing.

    This draws a regular grid in pressure and temperature and then uses
    newton's method or bisection to find the temperature that gives
    the desired density.  This makes interpolation on the resulting
    grid trivial.

    This function just makes the interpolation tables to facilitate
    storing them since it takes a little time to generate a fine grid.

    Extra args passed to entropy_gsn_nn_kt(), pressure_gsn_nn_kt()"""

    def the_ff(log_nn):
        """The function of which we're finding a root"""
        result = np.log(the_pp(np.exp(log_nn), tt)) - np.log(desired_pressure)
        return result

    def init(desired_pressure, tt):
        """Make an initial guess for the density"""        
        # electron rest energy
        e_rest = _me*_cc**2

        nn_ideal_gas = desired_pressure / (kB*tt)
        nn_degen_nonrel = (30*_me*desired_pressure / 
                           (3**0.66*np.pi**1.33*_hbar**2))**(3/5.0)
        nn_degen_rel = (4*desired_pressure / 
                        (3**1.33*np.pi**0.66*_hbar*_cc))**0.75
        # If the temperature gives relativistic electrons, the density
        # can have any value
        if (kB*tt) > e_rest: 
            print "Warning, pressure is not a function of density in relativistic regime."
            return np.nan
        # Compute fermi energy under different assumptions
        ef_ideal_gas = _hbar**2*nn_ideal_gas**(2/3.0)/(2*_me)
        ef_nonrel = _hbar**2*nn_degen_nonrel**(2/3.0)/(2*_me)
        ef_rel = _hbar*_cc*nn_degen_rel**(1/3.0)

        ideal = ef_ideal_gas < e_rest and ef_ideal_gas < kB*tt
        degen_nonrel = ef_nonrel < e_rest and ef_nonrel > kB*tt
        degen_rel = ef_rel > e_rest and ef_rel > kB*tt
        
        if ((ideal and degen_nonrel) or (ideal and degen_rel) or (degen_nonrel and degen_rel)):
            print "Warning, more than one set of assumptions satisfied"
        elif not (ideal or degen_nonrel or degen_rel):
            print "Warning, no set of assumptions satisfied"

        if degen_rel:
            return np.log(nn_degen_rel)
        elif degen_nonrel: 
            return np.log(nn_degen_nonrel)
        else: 
            return np.log(nn_ideal_gas)
        
    def find_root_newton(log_nn0, tt):
        """Find the temperature using newton's method."""
        try: 
            log_nn = scipy.optimize.newton(the_ff, log_nn0)
            nn = np.exp(log_nn)
            sig = the_ss(nn, tt)
        except RuntimeError: 
            nn = np.nan
            sig = np.nan
        return nn, sig

    def find_root_bisect(log_nn0, tt):
        """Find the temperature using bisection"""
        delta = 1.0
        # initial guess at range
        log_nnl, log_nnh = log_nn0, (log_nn0+delta)

        # expand range until it includes the root
        nplus, nminus = 0,0
        while(the_ff(log_nnh) < 0):
            log_nnh += delta
            nplus += 1
        while(the_ff(log_nnl) > 0):
            log_nnl -= delta
            nminus += 1

        # find the root
        log_nn = scipy.optimize.bisect(the_ff, log_nnl, log_nnh, 
                                        xtol=1e-5, rtol=1e-5)
        nn = np.exp(log_nn)
        sig = the_ss(nn, tt)
        return nn, sig
    
    def clean_nans(aa):         
        """Newton's method sometimes fails and I've put nans into the
        array in that case.  Try to interpolate from neighboring
        values to remove them."""
        # nan is the only thing that's not equal to itself
        # This is a pain... edges/corners require special handling, can have bad cells next to each other
        ii,jj = np.nonzero(aa!=aa)
        nn = len(ii)
        if nn != 0: 
            print "Warning: structure.py: Fixing", nn, "failed points"
            if ((ii==0) | (jj==0) | (ii==aa.shape[0]-1) | (jj==aa.shape[1]-1)).any():
                raise RuntimeError, "structure.py: Don't know how to fix failed edge points"
            # arithmetic mean
            # aa[ii,jj] = 0.25*(aa[ii+1,jj] + aa[ii-1,jj] + aa[ii,jj+1] + aa[ii,jj-1])
            # geometric mean
            aa[ii,jj] = (aa[ii+1,jj]*aa[ii-1,jj]*aa[ii,jj+1]*aa[ii,jj-1])**0.25
            if (aa!=aa).any():
                raise RuntimeError, "structure.py: Don't know how to fix adjacent failed points"
        return aa
    
    if method.lower()=='newton': find_root = find_root_newton
    elif method.lower()=='bisect': find_root = find_root_bisect
    else: raise RuntimeError, "Unknown root finding method!"

    # Do a copy here to prevent a mutable copy of a default arg from
    # escaping into the wild.
    ps, ts = np.array(ps), np.array(ts)

    the_pp = pressure_gsn_nn_kt(_mp=_mp, _me=_me, _cc=_cc, _hbar=_hbar, **kw)    
    the_ss = entropy_gsn_nn_kt(_mp=_mp, _me=_me, _cc=_cc, _hbar=_hbar, **kw)

    # natural input -- density, temperature
    # natural output -- pressure, entropy
    # desired input -- pressure, temperature
    # desired output -- density entropy
    # therefore iterate on _density_ to match _pressure_

    # construct tables
    nns, sigmas = [], []
    for desired_pressure in ps:
        nn_row, sigma_row = [], []
        for tt in ts:
            log_nn0 = init(desired_pressure, tt)
            nn_val, sigma_val = find_root(log_nn0, tt)
            
            nn_row.append(nn_val)
            sigma_row.append(sigma_val)
        nns.append(nn_row)
        sigmas.append(sigma_row)

    nns, sigmas = np.array(nns), np.array(sigmas)
    nns, sigmas = clean_nans(nns), clean_nans(sigmas)

    return ps, ts, nns, sigmas

# Create my EOS functions of "natural" variables, which must always exist
gsn_natural_pressure = pressure_gsn_nn_kt(**gsn_natural_parameters)
gsn_natural_entropy = entropy_gsn_nn_kt(**gsn_natural_parameters)
gsn_natural_energy = energy_gsn_nn_kt(**gsn_natural_parameters)

eos_gsn_pt_tables = memoize(eos_gsn_pt_tables_internal)
#eos_gsn_pt_tables = eos_gsn_pt_tables_internal

##############################
### Find and plot regions where relevant physics for protons and
### electrons changes for pressure_gsn EOS
##############################


def find_region_boundaries(_me, _mp, _hbar, _cc, **kw):
    """Find values of density, entropy, etc, that delimit regions
    where the physics changes for protons and electrons."""
    # everything meets when t=m c2 and rho = lambda^-3, so center things there
    import pdb
    compton_e = _hbar/(_me*_cc)
    compton_p = _hbar/(_mp*_cc)
    
    rhov = np.logspace(-3, # np.log10(_mp*compton_e**-3) - 3,
        
                       np.log10(_mp*compton_p**-3) + 3,30)
    ttv = np.logspace(np.log10(_me*_cc**2/kB) - 3, 
                      np.log10(_mp*_cc**2/kB) + 3, 30)

    # get interpolation based calculators
    tt_rho_sig = temperature_gsn(_me=_me, _mp=_mp, _hbar=_hbar, _cc=_cc, **kw)
    pp_rho_sig = pressure_gsn(_me=_me, _mp=_mp, _hbar=_hbar, _cc=_cc, **kw)

    # get rid of args that cause problems for non-interpolation based calculators
    popKeys(kw, 'rhos', 'sigmas', 'method')

    # get explicit calculators
    pp_nn_kt = pressure_gsn_nn_kt(_me=_me, _mp=_mp, _hbar=_hbar, _cc=_cc, **kw)
    ss_nn_kt = entropy_gsn_nn_kt(_me=_me, _mp=_mp, _hbar=_hbar, _cc=_cc, **kw)

    result = dict()
    # Relativity for protons
    tt = _mp*_cc**2/kB
    result['ss_rel_p'] = ss_nn_kt(rhov/_mp, tt)
    result['pp_rel_p'] = pp_nn_kt(rhov/_mp, tt)
    result['rr_rel_p'] = rhov
    result['tt_rel_p'] = tt + 0*rhov
    
    # Relativity for electrons
    tt = _me*_cc**2/kB
    result['ss_rel_e'] = ss_nn_kt(rhov/_mp, tt)
    result['pp_rel_e'] = pp_nn_kt(rhov/_mp, tt)
    result['rr_rel_e'] = rhov
    result['tt_rel_e'] = tt + 0*rhov

    # degeneracy for protons
    ss = 1.0 + 0*rhov
    result['tt_deg_p'] = tt_rho_sig(rhov, ss)
    result['pp_deg_p'] = pp_rho_sig(rhov, ss)
    result['rr_deg_p'] = rhov
    result['ss_deg_p'] = ss + 0*rhov

    # degeneracy for electrons
    ss = 2+1.5*np.log(_mp/_me) + 0*rhov
    result['tt_deg_e'] = tt_rho_sig(rhov, ss)
    result['pp_deg_e'] = pp_rho_sig(rhov, ss)
    result['rr_deg_e'] = rhov
    result['ss_deg_e'] = ss + 0*rhov

    # degeneracy + relativity for protons
    rho = _mp*compton_p**-3
    result['ss_deg_rel_p'] = ss_nn_kt(rho/_mp, ttv)
    result['pp_deg_rel_p'] = pp_nn_kt(rho/_mp, ttv)
    result['rr_deg_rel_p'] = rho + 0*ttv
    result['tt_deg_rel_p'] = ttv

    # degeneracy + relativity for electrons
    rho = _mp*compton_e**-3
    result['ss_deg_rel_e'] = ss_nn_kt(rho/_mp, ttv)
    result['pp_deg_rel_e'] = pp_nn_kt(rho/_mp, ttv)
    result['rr_deg_rel_e'] = rho + 0*ttv
    result['tt_deg_rel_e'] = ttv
    
    return result


def plot_region_boundaries(dd):
    """Make a big plot of what physics is relavant given values of the
    various thermodynamic quantities."""
    pylab.clf();

    pylab.subplot(3,4,1)
    pylab.loglog(dd['rr_rel_p'], dd['tt_rel_p'], 'b-')
    pylab.loglog(dd['rr_deg_p'], dd['tt_deg_p'], 'b--')
    pylab.loglog(dd['rr_deg_rel_p'], dd['tt_deg_rel_p'], 'b:')
    pylab.loglog(dd['rr_rel_e'], dd['tt_rel_e'], 'r-')
    pylab.loglog(dd['rr_deg_e'], dd['tt_deg_e'], 'r--')
    pylab.loglog(dd['rr_deg_rel_e'], dd['tt_deg_rel_e'], 'r:')
    pylab.ylabel('T')

    pylab.subplot(3,4,5)
    pylab.loglog(dd['rr_rel_p'], dd['ss_rel_p'], 'b-')
    pylab.loglog(dd['rr_deg_p'], dd['ss_deg_p'], 'b--')
    pylab.loglog(dd['rr_deg_rel_p'], dd['ss_deg_rel_p'], 'b:')
    pylab.loglog(dd['rr_rel_e'], dd['ss_rel_e'], 'r-')
    pylab.loglog(dd['rr_deg_e'], dd['ss_deg_e'], 'r--')
    pylab.loglog(dd['rr_deg_rel_e'], dd['ss_deg_rel_e'], 'r:')
    pylab.ylabel('sigma')

    pylab.subplot(3,4,9)
    pylab.loglog(dd['rr_rel_p'], dd['pp_rel_p'], 'b-')
    pylab.loglog(dd['rr_deg_p'], dd['pp_deg_p'], 'b--')
    pylab.loglog(dd['rr_deg_rel_p'], dd['pp_deg_rel_p'], 'b:')
    pylab.loglog(dd['rr_rel_e'], dd['pp_rel_e'], 'r-')
    pylab.loglog(dd['rr_deg_e'], dd['pp_deg_e'], 'r--')
    pylab.loglog(dd['rr_deg_rel_e'], dd['pp_deg_rel_e'], 'r:')
    pylab.xlabel('rho')
    pylab.ylabel('P')

    pylab.subplot(3,4,2)
    pylab.loglog(dd['tt_rel_p'], dd['rr_rel_p'], 'b-')
    pylab.loglog(dd['tt_deg_p'], dd['rr_deg_p'], 'b--')
    pylab.loglog(dd['tt_deg_rel_p'], dd['rr_deg_rel_p'], 'b:')
    pylab.loglog(dd['tt_rel_e'], dd['rr_rel_e'], 'r-')
    pylab.loglog(dd['tt_deg_e'], dd['rr_deg_e'], 'r--')
    pylab.loglog(dd['tt_deg_rel_e'], dd['rr_deg_rel_e'], 'r:')
    pylab.ylabel('rho')

    pylab.subplot(3,4,6)
    pylab.loglog(dd['tt_rel_p'], dd['ss_rel_p'], 'b-')
    pylab.loglog(dd['tt_deg_p'], dd['ss_deg_p'], 'b--')
    pylab.loglog(dd['tt_deg_rel_p'], dd['ss_deg_rel_p'], 'b:')
    pylab.loglog(dd['tt_rel_e'], dd['ss_rel_e'], 'r-')
    pylab.loglog(dd['tt_deg_e'], dd['ss_deg_e'], 'r--')
    pylab.loglog(dd['tt_deg_rel_e'], dd['ss_deg_rel_e'], 'r:')
    pylab.ylabel('sigma')

    pylab.subplot(3,4,10)
    pylab.loglog(dd['tt_rel_p'], dd['pp_rel_p'], 'b-')
    pylab.loglog(dd['tt_deg_p'], dd['pp_deg_p'], 'b--')
    pylab.loglog(dd['tt_deg_rel_p'], dd['pp_deg_rel_p'], 'b:')
    pylab.loglog(dd['tt_rel_e'], dd['pp_rel_e'], 'r-')
    pylab.loglog(dd['tt_deg_e'], dd['pp_deg_e'], 'r--')
    pylab.loglog(dd['tt_deg_rel_e'], dd['pp_deg_rel_e'], 'r:')
    pylab.xlabel('T')
    pylab.ylabel('P')

    pylab.subplot(3,4,3)
    pylab.loglog(dd['ss_rel_p'], dd['rr_rel_p'], 'b-')
    pylab.loglog(dd['ss_deg_p'], dd['rr_deg_p'], 'b--')
    pylab.loglog(dd['ss_deg_rel_p'], dd['rr_deg_rel_p'], 'b:')
    pylab.loglog(dd['ss_rel_e'], dd['rr_rel_e'], 'r-')
    pylab.loglog(dd['ss_deg_e'], dd['rr_deg_e'], 'r--')
    pylab.loglog(dd['ss_deg_rel_e'], dd['rr_deg_rel_e'], 'r:')
    pylab.ylabel('rho')

    pylab.subplot(3,4,7)
    pylab.loglog(dd['ss_rel_p'], dd['tt_rel_p'], 'b-')
    pylab.loglog(dd['ss_deg_p'], dd['tt_deg_p'], 'b--')
    pylab.loglog(dd['ss_deg_rel_p'], dd['tt_deg_rel_p'], 'b:')
    pylab.loglog(dd['ss_rel_e'], dd['tt_rel_e'], 'r-')
    pylab.loglog(dd['ss_deg_e'], dd['tt_deg_e'], 'r--')
    pylab.loglog(dd['ss_deg_rel_e'], dd['tt_deg_rel_e'], 'r:')
    pylab.ylabel('T')

    pylab.subplot(3,4,11)
    pylab.loglog(dd['ss_rel_p'], dd['pp_rel_p'], 'b-')
    pylab.loglog(dd['ss_deg_p'], dd['pp_deg_p'], 'b--')
    pylab.loglog(dd['ss_deg_rel_p'], dd['pp_deg_rel_p'], 'b:')
    pylab.loglog(dd['ss_rel_e'], dd['pp_rel_e'], 'r-')
    pylab.loglog(dd['ss_deg_e'], dd['pp_deg_e'], 'r--')
    pylab.loglog(dd['ss_deg_rel_e'], dd['pp_deg_rel_e'], 'r:')
    pylab.xlabel('sigma')
    pylab.ylabel('P')

    pylab.subplot(3,4,4)
    pylab.loglog(dd['pp_rel_p'], dd['rr_rel_p'], 'b-')
    pylab.loglog(dd['pp_deg_p'], dd['rr_deg_p'], 'b--')
    pylab.loglog(dd['pp_deg_rel_p'], dd['rr_deg_rel_p'], 'b:')
    pylab.loglog(dd['pp_rel_e'], dd['rr_rel_e'], 'r-')
    pylab.loglog(dd['pp_deg_e'], dd['rr_deg_e'], 'r--')
    pylab.loglog(dd['pp_deg_rel_e'], dd['rr_deg_rel_e'], 'r:')
    pylab.ylabel('rho')

    pylab.subplot(3,4,8)
    pylab.loglog(dd['pp_rel_p'], dd['tt_rel_p'], 'b-')
    pylab.loglog(dd['pp_deg_p'], dd['tt_deg_p'], 'b--')
    pylab.loglog(dd['pp_deg_rel_p'], dd['tt_deg_rel_p'], 'b:')
    pylab.loglog(dd['pp_rel_e'], dd['tt_rel_e'], 'r-')
    pylab.loglog(dd['pp_deg_e'], dd['tt_deg_e'], 'r--')
    pylab.loglog(dd['pp_deg_rel_e'], dd['tt_deg_rel_e'], 'r:')
    pylab.ylabel('T')

    pylab.subplot(3,4,12)
    pylab.loglog(dd['pp_rel_p'], dd['ss_rel_p'], 'b-')
    pylab.loglog(dd['pp_deg_p'], dd['ss_deg_p'], 'b--')
    pylab.loglog(dd['pp_deg_rel_p'], dd['ss_deg_rel_p'], 'b:')
    pylab.loglog(dd['pp_rel_e'], dd['ss_rel_e'], 'r-')
    pylab.loglog(dd['pp_deg_e'], dd['ss_deg_e'], 'r--')
    pylab.loglog(dd['pp_deg_rel_e'], dd['ss_deg_rel_e'], 'r:')
    pylab.xlabel('P')
    pylab.ylabel('sigma')

##############################
### Equation of state based on generating a table that's rectangular
### in density and temperature, then trying to interpolate based on
### the non-rectangular entropy grid.  It turns out to be hard to get
### this to work well.
##############################


def pressure_gsn_irregular_interpolation(rhos=None, temps=None,
                 _me=me, ff_e=1.0, nexp_e=4.0, Gamma_e=4.0, 
                 _mp=mp, ff_p=1.0, nexp_p=4.0, Gamma_p=4.0,
                 _hbar=hbar, _cc=cc, 
                 interp=None):
    """Equation of state taking into account ideal gas pressure,
    degeneracy pressure, relativistic degeneracy pressure, and
    relativistic gas pressure.  You can move around the boundaries
    where the physics switches between one expression and another via
    ff_e and ff_p.  The idea is to say things like 'how would it
    change the radius/mass relation if I turn off electron degneracy
    pressure?'  This set of functions uses an entirely separate set of
    physics constants _me, _mp, _cc, and _hbar, and tries to pass them
    around appropriately so that you can change the physics constants
    and draw plots of things like what happens when you change the
    speed of light (to see what relativistic effects are
    contributing.

    This draws a regular grid in density and temperature and then
    tries to use those tables to interpolate pressure as a function of
    density and entropy.  This doens't work very well."""
    # Function should take (density, entropy) and produce (pressure)
    # interpolation 

    # creation
    # interp_rbf is ~nn^5 to create, giving 70 or so as practical upper limit
    # interp_spline is ~nn^2 to create, giving ~800 or so as the practical upper limit
    # interp_griddata is ~nn^2 to create, giving ~800 or so as practical upper limit
    
    # evaluation
    # interp_rbf with 50x50 table is linear to evaluate w/ upper limit of ~30,000
    # interp_spline with 500x500 table is linear to eval with upper limit of 1e6
    # interp_griddata with 500x500 table is linear to eval with upper limit of 1e6

    # crazy table
    #if rhos is None: rhos = np.logspace(-30,30,100)
    #if temps is None: temps = np.logspace(0,20,100)

    # more reasonable table
    if rhos is None: rhos = np.logspace(-5,5,50)
    if temps is None: temps = np.logspace(2,9,50)

    if interp is None: interp = interp_spline

    R, T = make_grid(rhos, temps)
        
    pp = (pressure_gsn_one_nn_kt(R/_mp, T, ffree, mm=_me, ff=ff_e, nexp=nexp_e, Gamma=Gamma_e, _hbar=_hbar, _cc=_cc) +
          pressure_gsn_one_nn_kt(R/_mp, T, 1.0, mm=_mp, ff=ff_p, nexp=nexp_p, Gamma=Gamma_p, _hbar=_hbar, _cc=_cc))
    sigma = (entropy_gsn_one_nn_kt(R/_mp, T, ffree, mm=_me, ff=ff_e, nexp=nexp_e, Gamma=Gamma_e, _hbar=_hbar, _cc=_cc) +
             entropy_gsn_one_nn_kt(R/_mp, T, 1.0, mm=_mp, ff=ff_p, nexp=nexp_p, Gamma=Gamma_p, _hbar=_hbar, _cc=_cc))
    pressure = interp(R, sigma, pp, logx=True, logy=True, logz=True)

    return pressure

##############################
## Analytic EOS
##
## Try to write set of functions that return thermodynamic variables
## given any input set.  The "natural" variables are rho and temp
## since it's easy to write an explicit expression for entropy and
## pressure.  But I may want other input variables.  
## 
## Do this by solving for all combinations of thermo variables in the
## various limits, producing a list of guesses that are valid in each
## limit, and then trying to find the root of the full expressions of
## p and s.
## 
## For the moment this assumes standard electron, proton mass, etc.  
##############################

# def entropy_gsn_nn_tt()
# def pressure_gsn_nn_tt()

# def aeos_guess_pp_sigma(): not necessary


def aeos_guess_nn_sigma(pp, tt):
    pi = np.pi
    nqp = (mp*kB*tt/(2*pi*hbar**2))**(3/2.0)

    result = []

    # ideal gas
    nn = pp/(kB*tt)
    sigma = np.log(nqp*kB*tt/pp) + 5/2.0
    result.append((nn, sigma))

    # degenerate electrons, nondegenerate protons
    nn = ((5*me)/(hbar**2*pi**(4/3.0)*3**(2/3.0)))**(3/5.0) * pp**(3/5.0)
    sigma = np.log(nqp/nn) + 5/2.0
    result.append((nn,sigma))

    # degenerate electrons, degenerate protons
    nn = (5*me/(hbar**2*pi**(4/3.0)*3**(2/3.0)))**(3/5.0) * pp**(3/5.0)
    sigma = (2*pi**(5/3.0)/3**(2/3.0)) * (nqp/nn)**(2/3.0)
    result.append((nn,sigma))

    # degenerate relativistic electrons, degenerate protons
    nn = (4/(3**(1/3.0)*pi**(2/3.0)*hbar*cc))**(3/4.0) * pp**(3/4.0)    
    sigma = (2*pi**(5/3.0)/3**(2/3.0))*(nqp/nn)**(2/3.0)
    result.append((nn,sigma))

    # degenerate relativistic electrons, degenerate relativistic protons
    # same expressions as above
    # relativistic electrons, nonrelativistic protons
    #   problematic case -- one equation relates p and t, so you can
    #   specify inconsistent values.  The other equation relates only
    #   the product sigma*n (ie, entropy per volume), that is to say
    #   that entropy per baryon is ill defined.

    # relativistic electrons, relativistic protons
    # problematic -- as above

    return result


def aeos_guess_nn_tt(pp, sigma):
    pi = np.pi

    result = []

    # ideal gas
    kt = (2*pi*hbar**2/mp)**(3/5.0) * np.exp(2*sigma/5.0-1) * pp**(2/5.0)
    nqp = (mp*kt/(2*pi*hbar**2))**(3/2.0)
    nn = nqp*np.exp(5/2.0-sigma)
    result.append((nn,kt/kB))

    # degenerate electrons, nondegenerate protons
    nn = (5*me/(hbar**2*pi**(4/3.0)*3**(2/3.0)))**(3/5.0) * pp**(3/5.0)
    kt = (2*pi*hbar**2/mp) * nn**(2/3.0) * np.exp(2*sigma/3.0 - 5/3.0)
    result.append((nn,kt/kB))

    # degenerate electrons, degenerate protons
    nn = (5*me/(hbar**2*pi**(4/3.0)*3**(2/3.0)))**(3/5.0) * pp**(3/5.0)
    kt = (3**(2/3.0)*hbar**2/(pi**(2/3.0)*mp))*sigma*nn**(2/3.0)
    result.append((nn,kt/kB))

    # degenerate relativistic electrons, degenerate protons
    nn = (4/(3**(1/3.0)*pi**(2/3.0)*hbar*cc))**(3/4.0) * pp**(3/4.0)
    kt = ((3**(2/3.0)*hbar**2)/(pi**(2/3.0)*mp)) * sigma * nn**(2/3.0)
    result.append((nn,kt/kB))

    # degenerate relativistic electrons, degenerate relativistic protons
    # same expressions as above
    # relativistic electrons, nonrelativistic protons
    gamma = 3.0
    kt = (45*hbar**3*cc**3*pp/(gamma*pi**2))**(1/4.0)
    nn = 4*pi**2*gamma*kt**3/(45*hbar**3*cc**3*sigma)
    result.append((nn,kt/kB))

    # relativistic electrons, relativistic protons
    gamma = 100.0
    kt = (45*hbar**3*cc**3*pp/(gamma*pi**2))**(1/4.0)
    nn = 4*pi**2*gamma*kt**3/(45*hbar**3*cc**3*sigma)
    result.append((nn,kt/kB))

    return result
    

def aeos_guess_pp_tt(nn, sigma):
    pi = np.pi
    result = []

    # ideal gas
    kt = (2*pi*hbar**2/mp) * np.exp(2*sigma/3.0 - 5/3.0) * nn**(2/3.0)
    pp = nn*kt
    result.append((pp,kt/kB))

    # degenerate electrons, nondegenerate protons
    pp = (hbar**2*pi**(4/3.0)*3**(2/3.0)/(5*me))**(3/5.0) * pp**(3/5.0)
    kt = (2*pi*hbar**2/mp) * nn**(2/3.0) * np.exp(2*sigma/3.0 - 5/3.0)
    result.append((pp,kt/kB))

    # degenerate electrons, degenerate protons
    pp = (hbar**2*pi**(4/3.0)*3**(2/3.0)/(5*me)) * nn**(5/3.0)
    kt = (3**(2/3.0)/(mp*pi**2/3.0)) * sigma * nn**(2/3.0)
    result.append((pp, kt/kB))

    # degenerate relativistic electrons, degenerate protons
    pp = (3**(1/3.0)*hbar*cc*pi**(2/3.0)/4.0) * nn**(4/3.0)
    kt = (3**(2/3.0)*hbar**2/(pi**(2/3.0)*mp)) * sigma * nn**(2/3.0)
    result.append((pp,kt/kB))

    # degenerate relativistic electrons, degenerate relativistic protons
    # same expressions as above
    # relativistic electrons, nonrelativistic protons
    gamma = 3.0
    kt = (nn*sigma*45*hbar**3*cc**3/(4*pi**2*gamma))**(1/3.0)
    pp = gamma*pi**2*kt**4/(45*hbar**3*cc**3)
    result.append((pp,kt/kB))

    # relativistic electrons, relativistic protons
    gamma = 100.0
    kt = (nn*sigma*45*hbar**3*cc**3/(4*pi**2*gamma))**(1/3.0)
    pp = gamma*pi**2*kt**4/(45*hbar**3*cc**3)
    result.append((pp,kt/kB))

    return result


def aeos_find_tt_given(nn, sigma):
    # natural eos: pp, sigma function of nn, tt
    def func(logt):
        result = sigma - gsn_natural_entropy(nn, np.exp(logt))
        print result
        return result

    def find_root_bisect(logt0):
        # initial guess at range
        logtl, logth = logt0, logt0

        # expand range until it includes the root
        while(func(logtl) > 0.0):
            print "low", logtl
            if logtl <= 0.0: raise RuntimeError
            tl /= 2.0
        while(func(logth) < 0.0):
            print "high", logth
            if logth >= np.log(1e10): raise RuntimeError
            logth += 0.3
        
        # find the root
        return scipy.optimize.bisect(func, logtl, logth)

    def find_root_newton(logt0):
        return scipy.optimize.newton(func, logt0)
    
    logtfin = None
    for pp, tt in aeos_guess_pp_tt(nn, sigma):
        try:
            print "trying", tt
            logtfin = find_root_newton(np.log(tt))
        except RuntimeError: continue

    if logtfin is None: 
        raise RuntimeError, "No root found!"
    return np.exp(logtfin)


def aeos_find_nn_given(pp, tt):
    # natural eos: pp, sigma function of nn, tt
    def func(nn):
        return pp - pressure(nn, tt)

    def find_root_bisect(n0):
        # initial guess at range
        nl, nh = t0, t0

        # expand range until it includes the root
        while(func(nl) > 0):
            if nl == 0.0: raise RuntimeError
            nl /= 2.0
        while(func(nh) < 0):
            if np.isinf(nh): raise RuntimeError
            nh *= 2.0
            
        # find the root
        return scipy.optimize.bisect(func, nl, nh)

    def find_root_newton(n0):
        return scipy.optimize.newton(func, n0)
    
    pfin = None
    for nn, tt in guess_nn_tt(pp, tt):
        try:  pfin = find_root_bisect(nn)
        except: continue

    if pfin is None: 
        raise RuntimeError, "No root found!"
    return pfin


def aeos_find_nn_tt_given(pp, sigma):
    # natural eos: pp, sigma function of nn, tt
    def func(xx):
        nn, tt = xx
        return [sigma - entropy(nn, tt), pp - pressure(nn, tt)]

    nfin, tfin = None, None
    for pp, tt in guess_nn_tt(pp, sigma):
        try:  nfin, tfin = scipy.optimize.newton(func, x0)
        except: continue

    if nfin is None or tfin is None: 
        raise RuntimeError, "No root found!"
    return nfin, tfin

##############################
## EOS
##############################
eos = eos.lower()
if eos == 'scvh':
    1/0
    global_pressure = pressure_scvh(**eos_parameters)
elif eos == 'polytrope':
    1/0
    global_pressure = pressure_polytrope(**eos_parameters)
elif eos == 'gsn':
    # global_pressure = pressure_gsn(**eos_parameters)
    global_density = None 
elif eos == 'mesa':
    global_density = density_mesa(**eos_parameters)
    global_energy = energy_mesa(**eos_parameters)
else:
    raise SyntaxError, "Unknown Equation of State: %s" % eos

##############################
### Start of converted matlab routines
##############################


##############################
# Note change in arg order
def calc_derivatives(PM, r, sigma):
    P, M = PM
    rho = rho_of_P(P, sigma)

    dPdr = -(G * M / r**2) * rho
    dMdr = 4 * np.pi * r**2 * rho

    return np.array( [dPdr, dMdr] )

######################
### Equation of state
######################
#
# Adiabatic:
# P = K rho**gamma
# where K = exp((s/kB)*mu*mp*(gamma-1)) * (kB/(mu*mp))**gamma
# unsure about the above
#
# s = cp*log(T) - R*log(p)
#   = cp*log(T) - R*(log(R) + log(rho) * log(T))
#   = (cp-R)log(T) - Rlog(R) - Rlog(rho)
#   = cv log(T) - RlogR - Rlog(rho)
# cv log(T) = s + R log(rho) + R log(R)
# log(T) = s/cv + (R/cv)log(rho) + (R/cv)log(R)


def P_of_rho_old(rho,sigma):

    mu = 1.2
    gamma = 5/3.0

    R = kB / (mu*mp)
    cp = R*(gamma/(gamma-1))  # unused
    cv = R*(1/(gamma-1))  # unused

    # entropy term
    # GSN: is this correct?  Why not written the following way, which
    # is the same if it is correct?
    # K = R**gamma * np.exp(sigma*(gamma-1)*mu)
    K = R**gamma * np.exp(sigma)**((gamma-1)*mu)
    # GSN I don't think this can be correct.  P has units of erg^gamma /
    # cm^3 gamma deg K^gamma.  There must be a temperature to get rid
    # of the deg K units.  It's there implicitly in the form of the
    # entropy, but the entropy as given is dimensionless, so there's
    # something missing.
    P = K * rho**gamma
    return P


def P_of_rho(rho,sigma):
    # Replacing this with expressions from gsn.pdf just to start to
    # get a handle on why the original implementation doesn't seem to
    # be doing the right thing.

    # This will be ok until protons start to be degenerate.
    
    xx = 0.0   # for now assume no ionization
    alpha = 0.0  # for now assume electrons always degenerate

    cc = 1.5*np.log(mp/me) + 2.5
    if xx != 0.0 and xx != 1.0: 
        dd = -(1-xx) * np.log(1-xx) - xx*(1+alpha)*np.log(xx) + 2.5*alpha*xx
    else:
        # put in the limit by hand.
        dd = 0.0

    dim_factor = ((2*np.pi*hbar**2)/(me*mp**(5/3.0))) 
    exp_factor = np.exp((2/3.0)*(sigma-dd-cc))
    P = (1+xx) * rho**(5/3.0) * exp_factor * dim_factor

    # Simple, silly way to make a dimensionally correct polytropic EOS:
    # add a density scale and a dimensionless factor to give the
    # scaling we want.
    rho_scale = 1.0  # cgs
    gamma = 5/3.0
    P *= (rho/rho_scale)**(gamma-5/3.0)

    return P

### Equation of state
# Adiabatic:
# P = K rho**gamma
# where K = exp((s/kB)*mu*mp*(gamma-1))


def rho_of_P(P, sigma, Npts=100):
    # FIXME: DSP: needs fixing
    rho_vec = np.logspace(-15,5,Npts)
    P_vec = P_of_rho(rho_vec,sigma)
    # I've changed this to linear interpolation.  Scipy has spline
    # implementations under scipy.interpolate.fitpack, but it's not
    # set up for for interpolation in a single line of code.    
    rho = np.exp(scipy.interp(np.log(P), np.log(P_vec),np.log(rho_vec)))
    return rho


def LE_derivatives(uy,x,n):
    # n is polytropic index
    #
    # equation is:
    # d^2 u /dx^2 = -2/x du/dx - u**n
    # y = du/dx

    u,y = uy
    dudx = y
    # Divisions by zero cause throw an exception, which stops code
    # execution, so you can't just fill dydx with nan and then replace
    # it later.
    if x != 0:
        dydx = -(2/x)*y - u**n    
    else: 
        dydx = 0

    return np.array( [dudx, dydx] )

##############################
### Find hydrostatic equilibrium by various methods
##############################


# from integrate_hydrostatic.m
def hse_old(filename=None):
    """Find HSE by integrating a bunch of central pressures until the
    pressure becomes negative."""
    nPc = 41
    nr = 100
    
    # Set up calculation
    rmin = 1e-2  # cgs
    rmax = 1e11  # cgs

    Rearth = 6.378e8  # cgs
    Mearth = 5.974e27 # cgs
    Rjup = 7.1492e8   # cgs
    Mjup = 1.8987e27  # cgs

    sigma = 2  

    Pc = 1e6*1e6; # cgs
    logPc = np.log10(Pc)
    #Pc_vec = np.logspace(logPc-2,logPc+2,nPc)
    Pc_vec = np.logspace(0,15,nPc)
    Mc = 0

    # Calculate
    M_of_R = np.zeros(len(Pc_vec))
    R_vec  = np.zeros(len(Pc_vec))

    all_models = []    
    
    for ii in range(len(Pc_vec)):

        thisPc = Pc_vec[ii]
        PM_init = [thisPc, Mc]  # central values of pressure, mass

        # ode solver is different.  Looks to me like the matlab
        # routine gives you points wherever it feels like it.  With
        # scipy, you specify the values of the independent var at
        # which you want the dependent var.  Then it takes adaptive
        # stepsizes to get through the region, while making sure that
        # it gets values for the points you want.
        R = np.logspace(np.log10(rmin), np.log10(rmax), nr)
        # odeint calls the function like this: f(array_of_dep_vars,
        # indep_var).  You can either tell it to pass some extra stuff
        # with the 'args' argument like this: 
        # PM = scipy.integrate.odeint(calc_derivatives, PM_init, R,
        #                       args=(sigma,), rtol=1e-4, atol=1e-4)
        # or you can do something slicker and more general: create an
        # unnamed function that binds some of its variables when it's
        # defined and other variables when it's called.
        PM = scipy.integrate.odeint(lambda y, x: calc_derivatives(y, x, sigma), 
                                    PM_init, R, rtol=1e-4, atol=1e-4)

        # Find where P goes negative
        P,M = PM.transpose()
        # Could also write this instead of transpose()
        # P, M = PM[:,0], PM[:,1]
        signP = np.sign(P)
        diffsignP = np.diff(signP)
        # numpy's find returns all instances.  
        indicies_flip = pylab.find(diffsignP < 0)

        # If a zero crossing wasn't found, fill in dummy values for
        # the results and continue
        if len(indicies_flip) == 0:
            all_models.append(dict(P=None, R=None, M=None))
            M_of_R[ii] = 0
            R_vec[ii] = 0
            continue

        # Get the index of the first zero crossing.
        ndx_flip = indicies_flip[0]
        ndces = np.array( [ndx_flip, ndx_flip+1] )
        # note different order of arguments.
        R_at_Pzero = scipy.interp(0, P[ndces],R[ndces])
        M_at_Pzero = scipy.interp(0, P[ndces],M[ndces])
        P[ndx_flip+1] = 0
        R[ndx_flip+1] = R_at_Pzero
        M[ndx_flip+1] = M_at_Pzero

        P = P[:ndx_flip+2]
        R = R[:ndx_flip+2]
        M = M[:ndx_flip+2]

        all_models.append(dict(P=P, R=R, M=M))
        # This also works if you want to be more verbose
        # all_models.append(dict(pressure=P, radius=R, mass=M))

        M_of_R[ii] = M_at_Pzero
        R_vec[ii] = R_at_Pzero

        # Is P provably a decreasing function of R?  If so all of the
        # sign-flip business can be replaced with something like:
        #
        # ndx_flip = P.searchsorted(0)
        # R_at_Pzero = scipy.interp(0, P, R)
        # M_at_Pzero = scipy.interp(0, P, M)

    pylab.figure(1)
    pylab.clf()
    pylab.loglog(M_of_R/Mearth, R_vec/Rearth,'o')
    pylab.xlabel(r'$M \, (M_\oplus)$')
    pylab.ylabel(r'$R \, (R_\oplus)$')

    if filename: 
        [pylab.savefig(filename + '-m-r-relation.' + ext) for ext in exts]

    # plot the models that we have
    pylab.figure(2)
    pylab.clf()

    pylab.subplot(221)
    pylab.xlabel(r'$R \, (R_\oplus)$')
    pylab.ylabel(r'$M \, (M_\oplus)$')
    for model in all_models:
        if model['P'] is not None:
            pylab.loglog(model['R']/Rearth, model['M']/Mearth)

    pylab.subplot(222)
    pylab.xlabel(r'$R \, (R_\oplus)$')
    pylab.ylabel(r'$\rho \, (cgs)$')
    for model in all_models:
        if model['P'] is not None:
            pylab.loglog(model['R']/Rearth, rho_of_P(model['P'], sigma))

    pylab.subplot(223)
    pylab.xlabel(r'$R \, (R_\oplus)$')
    pylab.ylabel(r'$P \, (cgs)$')
    for model in all_models:
        if model['P'] is not None:
            pylab.loglog(model['R']/Rearth, model['P'])            

    if filename: 
        [pylab.savefig(filename + '-models.' + ext) for ext in exts]


# from integrate_hydrostatic_v02_laneemden.m
def hse_lane_emden(filename=None, nx = 101, nrho=102):
    """Find hydrostatic equilibrium by integrating the Lane Emden
    equation and then choosing a bunch of different central pressures
    to make it dimensional."""
    # Lane-Emden equation
    # set up A, z, W variables.
    
    # constants
    mu = 1.2 
    gamma = 5.0/3.0  # beware integer division in Python
    n = 1/(gamma-1)  # polytropic index
    # Rgas has units cm^2 / deg K s^2
    Rgas = kB / (mu*mp)
    cp = Rgas*(gamma/(gamma-1)) # unused
    cv = Rgas*(1/(gamma-1))  # unused

    ### L-E Equation
    # Lane-Emden equation:
    # 1/xi^2 d/dxi {xi^2 dtheta/dxi} = -theta**n
    # theta(0) = 1
    # dtheta/dxi at xi=0 = 0
    # xi = r/alpha
    #
    # or
    #
    # d^2theta/dxi^2 = -2/xi dtheta/dxi - theta**n
    # set u = theta, x = xi, y = dtheta/dxi
    # so,
    # coupled equations:
    # dy/dx = -(2/x) y - u**n
    # du/dx = y

    xvec = np.linspace(0, 10, nx)
    # u[0] = theta[0] = 1;
    # y[0] = du/dx[0] = dtheta/dxi[0] = 0
    uy_init = [1, 0]
    UY = scipy.integrate.odeint(lambda y,x: LE_derivatives(y, x, n),
                                uy_init, xvec, rtol=1e-9)
    # here 'array' is being used to indicate that a new copy should be made.
    Xfull = np.array(xvec)
    Ufull, Yfull = np.array(UY).transpose()

    ndx_good = pylab.find((Ufull == Ufull.real) & (Ufull >= 0))
    U = Ufull[ndx_good] # this is theta
    Y = Yfull[ndx_good] # this is dtheta/dxi
    X = Xfull[ndx_good] # this is xi

    # Instead of the following, you can write what's below
    # ndx_Y_pos = pylab.find(Y>0)
    # Y[ndx_Y_pos] = 0
    Y[Y>0] = 0
    
    xi = X
    theta = U

    # Solved L-E Equation
    # Entropy, rho_c terms
    # sigma = 3 # specific entropy
    sigma = 0  # specific entropy
    K = Rgas**gamma * np.exp(sigma)**((gamma-1)*mu)

    min_rho = 1e-6  # cgs
    max_rho = 10**8  # cgs
    # rho_c from 100 to 10**8 g/cm**3

    rho_c_vec = np.logspace(np.log10(min_rho),np.log10(max_rho),nrho)

    # GSN: Don't have K+W in front of me, but don't see how this has the correct units.
    # from the notes ... inverse of Kippenhahn & Weigert's A
    alpha_vec = 1/np.sqrt( (4*np.pi*G/((n+1)*K)) * rho_c_vec**((n-1)/n) )
    # Apply to different stars/planets with different central rho
    # r_array = zeros(numel(xi),numel(rho_c_vec));
    # rho_array = zeros(size(r_array));
    # for ii = 1:numel(rho_c_vec)
    #    this_rho_c = rho_c_vec(ii);
    #    this_alpha = alpha_vec(ii);
    #    this_r   = xi*this_alpha;
    #    this_rho = this_rho_c * theta;
    #    r_array(:,ii)   = this_r;
    #    rho_array(:,ii) = this_rho;
    #end
    # Can do it simpler by just multiplying column vector by row vector
    #
    # Somewhat tortured syntax here.  dot() gives matrix
    # multiplication, and then numpy needs a little help figuring out
    # which are row vectors and which are column vectors, which is
    # what the np.newaxis things are doing.
    #
    r_array = np.dot(xi[:, np.newaxis], alpha_vec[np.newaxis, :])
    rho_array = np.dot(theta[:, np.newaxis], rho_c_vec[np.newaxis,:])
    P_array = K * rho_array**gamma
    ## Postprocess models
    #
    # numpy arrays aren't assumed to be 2d, so this looks a little different.    
    rho_c_array = np.tile(rho_c_vec[np.newaxis,:], (len(xi), 1)) 
    xi_array    = np.tile(xi[:,np.newaxis], (1,len(rho_c_vec))) 
    Y_array     = np.tile(Y[:,np.newaxis], (1,len(rho_c_vec)))

    # negative indicies count from the end
    Rvec = r_array[-1,:]
    # This seems to give the wrong value for the mass
    #Menc_array = zeros(size(r_array));
    #for ii = 1:numel(rho_c_vec)
    #    this_r   = r_array(:,ii);
    #    this_rho = rho_array(:,ii);
    #    dmdr = 4 * pi * this_r.**2 .* this_rho;
    #    Menc_array(:,ii) = cumtrapz(this_r,dmdr);
    #end
    # Mvec = Menc_array(end,:);
    # Looked at K+W, think what you've written is right, but much
    # prefer to make the associativity explicit.
    # from K-W eq. (19.19)
    Mvec_analytic = 4*np.pi*rho_c_vec * Rvec**3 * (-Y[-1]/xi[-1])
    rho_bar_vec_analytic = 3*Mvec_analytic / (4*np.pi*Rvec**3)
    # this gives the correct ratio; the numerical one overestimates
    # the mass.  Why?
    ratio = rho_c_vec / rho_bar_vec_analytic; 
    # Calculate mass enclosed
    # From K & W p. 179, section 19.4, equation (19.18)
    Menc_array_analytic = (4*np.pi) * rho_c_array * r_array**3 * (-Y_array/xi_array)    
    Menc_array_numerical = np.zeros_like(Menc_array_analytic)
    Eint_array_numerical = np.zeros_like(Menc_array_analytic)
    for ii in range(1,len(rho_c_vec)):
        this_r   = r_array[:,ii]
        this_rho = rho_array[:,ii]
        this_P   = P_array[:,ii]
        dmdr = 4 * np.pi * this_r**2 * this_rho
        # Note the change in order of arguments
        Menc_array_numerical[1:,ii] = scipy.integrate.cumtrapz(dmdr, this_r)
        dEdr = 4 * np.pi * this_r**2 * this_P
        Eint_array_numerical[1:,ii] = scipy.integrate.cumtrapz(dEdr, this_r)

    Mvec_numerical = Menc_array_numerical[-1,:]
    Eint_vec = Eint_array_numerical[-1,:]
    # For some reason, the numerical one is 1.5418 times the analytic one.

    # For Jupiter
    Eint_jup = scipy.interp(Mjup, Mvec_numerical,Eint_vec)
    R_jup = scipy.interp(Mjup, Mvec_numerical,Rvec)
    # For RGHJ
    Ls = 4000*Lsun 
    a = 5*AU
    F0 = Ls / (4 * np.pi * a**2)
    P0 = np.pi * Rjup**2 * F0
    Deltat = 1e6*year
    E0 = P0 * Deltat

    ########################
    ### Plotting
    
    pylab.figure(1)
    pylab.clf()
    pylab.loglog(Mvec_numerical/Mjup, Rvec/Rjup,'o')
    pylab.xlabel(r'$M \, (M_J)$')
    pylab.ylabel(r'$R \, (R_J)$')

    if filename: 
        [pylab.savefig(filename + '-m-r-relation.' + ext) for ext in exts]

    # plot the models that we have
    pylab.figure(2)
    pylab.clf()

    pylab.subplot(221)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$M \, (M_J)$')
    pylab.loglog(r_array/Rjup, Menc_array_numerical/Mjup)

    pylab.subplot(222)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$\rho \, (cgs)$')
    pylab.loglog(r_array/Rjup, rho_array)

    pylab.subplot(223)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$P \, (cgs)$')
    pylab.loglog(r_array/Rjup, P_array)            

    pylab.subplot(224)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$E_I \, (cgs)$')
    pylab.loglog(r_array/Rjup, Eint_array_numerical)            

    if filename: 
        [pylab.savefig(filename + '-models.' + ext) for ext in exts]

##############################
### End of converted matlab routines
##############################


def make_global_density(sigma):
    # for interpolation table
    rho_min = 1e-30 # cgs
    rho_max = 1e30 # cgs
    rho_min = 1e-5
    rho_max = 1e5
    n_rho = 10000

    ##############################
    # set up function to interpolate to get rho
    rho = np.logspace(np.log10(rho_min),np.log10(rho_max),n_rho)
    sigma = 0*rho + sigma
    # FIXME -- temporary hack
    log_rho_1d = scipy.interpolate.interp1d(np.log(global_pressure(rho, sigma)),
                                            np.log(rho))
    def rho(pp, sig):
        return 10**log_rho_1d(log10(pp))
    return rho
        

def hse(pc=None, sigma=5.0, filename=None, 
        density = global_density, 
        relative_p_min = True, p_min_rel = 1e-9, p_min_abs = 1e-6*bar):

    """Find hydrostatic equilibrium using pressure as the independent
    variable."""
    if pc is None:
        pc = np.logspace(5,25,40)
    elif not np.iterable(pc):
        pc = [pc]

    def derivs(yy,xx):
        lm, lr = yy
        lp = xx
        # Note that we're grabbing sigma from the surrounding context
        lrho = np.log(density(np.exp(lp), sigma))
        dlm_dlp = - (4*pi/G)*np.exp(4*lr + lp - 2*lm)
        dlr_dlp = - (1/G)*np.exp(lr+lp-lm-lrho)
        return [dlm_dlp, dlr_dlp]
    
    ##############################
    # practical constants
    pi = np.pi

    # number of values of the pressure to include in a single model calc.
    n_p_model = 10000

    # radius at which to start the calculation
    r_min = 1e5 # cgs

    # Stop the calculation when pressure reaches a specific value or
    # when pressure reaches a specific fraction of the central
    # pressure.    

    # density = global_density(sigma)
    
    ##############################
    # loop over central pressures to calculate models
    p_models, m_models, r_models, rho_models = [], [], [], []
    for the_pc in pc:

        if relative_p_min:
            p_min = the_pc*p_min_rel
        else: 
            p_min = p_min_abs

        lp = np.linspace(np.log(the_pc), np.log(p_min), n_p_model)

        # set up central conditions.  This assumes that there's no
        # density cusp at the center, but don't expect it to make much
        # difference even if there was.
        rhoc = density(the_pc, sigma)

        mc = 4*pi*r_min**3*rhoc / 3.0

        # integration in log space, set atol for const fractional
        # error per timestep.  rtol shouldn't be used since it will
        # have a weird mapping into physical space.
        #derivs([1.1, 2.2], 3.3)
        
        result = scipy.integrate.odeint(derivs,
                                        [np.log(mc), np.log(r_min)], lp,
                                        rtol=None, atol=1e-4)
        lm, lr = result.transpose()
        p_models.append(np.exp(lp))
        m_models.append(np.exp(lm))
        r_models.append(np.exp(lr))
        # FIXME -- make less annoying
        sigma_arr = 0*lp + sigma
        rho_models.append(density(np.exp(lp), sigma_arr))
            
    p_models = np.array(p_models)
    m_models = np.array(m_models)
    r_models = np.array(r_models)
    rho_models = np.array(rho_models)
    
    # If any of the final radii are less than 10x the min radius, print a warning
    if (r_models[:,-1] < 100*r_min).any():
        print "WARNING: r_min is close to final radius", (r_models[:,-1]/r_min).min()
    # If the central pressure is too close to the final pressure, print a warning
    if not relative_p_min and (p_models[:,0] < 100*p_min_abs).any():
        print "WARNING: surface pressure is close to central pressure"
        
    ########################
    ### Plotting

    pylab.figure(1)
    pylab.loglog(m_models[:,-1]/Mjup, r_models[:,-1]/Rjup,'-o')
    pylab.xlabel(r'$M \, (M_J)$')
    pylab.ylabel(r'$R \, (R_J)$')
    
    if filename: 
        [pylab.savefig(filename + '-m-r-relation.' + ext) for ext in exts]

    # plot the models that we have
    pylab.figure(2)
    # pylab.clf()

    pylab.subplot(221)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$M \, (M_J)$')
    pylab.loglog(r_models.transpose()/Rjup, m_models.transpose()/Mjup)
    # pylab.xlim(1e-2,1)
    # pylab.ylim(1e-4,1e2)

    pylab.subplot(222)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$\rho \, (cgs)$')
    pylab.loglog(r_models.transpose()/Rjup, rho_models.transpose())
    # pylab.xlim(1e-2,1e1)
    # pylab.ylim(1e-2,1e4)

    pylab.subplot(223)
    pylab.xlabel(r'$R \, (R_J)$')
    pylab.ylabel(r'$P \, (bar)$')
    pylab.loglog(r_models.transpose()/Rjup, p_models.transpose()/BAR)
    # pylab.xlim(1e-2,1)
    # pylab.ylim(1e5,1e12)

    if filename: 
        [pylab.savefig(filename + '-models.' + ext) for ext in exts]
    
    return p_models, m_models, r_models, rho_models


def make_density_function(pressure, sigma, rho=None):
    """ Returns a functions that takes log base e of the pressure and
    returns log base e of the density."""
  
    if rho is None:
        rho = np.logspace(-30, 30, 10000)

    log_rho = scipy.interpolate.interp1d(np.log(pressure(rho, sigma)),
                                         np.log(rho))
    return log_rho


def one_simple_model(mm, pc, log_rho):
    """This integrates equations of HSE using mass as the independent
    variable.  mm are the mass points, mm[0] is the central point,
    mm[-1] is the total mass of the object, pc is the central
    pressure, and log_rho is a function that gives the log of the
    density given the log of the pressure.  The idea is to wrap this
    wrap this function inside a loop that will adjust pc until the
    sufrace pressure meets some condition.

    Mass is used as the indep variable with the idea that this will be
    necessary for an evolutionary calculation.  This is true if the
    entropy does not remain constant.  If the planet remains
    entropically well-mixed, we can use pressure as the indep var and
    be happy."""

    def derivs(yy,xx):
        lr, lp = yy
        lm = xx
        lrho = log_rho(lp)
        dlr_dlm = (1/(4*np.pi)) * np.exp(lm - 3*lr - lrho)
        dlp_dlm = -(G/(4*np.pi)) * np.exp(2*lm - 4*lr - lp)
        return [dlr_dlm, dlp_dlm]
    
    # Know central density and mass enclosed within first grid point.
    # So find the location of the first grid point.  This assumes that
    # there's no density cusp at the center, but don't expect it to
    # make much difference even if there was.

    rhoc = np.exp(log_rho(np.log(pc)))
    r_min = (3*mm[0]/(4*np.pi*rhoc))**(1/3.0)

    # integration in log space, set atol for const fractional
    # error per timestep.  rtol shouldn't be used since it will
    # have a weird mapping into physical space.
    result = scipy.integrate.odeint(derivs,
                                    [np.log(r_min), np.log(pc)], np.log(mm),
                                    rtol=None, atol=1e-4)
    lr, lp = result.transpose()
    lrho = log_rho(lp)

    return lr, lp, lrho


def estimate_central_pressure(mm, log_rho, pci=None):
    # estimate of central pressure for a polytrope
    # num = (4*pi)**(1/3) G rho_0**(4/3) M**(2/3) 
    # denom = p_0
    # qty = num/denom
    # p_c = p_0 * qty **(gamma/(gamma-4/3))
    # with rho_0, p_0 defined by
    # p = p_0 (rho/rho_0)**gamma
    # so p_0 = hbar^2/m_e m_p^5/3 
    # and rho_0 = 1g/cc

    """Take a guess at the central pressure for a given model to start
    the iteration."""
    # hse(pc=None, sigma=5.0, filename=None, 
    #     pressure = global_pressure, 
    #     relative_p_min = True, p_min_rel = 1e-9, p_min_abs = 1e-6*bar):
    #hse_gsn(pc=None, sigma=5.0, gamma = (5,3), filename=None)
    pass


def one_model(mm, log_rho, pci,
              relative_p_min = True, p_min_rel = 1e-9, p_min_abs = 1e-6*bar):
    """Calculate one model by finding the correct central pressure."""

    # This does indeed converge for the following setup:
    # pressure = pressure_polytrope(gamma=(5,3))
    # lrho = make_density_function(pressure, 10.0)
    # one_model([5.5e26, 5.5e29], lrho, exp(27.5718609887), relative_p_min=False, p_min_abs=1e3)    
    def function(lpc):
        if relative_p_min:
            lp_min = lpc + log(p_min_rel)
        else: 
            lp_min = np.log(p_min_abs)

        lr, lp, lrho = one_simple_model(mm, np.exp(lpc), log_rho)
        return lp[-1] - lp_min

    lpcf = scipy.optimize.newton(function, np.log(pci), tol=1e-05)
    lpcf = scipy.optimize.bisect(function, np.log(0.9*pci), np.log(1.1*pci), rtol=1.0)

    print np.exp(lpcf)
    
    lr, lp, lrho = one_simple_model(mm, np.exp(lpcf), log_rho)
    return np.exp(lr), np.exp(lp), np.exp(lrho)


def thermal_en_1(rs, ps, sigma):
    # query equation of state
    # FIXME -- argh, fix this already
    sigmas = 0*ps + sigma
    en_dens = global_energy(ps, sigmas)
    igrand = ave(4*np.py*rs**2*en_dens)
    return (igrand*diff(rs)).sum()
    
    
def thermal_en_2(rs, ps, sigma):
    def igrand(rr):
        return 4*np.pi*rr**2 * en_func(rr)

    # FIXME -- argh, fix this already
    sigmas = 0*ps + sigma
    en_dens = global_energy(ps, sigmas)
    en_func = interp_1d(rs, en_dens)

    return scipy.integrate.quad(igrand, rs[0], rs[-1])[0]

def grav_pe(rs, rhos):
    igrand = ave(4*np.pi*rs**2*rhos)
    ms = (igrand*np.diff(rs)).cumsum()
    igrand2 = - G*ms*ave(4*np.pi*rs*rhos)
    return (igrand2*np.diff(rs)).sum()


def grav_pe_very_slow(rs, rhos):
    # Should take integrals from zero... could add small correction
    # for central bit assuming constant density.
    # Maybe someday check error estimates on integrals...
    rhof = interp_1d(rs,rhos)

    def igrand(rr):
        return 4*np.pi*rr**2 * rhof(rr)

    def m_func(rr):
        return scipy.integrate.quad(igrand, rs[0], rr)[0]

    def igrand_2(rr):
        return - G*m_func(rr)*4*np.pi*rr*rhof(rr)

    return scipy.integrate.quad(igrand_2, rs[0], rs[-1])


def grav_pe_slow(rs, rhos):
    # Should take integrals from zero... could add small correction
    # for central bit assuming constant density.
    # Maybe someday check error estimates on integrals...
    rhof = interp_1d(rs,rhos)

    def igrand(rr):
        return 4*np.pi*rr**2 * rhof(rr)

    def m_integral(rl, rh):
        return scipy.integrate.quad(igrand, rl, rh)[0]

    def igrand_2(rr):
        return - G*m_func(rr)*4*np.pi*rr*rhof(rr)

    ms = [m_integral(rl, rh) for rl, rh in zip(rs[0:], rs[1:])]
    m0 = [ 4*np.pi*rs[0]**3 * rhos[0] / 3.0 ] 
    
    dm_points = m0 + ms
    m_points = np.array(dm_points).cumsum()

    m_func = interp_1d(rs, m_points) 

    return scipy.integrate.quad(igrand_2, rs[0], rs[-1])


##############################
### Utility functions
##############################


def make_grid(*axs):
    """Take an arbitrary number of 1d arrays and return a "filled-out"
    grid of values suitable for computing things at, e.g. every point
    in space.

    X,Y,Z = make_grid(xs, ys, zs) 

    X is a 3d array containing all of the x coordinates so you can do
    things like

    f = X**2 + Y*sin(Z)

    and have the value at every point in space."""
    import operator
    shape = [len(ax) for ax in axs]
    ntot = reduce(operator.mul, shape)
    ntile = 1
    nrep = ntot
    result = []
    for ax in axs:
        nrep /= len(ax)
        mesh = np.tile(np.repeat(ax,nrep),ntile).reshape(shape)
        ntile *= len(ax)
        result.append(mesh)
    return result


def grid_to_points(*vs):
    """take a (nx, ny, ...) grid of points as produced by make_grid or
    mgrid and produce a list of points with shape (ntps, ndim)"""
    if len(vs)==1: 
        return vs[0].ravel()
    return np.array([vv.ravel() for vv in vs]).transpose()


def popKeys(d, *names):
    """Pull keywords with certain names out of a dictionary.  Return a
    new dict of with the desired names/values, and delete them from
    the original dict.  Typically useful when processing keyword
    args"""
    return dict([(k, d.pop(k)) for k in names if k in d])


def ave(a, n=1, axis=-1):
    if n==0: return np.array(a)
    if n < 0: raise ValueError, 'order must be non-negative but got ' + repr(n)
    a = np.asarray(a)
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return ave(0.5*(a[slice1]+a[slice2]), n-1, axis=axis)
    else:
        return 0.5*(a[slice1]+a[slice2])

##############################
### Routines to read and write fortran files.  
###
### Not as complicated as they look--- the complexity comes from the
### fact that scipy recently removed the fread functions so I'm
### keeping both versions around for a while via if statements
##############################


def version_string_to_tuple(ss):
    return tuple([int(aa) for aa in ss.split('.')])

# Not positive which version of scipy removed them 
# 0.10.0: no
# circa 0.7: yes
scipy_has_fwrite = version_string_to_tuple(scipy.__version__) < (0,10)


def write_fortran(f, arr, swap=False, pad_type='i'):
    # not sure where this numpy version should happen
    if scipy_has_fwrite:
        scipy.io.fwrite(f, 1, np.array([arr.nbytes]), pad_type, swap)
        scipy.io.fwrite(f, arr.size, arr, arr.dtype.char, swap)
        scipy.io.fwrite(f, 1, np.array([arr.nbytes]), pad_type, swap)
    else:
        pad_array = np.array([arr.nbytes]).astype(pad_type)
        
        if swap: 
            pad_array = pad_array.byteswap()
            arr = arr.byteswap()

        pad_array.tofile(f)
        arr.tofile(f) 
        pad_array.tofile(f)
        

def read_fortran(f, type_='f', num=None, swap=False, pad_type='i'):
    """Read one unformatted fortran record.
    num = number of data to read.  If None, compute it from the pad word
    swap = whether or not to byte swap
    intType = specifies width of pad words on the computer where the
      files were written."""

    if scipy_has_fwrite:
        pad = scipy.io.fread(f, 1, pad_type, pad_type, swap)
    else: 
        pad = np.fromfile(f, pad_type, 1)
        if swap: pad = pad.byteswap()

    if len(pad) != 1: raise EOFError

    c1 = pad[0]
    if num is None: num = c1/np.array([0], type_).itemsize

    if scipy_has_fwrite:
        dat = scipy.io.fread(f, num, type_, type_, swap)
    else:
        dat = np.fromfile(f, type_, num)
        if swap: dat = dat.byteswap()

    if scipy_has_fwrite:
        c2 = scipy.io.fread(f, 1, pad_type, pad_type, swap)[0]
    else:
        pad2 = np.fromfile(f, pad_type, 1)
        if swap: pad2 = pad2.byteswap()
        c2 = pad2[0]

    assert c1 == c2 == dat.nbytes
    return dat


def skip_fortran(f, n=1, swap=False, pad_type='i'):
    """Skip one unformatted fortran record.
    intType = specifies width of pad words on the computer where the
      files were written."""
    for i in range(n):
        if scipy_has_fwrite:
            c1 = scipy.io.fread(f, 1, pad_type)[0]
        else:
            pad1 = np.fromfile(f, pad_type, 1)
            if swap: pad1 = pad1.byteswap()
            c1 = pad1[0]
            
        p1 = f.tell()
        f.seek(c1, 1)
        p2 = f.tell()

        if scipy_has_fwrite:            
            c2 = scipy.io.fread(f, 1, pad_type)[0]
        else:
            pad2 = np.fromfile(f, pad_type, 1)
            if swap: pad2 = pad2.byteswap()
            c2 = pad2[0]

        assert c1 == c2 == p2-p1


def n_quantum(temp, mm=1.67e-24/1836.0):
    return (mm*kb*temp/(2*pi*hbar**2))**1.5


n_quantum_electron = n_quantum 


def n_quantum_proton(temp):
    return n_quantum(temp, mm=1.67e-24)


def read_all_fortran(f, spec, swap=False, warning=False, intType='i'):
    """Read all fortran data from a file.  spec is a list of types of each block"""
# What are the use cases?
# 1) read all blocks using one type
# 2) read a succession of blocks of different type, specifying types as a list
#
#     """Read all data, specifying the types and number or elements, or
#     just the types (taking number of elements from the file, or
#     nothing, in which case everything is read as bytes"""
    closeFile = False
    if type(f) in types.StringTypes:
        closeFile = True
        f = open(f)
        
    if len(spec) > 1:
        result = [read_fortran(f, s, None, swap=swap, pad_type=intType)
                  for s in spec]
    else:
        result = []
        while not f.read(1) == '':
            f.seek(-1,1)
            dat = read_fortran(f, spec, None, swap=swap, pad_type=intType)
            result.append(dat)
        
    if closeFile:
        # Check to make sure that everything was read
        if f.read() != '':
            if warning: print "Warning, there's more to be read."
            else: raise RuntimeError
        f.close()

    return result


##############################

# ##############################
# # For DSP, meditation on functions as data 
# 
# # Python has special syntax for a useful function argument: *, which
# # means "All arguments not mentioned so far."  It's used like this:
# 
# def f1(x, y, *args):
#     print "named args:", x, y, "  other args:", args
# 
# # Calling this gives:
# # In [42]: f1(1,2)  => named args: 1 2   other args: ()
# # In [43]: f1(1,2,3) => named args: 1 2   other args: (3,)
# # In [44]: f1(1,2,3,4) => named args: 1 2   other args: (3, 4)
# # 
# # x and y are mandatory, so f(1) is an error, but args can be anything
# # from an empty tuple to... anything at all.
# #
# # There's special syntax for another function argument, **, which
# # means "all keyword arguments not mentioned so far."  The indicated
# # arg will be a dict containing all keyword args.  
# 
# def f2(x, y, z=None, **kw):
#     print "args:", x, y, "  kw arg", z, "  other args:", kw
# 
# # This lets you do:
# # In [60]: f2(1,2) => args: 1 2   kw arg None   other args: {}
# # In [61]: f2(1,2, z=3) => args: 1 2   kw arg 3   other args: {}
# # In [62]: f2(1,2, z=3, mu=55, nu=66) => args: 1 2   kw arg 3   other args: {'mu': 55, 'nu': 66}
# 
# # Now you can combine them to produce a function that will take
# # _anything_ as arguments.
# 
# def f3(*aa, **kw):
#     print "args", aa, "  kw args", kw
# 
# # In [64]: f3(1,2) => args (1, 2)   kw args {}
# # In [65]: f3(1,2, a=3, b=4) => args (1, 2)   kw args {'a': 3, 'b': 4}
# 
# # Now, what if you have a tuple floating around and want to use it as
# # a function argument?  This doesn't work:
# # args = (1,2); f1(args)
# 
# # because f1 needs _two_ arguments, and it just received _one_
# # argument (the tuple that contained both values).  You need syntax
# # that says "Explode this tuple into a function argument so that it
# # looks like a normal function call.  That is (again) *:
# 
# # In [67]: f1(args) => TypeError: f1() takes at least 2 arguments (1 given)
# # In [68]: f1(*args)=> named args: 1 2   other args: ()
# 
# # and similarly for **kw:
# # In [72]: f3(1,2,z=3,kw) => SyntaxError: non-keyword arg after keyword arg
# # In [73]: f3(1,2,z=3,**kw) => args: 1 2   kw arg 3   other args: {'a': 5, 'b': 7}
# 
# # What this allows you to do is write functions that operate on functions
# 
# # This is pretty easy:
# 
# def func_add_1(f1, f2):
#     def result(*aa, **kw):
#         return f1(*aa, **kw) + f2(*aa, **kw)
#     return result
# 
# # and would be used like this:
# # def rho_h(p,t): return foo
# # def rho_he(p,t): return bar
# # rho = func_add_1(rho_h, rho_he)
# # rho(1e6, 100)
# 
# # This syntax is standard for many languages but is falling out of
# # favor in Python: Lambda just allows you to define a function without
# # naming it.
# 
# def func_add_1(f1, f2):
#     return lambda *aa, **kw: f1(*aa, **kw) + f2(*aa, **kw)
# 
# # But anyway, why limit ourselves to adding _two_ functions.  How about this:
# 
# def func_add_2(*fs):
#     def result(*aa, **kw):
#         the_sum = fs[0](*aa, **kw)
#         for ff in fs:
#             the_sum += ff(*aa, **kw) # dss comment: does this double-count fs[0]?
#         return the_sum
#     return result
# 
# # Or this, somewhat more aesthetically appealing version of the same thing:
# 
# def func_add_3(*fs):
#     def result(*aa, **kw):
#         values = [ff(*aa, **kw) for ff in fs]
#         return reduce(lambda x,y: x+y, values)
#     return result
# 
# # But, notice that the fact that we're adding things just appears in
# # the form of that anonymous function passed to 'reduce'.  Why not
# # make that an argument?
# 
# def func_reduce(f_reduce, *fs):
#     def result(*aa, **kw):
#         values = [ff(*aa, **kw) for ff in fs]
#         return reduce(f_reduce, values)
#     return result
# 
# # now you can do things like this:
# # rho = func_reduce(lambda x,y: x+y, rho_h, rho_he)
# # rho = func_reduce(lambda x,y: x+y, rho_h, rho_he, rho_metals)
# # or 
# # total_prob = func_reduce(lambda x,y: x*y, p_event_1, p_event_2, p_event3)
# # and func_reduce handles all of these cases.  
# 
# # Incidentally, python defines some of the common operations for you,
# # so you can type this instead of the lambda functions:
# # import operator
# # rho = func_reduce(operator.add, rho_h, rho_he)
# 
# # Bringing this back to earth a little bit, you mentioned weighted
# # sums for the density at least, which would look something like this.
# # The function is getting two different things, functions and weights,
# # so it can't use Python's * arg syntax anymore.  You have to pass in
# # a list of weights and a list of functions.
# def func_wadd(ws, fs):
#     def result(*aa, **kw):
#         the_sum = ws[0] * fs[0](*aa, **kw)
#         for ww, ff in zip(ws, fs):
#             the_sum += ww * ff(*aa, **kw)
#         return the_sum
#     return result
# 
# # then it's called like this:
# # rho = func_wadd((0.74, 0.24, 0.02), (rho_h, rho_he, rho_z))
# 
# # End rumination
# ##############################
