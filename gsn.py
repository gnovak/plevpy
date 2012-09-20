####################
# gsn.py : "playground" for code development
#
# Contains sample implementations of scipy interpolation routines.
#

####################
# which ones to use:
# Rbf seems robust.
# griddata: 'nearest'
# SmoothBivariateSpline: Seems good

####################
# unstructured data
# 
# interp2d: works, but very touchy
# Rbf: seems very nice
# griddata: can only get 'nearest' to work
# SmoothBivariateSpline: Seems good
# LSQBivariateSpline: Don't know how to feed it a sequence of knots

# These are inside griddata
# NearestNDInterpolator
# LinearNDInterpoolator
# CloughToucher2DInterpolator

####################
# grid data
#
# 1) interp2d: can't get this to work.
# 2) RectBivariateSpline: seems to work
# 3) RectSphereBivariateSpline: not in my version of scipy

import scipy.interpolate
import pylab as pl
from numpy import *

def ff(xx,yy):
    return sin(xx)*yy

def the_plot():
    x,y = linspace(0,2*pi,100), linspace(-2,2,100)
    X,Y = make_grid(x,y)
    Z = ff(X,Y)
    pl.clf()
    pl.pcolormesh(X,Y,Z)
    pl.colorbar()

def grid_to_points(*vs):
    # take a (nx, ny, ...) grid of points as produced by make_grid or
    # mgrid and produce a list of points with shape (ntps, ndim)
    if len(vs)==1: 
        return vs[0].ravel()
    return array([vv.ravel() for vv in vs]).transpose()

##############################
# Retangular grids
def i1():
    # interp2d
    # This does not seem to work.  It also does not seem to honor
    # bounds_error or fill_value

    #xx = linspace(0,2*pi,39)
    #yy = linspace(-2,2,41)
    xx = linspace(0,2*pi,12)
    yy = linspace(-2,2,11)
    X,Y = make_grid(xx,yy)
    Z = ff(X,Y)

    # Note that in this use, xx,yy and 1d, Z is 2d.  
    # The output shapes are 
    # fint: ()   ()   => (1,)
    # fint: (n,) ()   => (n,)
    # fint: ()   (m,) => (1,m)
    # fint: (n,) (m,) => (n,m)

    fint = scipy.interpolate.interp2d(xx,yy, Z)

    xfine = linspace(0.0,2*pi,99)
    yfine = linspace(-2,2,101)
    XF, YF = make_grid(xfine, yfine)
    ZF = fint(xfine,yfine).transpose()

    pl.clf()
    pl.pcolormesh(XF,YF,ZF)
    pl.colorbar()

##############################
# Unstructured grids
def i2():
    # interp2d -- do it on a structured gird, but use calling
    # convention for unstructured grid.

    xx = linspace(0,2*pi,15)
    yy = linspace(-2,2,14)
    X,Y = make_grid(xx,yy)
    Z = ff(X,Y)

    # The output shapes are 
    # fint: ()   ()   => (1,)
    # fint: (n,) ()   => (n,)
    # fint: ()   (m,) => (1,m)
    # fint: (n,) (m,) => (n,m)
    # 
    # Linear interpolation is all messed up. Cant get sensible results.
    # Cubic seems extremely sensitive to the exact number of data points.
    # Doesn't respect bounds_error
    # Doesn't respect fill_value
    # Doesn't respect copy
    fint = scipy.interpolate.interp2d(X,Y,Z, kind='quintic')
    
    xfine = linspace(-2,2*pi+2,99)
    yfine = linspace(-4,4,101)
    XF, YF = make_grid(xfine, yfine)
    # NB -- TRANSPOSE HERE!
    ZF = fint(xfine,yfine).transpose()

    pl.clf()
    pl.pcolormesh(XF, YF, ZF)
    pl.colorbar()

def i3():
    # interp2d -- can't get sensible results out of this
    r, dr = 10, 4
    dth = dr/(1.0*r)
    
    rr = linspace(r,r+dr,15)
    th = linspace(-0.5*dth,0.5*dth,16)
    R,TH = make_grid(rr,th)
    X,Y = R*cos(TH),R*sin(TH)
    Z = ff(X,Y)

    # see comments in i2()
    fint = scipy.interpolate.interp2d(X,Y,Z, kind='cubic')
    
    xfine = linspace(r,r+dr,99)
    yfine = linspace(-0.5*dr,0.5*dr,101)
    XF, YF = make_grid(xfine, yfine)
    # NB -- TRANSPOSE HERE!
    ZF = fint(xfine,yfine).transpose()

    pl.clf()
    #pl.pcolormesh(X, Y, Z)
    pl.pcolormesh(XF, YF, ZF, vmin=-10, vmax=10)
    pl.colorbar()

def i4():
    # RectBivariateSpline(x, y, z, bbox = [None]*4, kx=3, ky=3, s=0):
    # Seems to work well, requires rectangular grid

    xx = linspace(0,2*pi,14)
    yy = linspace(-2,2,15)
    X,Y = make_grid(xx,yy)
    Z = ff(X,Y)

    # bbox doesn't generate an error, but looks to periodically wrap or something.
    fint = scipy.interpolate.RectBivariateSpline(xx,yy,Z, kx=5, ky=5)

    xfine = linspace(-2,2*pi+2,99)
    yfine = linspace(-4,4,101)
    XF, YF = make_grid(xfine, yfine)
    ZF = fint(xfine,yfine)

    pl.clf()
    pl.pcolormesh(XF, YF, ZF)
    pl.colorbar()

def i5():
    # griddata
    r, dr = 10, 4
    dth = dr/(1.0*r)
    
    rr = linspace(r,r+dr,15)
    th = linspace(-0.5*dth,0.5*dth,16)
    R,TH = make_grid(rr,th)
    X,Y = R*cos(TH),R*sin(TH)
    Z = ff(X,Y)
    points = grid_to_points(X,Y)
    values = grid_to_points(Z)
    
    xfine = linspace(r,r+dr,50)
    yfine = linspace(-0.5*dr,0.5*dr,51)
    XF, YF = make_grid(xfine, yfine)
    desired_points = grid_to_points(XF, YF)

    # Only 'nearest' seems to work
    # 'linear' and 'cubic' just give fill value
    desired_values = scipy.interpolate.griddata(points, values, desired_points, method='nearest', fill_value=1.0)
    ZF = desired_values.reshape(XF.shape)
    
    pl.clf()
    pl.pcolormesh(XF, YF, ZF)
    pl.colorbar()
    return ZF

def i6():
    # Rbf: this works great except that somewhat weird stuff happens
    # when you're off the grid.
    r, dr = 10, 4
    dth = dr/(1.0*r)
    
    rr = linspace(r,r+dr,25)
    th = linspace(-0.5*dth,0.5*dth,26)
    R,TH = make_grid(rr,th)
    X,Y = R*cos(TH),R*sin(TH)
    Z = ff(X,Y)
    xlist, ylist, zlist = X.ravel(), Y.ravel(), Z.ravel()
    
    # function, epsilon, smooth, norm, 
    fint = scipy.interpolate.Rbf(xlist, ylist, zlist )

    xfine = linspace(-10+r,r+dr+10,99)
    yfine = linspace(-10-0.5*dr,10+0.5*dr,101)
    XF, YF = make_grid(xfine, yfine)
    ZF = fint(XF, YF)
    
    pl.clf()
    pl.pcolormesh(XF, YF, ZF)
    pl.colorbar()
    return ZF

def i7():
    # SmoothBivariateSpline
    r, dr = 10, 4
    dth = dr/(1.0*r)
    
    rr = linspace(r,r+dr,11)
    th = linspace(-0.5*dth,0.5*dth,12)
    R,TH = make_grid(rr,th)
    X,Y = R*cos(TH),R*sin(TH)
    Z = ff(X,Y)
    xlist, ylist, zlist = X.ravel(), Y.ravel(), Z.ravel()
    
    # SmoothBivariateSpline(x, y, z, w=None, bbox = [None]*4, kx=3, ky=3, s=None,
    #              eps=None):
    fint = scipy.interpolate.SmoothBivariateSpline(xlist, ylist, zlist)

    xfine = linspace(r,r+dr,99)
    yfine = linspace(-0.5*dr,0.5*dr,101)
    XF, YF = make_grid(xfine, yfine)
    ZF = fint(xfine.ravel(), yfine.ravel()).reshape(XF.shape)
    
    pl.clf()
    pl.pcolormesh(XF, YF, ZF)
    pl.colorbar()


 
def make_grid(*axs):
    import operator
    shape = [len(ax) for ax in axs]
    ntot = reduce(operator.mul, shape)
    ntile = 1
    nrep = ntot
    result = []
    for ax in axs:
        nrep /= len(ax)
        mesh = tile(repeat(ax,nrep),ntile).reshape(shape)
        ntile *= len(ax)
        result.append(mesh)
    return result

