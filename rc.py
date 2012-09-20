# GSN Sept 18
#
# Implementation of analytic model of radiative/convective planet
# following Robinson + Catling ApJ 757:104 (2012)
#
# Models should absolutely be made into objects.

from numpy import *
import scipy.optimize, scipy.integrate
import pylab as pl

import structure

##############################
## Convective region

sigma_cgs = 5.67e-5
# Parameters
# tau tau0 sig0 t0 nn alpha f1_cgs k1 f2_cgs k2 fint_cgs gamma dd p0
# 
# outputs t_rc t0

def find_pressure(sig, tt_cgs):
    """Find pressure given entropy per baryon and temperature in cgs.
    Assume that you're talking about a monatomic ideal gas and that
    the contribution of electrons to the entropy is negligible.  This
    is sort of dumb but I want to make it easy to specify a class of
    models with constant entropy."""

    hbar_cgs = 1.05e-27
    kb_cgs = 1.38e-16
    mp_cgs = 1.67e-24
    nq_cgs = (mp_cgs*kb_cgs*tt_cgs/(2*pi*hbar_cgs**2))**1.5

    return kb_cgs*tt_cgs*nq_cgs*exp(2.5-sig)

def frad_up_conv(tau, tau0, t0_cgs, nn, alpha, gamma, dd):
    """Upward radiative flux in the convective region, RC eq 13"""
    nn, gamma = float(nn), float(gamma)

    # Gamma from their paper is defined thusly ito scipy functions
    def Gamma(a,x):
        return scipy.special.gamma(a)*scipy.special.gammaincc(a,x)
    beta = alpha*(gamma-1)/gamma
    ex = 4*beta/nn

    prefactor = sigma_cgs*t0_cgs**4
    gamfactor = exp(dd*tau)*(dd*tau0)**(-ex)
    expterm = exp(dd*(tau-tau0))
    gammadiff = (Gamma(1+ex, dd*tau) - Gamma(1+ex, dd*tau0))
    return prefactor*(expterm + gamfactor*gammadiff)
                                    
def frad_down_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha,
                   f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd):    
    """Downward radiative flux in the convective region, RC eq 14"""    
    # This is very likely messed up, and is in turn messing up the computation of the convective flux.  However, it's hard to see how it's so messed up since the soln satisfies the given flux constraints.  On the other hand, that includes the convective flux, which is computed to make the flux constraints correct.  So...  functions to compute convection, etc, depend in the correct way on this, which is messed up.  
    nn, tau0 = float(nn), float(tau0)
    
    if iterable(tau): return [frad_down_conv(the_tau, tau_rc, tau0, t0_cgs, nn, alpha,
                                             f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd)
                              for the_tau in tau]
    def integrand(xx):
        return (xx/tau0)**ex * exp(-dd*(tau-xx))

    beta = alpha*(gamma-1)/gamma
    ex = 4*beta/nn
    factor = dd*sigma_cgs*t0_cgs**4
    term1 = frad_down_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd)*exp(-dd*(tau-tau_rc))

    # do the whole integral every time.  Extremely dumb.  Fix this
    # later.  This is not needed and not typically interesting,
    # though, so don't worry about it for now.
    integ, err = scipy.integrate.quad(integrand, tau_rc, tau)

    return term1 + factor*integ

def frad_net_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd):
    return (frad_up_conv(tau, tau0, t0_cgs, nn, alpha, gamma, dd) -
            frad_down_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd))

def fconv_up_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd):
    """Upward convective flux in the convective region, RC eq 22"""        
    return (fint_cgs
            + fstar_net(tau, f1_cgs, k1, f2_cgs, k2)
            - frad_up_conv(tau, tau0, t0_cgs, nn, alpha, gamma, dd) 
            + frad_down_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd))

def temp_conv(tau, tau0, t0_cgs, nn, alpha, gamma):
    """Temp profile in convective region, RC eq 11"""
    nn, tau0, gamma = float(nn), float(tau0), float(gamma)
    
    beta = alpha*(gamma-1)/gamma
    ex = beta/nn
    return t0_cgs*(tau/tau0)**ex

def pressure_conv(tau, tau0, sig0, t0_cgs, nn, p0=None):
    """pressure profile in convective region, RC eq 6"""
    tau0, nn = float(tau0), float(nn)
    p0 = p0 or find_pressure(sig0, t0_cgs)        
    return p0*(tau/tau0)**(1/nn)

def fcheck_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd):
    """Sum of fluxes, should be zero..."""
    return (frad_net_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd) 
            + fconv_up_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, f1_cgs, k1, f2_cgs, k2, fint_cgs, gamma, dd)
            - fint_cgs
            - fstar_net(tau, f1_cgs, k1, f2_cgs, k2))

def fcheck_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd):
    """Sum of fluxes, should be zero..."""
    return (frad_net_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd) 
            - fint_cgs
            - fstar_net(tau, f1_cgs, k1, f2_cgs, k2))

    
##############################
## Radiative region

def tst(gg_cgs = 980):
    tau0 = 100
    nn = 1
    alpha = 1
    t1_cgs = 100
    k1 = 0
    t2_cgs = 0
    k2 = 0
    tint_cgs = 0
    gamma = 1.67
    dd = 1.5
    kappa_cgs = 0.2
    
    sig0=9.0

    f1_cgs, f2_cgs, fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]

    tau_rc, t0 = model(tau0=tau0, nn=nn, alpha=alpha, 
                       t1_cgs=t1_cgs, k1=k1, t2_cgs=t2_cgs, k2=k2, tint_cgs=tint_cgs, 
                       gamma=gamma, dd=dd)

    taus = logspace(log10(tau_rc), -3, 100)
    result = simple_pressure_rad(taus=taus, 
                 kappa_cgs=kappa_cgs, gg_cgs=gg_cgs, 
                 tau0=tau0, sig0=sig0, t0_cgs=t0, nn=nn, 
                 f1_cgs=f1_cgs, k1=k1, f2_cgs=f2_cgs, k2=k2, fint_cgs=fint_cgs, dd=dd)

    surface_gravity(tau_rc=tau_rc, 
                 kappa_cgs=kappa_cgs, 
                 tau0=tau0, sig0=sig0, t0_cgs=t0, nn=nn, 
                 f1_cgs=f1_cgs, k1=k1, f2_cgs=f2_cgs, k2=k2, fint_cgs=fint_cgs, dd=dd)
    
    return taus, result

def surface_gravity(tau_rc, 
                 kappa_cgs, 
                 tau0, sig0, t0_cgs, nn, 
                 f1_cgs, k1, f2_cgs, k2, fint_cgs, dd,
                 p0=None):
    """find surface gravity by requiring that p=0 at tau=0"""
    
    def ff(xx):
        result = simple_pressure_rad([tau_rc, 0], 
                            kappa_cgs, xx, 
                            tau0, sig0, t0_cgs, nn, 
                            f1_cgs, k1, f2_cgs, k2, fint_cgs, dd, p0=p0)
        print "GSN", xx, result[1]
        return result[1]

    gl, gh = 0.0, 1.0
    while ff(gh) > 0 and 2*gh != gh: gh *= 2
    return scipy.optimize.bisect(ff, gl, gh)
    
def simple_pressure_rad(taus, 
                 kappa_cgs, gg_cgs, 
                 tau0, sig0, t0_cgs, nn, 
                 f1_cgs, k1, f2_cgs, k2, fint_cgs, dd,
                 p0=None):
    """pressure profile in radiative region.  This is not explicitly
    computed in RC"""
    # I don't see how to get this.  I have T(tau), and defn of tau =
    # int rho kappa dz.  Kappa will depend on rho, temp, and therefore
    # implicitly tau.  I've introduced a new variable, z.  I can
    # combine this with HSE to get rid of z.  This gives dp / dtau =
    # -g / kappa, so basically p ~ tau.  But in more detail I have dp
    # / dtau = -g / kappa(m_p p / k T(tau), T(tau)).  So if I write
    # down kappa(rho, T) and I know T(tau) then I have a differential
    # equation to solve between p and tau.  So I guess that part is
    # ok.  But I don't know what I should adopt as a boundary
    # condition as p -> 0 and tau -> 0.  Can I do this from the RC
    # boundary instead?  Then I have a pressure from the convective
    # region and an optical depth.  So, ok, I think I've talked myself
    # through this.      
    #
    # So, trying to make this work:
    #
    # specify surface gravity gg_cgs, assume to be constant specify
    # opacity as a function of pressure and temperature.  solve
    # differential equation dp/dtau = g/kappa(p, t(tau)) with boundary
    # condition that p(tau_rc) = p_conv(tau_rc), going down into the region of smaller tau.
    #
    # The integration _must_ start at the radiative/convective
    # boundary, so take that to be the first entry in the desired
    # output points tau.
    #
    # Kramer's opacity law => kappa ~ p T^{-9/2}
    #
    # Good lord, every day you learn something.  THIS is where surface
    # gravity enters.  Solving for constant opacity gives simple
    # expression for p, which you can evaluate at tau=0 and don't
    # necessarily get zero.  That is, there will be positive (or
    # negative) pressure at the surface and the planet will expand or
    # contract to achieve zero pressure at the surface.  For const
    # opacity you get g = p_rc kappa / tau_rc
    # 
    # So the _right_ thing to do is iterate on this, choosing
    # different surface gravities until you get the correct solution,
    # p = 0 at tau = 0.  It's a 1d problem and behavior should be
    # monotonic so shouldn't be too hard to solve this.

    def derivs(yy, tau):
        pp = yy[0]
        trad = temp_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd, relaxed=True)
        result = [gg_cgs/kappa_cgs(pp, trad)]
        return result

    # if user specifies a constant kappa, turn it into a function that
    # returns something of the correct dimensionality.
    if not callable(kappa_cgs): 
        kappa_value = kappa_cgs
        kappa_cgs = lambda x,y: 0*x + 0*y + kappa_value

    # don't rely on the user to give a decreasing array of optical
    # depths, starting at the right value, etc.: give sensible answers
    # even if they mess up the input.

    flip = True if taus[0] < taus[-1] else False
    if flip: taus = taus[::-1]

    tau_rc = taus[0]
    p_rc = pressure_conv(tau_rc, tau0, sig0, t0_cgs, nn, p0=p0)    
    result = scipy.integrate.odeint(derivs, [p_rc], taus, hmax=0.1, mxstep=5000)
    return result

def temp_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd, relaxed=False):
    """Temp profile in radiative region, RC eq 18"""
    k1, k2, dd = float(k1), float(k2), float(dd)
    tau = asarray(tau)

    # take the limit as k->0 by hand
    kmin = 1e-3
    term1 = ((1 + dd*tau + k1/dd) if k1<kmin
             else 1+dd/k1 + (k1/dd - dd/k1)*exp(-k1*tau))
    term2 = ((1 + dd*tau + k2/dd) if k2<kmin
             else 1+dd/k2 + (k2/dd - dd/k2)*exp(-k2*tau))
    sigt4 = 0.5*(f1_cgs*term1 + f2_cgs*term2 + fint_cgs*(1+dd*tau))

    # Make this do something not crazy for negative tau so that it can
    # go inside an adaptive step integration routine and not flip out.
    #
    # might want this, which just flips the sign of the troublesome term.
    #if relaxed:
    #    return (abs(sigt4)/sigma_cgs)**0.25
    # or might want this, which just fixes negative taus to the value at zero
    # if relaxed and (tau < 0).any():
    #     val = temp_rad(0, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd, relaxed=False)
    #     result = (sigt4/sigma_cgs)**0.25
    #     result[tau<0] = val
    #     return result
    # or might want this, which ensure continuous function and first
    # derivative that goes to a constant positive value at neg tau
    if relaxed and (tau < 0).any():
        dtau = 1e-4
        ff = temp_rad(0, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd, relaxed=False)
        f1 = temp_rad(dtau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd, relaxed=False)
        fp = (f1-ff)/dtau
        frac = 0.8  # allowed drop before constant kicks in
        cc = fp/(ff*(1-frac))
        bb = fp/cc
        aa = ff-bb
        
        if len(tau.shape)==0: # scalar
            result = (sigt4/sigma_cgs)**0.25
            result = aa + bb*exp(cc*tau)
        else:
            result = (sigt4/sigma_cgs)**0.25
            result[tau<0] = aa + bb*exp(cc*tau[tau<0])

        return result
    
    return (sigt4/sigma_cgs)**0.25

def frad_up_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd):
    """Upward radiative flux in the radiative region, RC eq 19"""
    k1, k2 = float(k1), float(k2)
    # take the limit as k->0 by hand
    kmin=1e-3
    term1 = (2+(dd-k1)*tau if k1 < kmin
             else 1 + dd/k1 + (1-dd/k1)*exp(-k1*tau))
    term2 = (2+(dd-k2)*tau if k2 < kmin
             else 1 + dd/k2 + (1-dd/k2)*exp(-k2*tau))
    return 0.5*(f1_cgs*term1 + f2_cgs*term2 + fint_cgs*(2+dd*tau))

def frad_down_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd):
    """Downward radiative flux in the radiative region, RC eq 20"""
    k1, k2 = float(k1), float(k2)
    # take the limit as k->0 by hand
    kmin=1e-3
    term1 = ( (D + k1)*tau if k1 < kmin
              else 1 + dd/k1 - (1+dd/k1)*exp(-k1*tau))
    term2 = ( (D + k2)*tau if k2 < kmin
              else 1 + dd/k2 - (1+dd/k2)*exp(-k2*tau))
    return 0.5*(f1_cgs*term1 + f2_cgs*term2 + fint_cgs*dd*tau)

def frad_net_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd):
    return (frad_up_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd) - 
            frad_down_rad(tau, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))

                
##############################
## Apply everywhere

def fstar_net(tau, f1_cgs, k1, f2_cgs, k2):
    """Net absorbed stellar flux, RC eq 15"""
    return f1_cgs*exp(-k1*tau) + f2_cgs*exp(-k2*tau)

##############################
## The model

def model_try1(tau0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd):
    # if you specify tau0 and solve for t0, you can easily reduce the
    # problem to 1d root finding.  Requiring temp continuity at r/c
    # border gives sigma t0^4 = sigma t_rad^4(tau_rc)
    # (tau0/tau_rc)^(4beta/n).  Then I can solve for flux continuity
    # by dividing out by t0^4 (for whatever value I used) and
    # inserting the above expression.  The result only depends on
    # tau_rc, and it saves me from manipulating the expressions more
    # than I have to.  Flux continuity is then f_conv_up(tau_rc, t_0)
    # (tau_0/tau_rc)^(4beta/n) (t_rad^4 / t_0^4) - f_rad_up(tau_rc) = 0
    #
    # This doesn't work if you specify T0 b/c the dependence of
    # f_conv on t0 is more complicated.  It's still possible to do the
    # reduction of dimensionality but you have to break apart the
    # expressions for temp and flux.

    def ff(xx):
        cnt[0] += 1
        value = (frad_up_conv(xx, tau0, t0_dummy, nn, alpha, gamma, dd) *
                 (tau0/xx)**(4*beta/nn) *
                 (temp_rad(xx, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd) / t0_dummy)**4 -
                 frad_up_rad(xx, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
        return value/ftot
        
    # This value shouldn't matter
    t0_dummy = 100.0

    nn,gamma = float(nn), float(gamma)    
    verbose = False
    cnt = [0] 
    f1_cgs, f2_cgs, fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]
    ftot = f1_cgs + f2_cgs + fint_cgs
    beta = alpha*(gamma-1)/gamma

    # per DSP's complaint, try to make this bulletproof (but don't
    # allow an infinite loop)
    taul, tauh = 1.0, 1.0
    while ff(taul) < 0 and 2*taul != taul: taul /= 2.0
    while ff(tauh) > 0 and 2*tauh != tauh: tauh *= 2.0
    if verbose: print "Starting at", taul, tauh, ff(taul), ff(tauh)
    tau_rc = scipy.optimize.bisect(ff, taul, tauh)
    if verbose: print "Ending at", tau_rc, ff(tau_rc), cnt[0], "iterations"

    # calculate t0_cgs
    t0_cgs = temp_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd)*(tau0/tau_rc)**(beta/nn)
    return [tau_rc, t0_cgs]

def model(tau0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd):
    """Solve for temperature and flux continuity at the
    radiative-convective boundary.  This is where the big money is.

    For planets with surfaces, you may want to fix the reference
    temperature and find the optical depth.  For planets without
    surfaces I think it makes more sense to fix the reference optical
    depth and find the temperature there.

    Follow suggestions from DSP and specify fluxes via effective temps."""

    def t0_taurc(xx):
        return temp_rad(xx, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd)*(tau0/xx)**(beta/nn)

    def ff(xx):
        cnt[0] += 1
        t0_cgs = t0_taurc(xx)
        value =  (frad_up_conv(xx, tau0, t0_cgs, nn, alpha, gamma, dd)
                  - frad_up_rad(xx, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
        return value/ftot
        
    nn,gamma = float(nn), float(gamma)    
    verbose = False    
    cnt = [0] 
    f1_cgs, f2_cgs, fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]
    ftot = f1_cgs + f2_cgs + fint_cgs
    beta = alpha*(gamma-1)/gamma

    # per DSP's complaint, try to make this bulletproof (but don't
    # allow an infinite loop)
    taul, tauh = 1.0, 1.0
    while ff(taul) < 0 and 2*taul != taul: taul /= 2.0
    while ff(tauh) > 0 and 2*tauh != tauh: tauh *= 2.0
    if verbose: print "Starting at", taul, tauh, ff(taul), ff(tauh)
    tau_rc = scipy.optimize.bisect(ff, taul, tauh)
    if verbose: print "Ending at", tau_rc, ff(tau_rc), cnt[0], "iterations"

    # calculate t0_cgs
    t0_cgs = temp_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd)*(tau0/tau_rc)**(beta/nn)
    return [tau_rc, t0_cgs]

def rc_model(t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd):
    """Solve for temperature and flux continuity at the
    radiative-convective boundary.  This is where the big money is.

    Follow RC and make T0 a model parameter, then solve for tau_rc and
    tau_0.  This is to facilitate comparison with their plots.
    
    Follow suggestions from DSP and specify fluxes via effective temps."""

    def tau0_taurc(xx):
        return xx*(t0_cgs/temp_rad(xx, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))**(nn/beta)

    def ff(xx):
        cnt[0] += 1
        tau0 = tau0_taurc(xx)
        value =  (frad_up_conv(xx, tau0, t0_cgs, nn, alpha, gamma, dd)
                  - frad_up_rad(xx, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
        return value/ftot

    verbose = False    
    cnt = [0] 
    f1_cgs, f2_cgs, fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]
    ftot = f1_cgs + f2_cgs + fint_cgs
    beta = alpha*(gamma-1)/gamma

    # per DSP's complaint, try to make this bulletproof (but don't
    # allow an infinite loop)
    taul, tauh = 1.0, 1.0
    while ff(taul) < 0 and 2*taul != taul: taul /= 2.0
    while ff(tauh) > 0 and 2*tauh != tauh: tauh *= 2.0
    if verbose: print "Starting at", taul, tauh, ff(taul), ff(tauh)
    tau_rc = scipy.optimize.bisect(ff, taul, tauh)
    if verbose: print "Ending at", tau_rc, ff(tau_rc), cnt[0], "iterations"

    return [tau_rc, tau0_taurc(tau_rc)]
    
# # one model that converges:
# # model(tau0=50, nn=1, alpha=1, f1_cgs=10, k1=1, f2_cgs=0, k2=1, fint_cgs=10, gamma=1.67, dd=1.5)
# # model(tau0=50, nn=1, alpha=1, t1_cgs=20, k1=1, t2_cgs=0, k2=1, tint_cgs=20, gamma=1.67, dd=1.5)
# def model_2d(tau0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd):
#     """Solve for temperature and flux continuity at the
#     radiative-convective boundary.  This is where the big money is.

#     For planets with surfaces, you may want to fix the reference
#     temperature and find the optical depth.  For planets without
#     surfaces I think it makes more sense to fix the reference optical
#     depth and find the temperature there.

#     Follow suggestions from DSP and specify fluxes via effective temps."""
#     def ff(xx):
#         cnt[0] += 1
#         tau_rc, t0_cgs = xx
#         tdiff = (temp_conv(tau_rc, tau0, t0_cgs, nn, alpha, gamma) -
#                  temp_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
#         frad_diff = (frad_up_conv(tau_rc, tau0, t0_cgs, nn, alpha, gamma, dd) -
#                      frad_up_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
#         return [tdiff/tscale, frad_diff/ftot]

#     verbose=True
#     gamma = float(gamma)
#     cnt = [0] 

#     f1_cgs, f2_cgs, fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]
#     ftot = f1_cgs + f2_cgs + fint_cgs
#     tscale = (ftot/sigma_cgs)**0.25

#     # Should we set T_0 using all of the flux or only the internal flux?
#     beta = alpha*(gamma-1)/gamma
#     x0 = [1.0, 100]
#     x0 = [1.0, ( ftot/sigma_cgs)**0.25 * tau0**(beta/nn)]
#     #x0 = [1.0, (fint_cgs/sigma_cgs)**0.25 * tau0**(beta/nn)]
#     if verbose: print "Starting at", x0, ff(x0)
#     result = scipy.optimize.newton_krylov(ff, x0, f_tol=0.05)
#     if verbose: print "Ending at", result, ff(result), cnt[0], "iterations"
#     return result 

# def rc_model_2d(t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd):
#     """Solve for temperature and flux continuity at the
#     radiative-convective boundary.  This is where the big money is.

#     Follow RC and make T0 a model parameter, then solve for tau_rc and
#     tau_0.  This is to facilitate comparison with their plots.

#     It also seems at first glance that this is _much_ easier to get to
#     converge.
    
#     Follow suggestions from DSP and specify fluxes via effective temps."""

#     def ff(xx):
#         cnt[0] += 1
#         tau_rc, tau0 = xx
#         tdiff = (temp_conv(tau_rc, tau0, t0_cgs, nn, alpha, gamma) -
#                  temp_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
#         frad_diff = (frad_up_conv(tau_rc, tau0, t0_cgs, nn, alpha, gamma, dd) -
#                      frad_up_rad(tau_rc, f1_cgs, k1, f2_cgs, k2, fint_cgs, dd))
#         return [tdiff/tscale, frad_diff/ftot]

#     verbose=True
#     gamma = float(gamma)
#     cnt = [0]

#     f1_cgs, f2_cgs, fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]
#     ftot = f1_cgs + f2_cgs + fint_cgs
#     tscale = (ftot/sigma_cgs)**0.25

#     beta = alpha*(gamma-1)/gamma
#     x0 = [1.0, 1.0]

#     if verbose: print "Starting at", x0, ff(x0)
#     result = scipy.optimize.newton_krylov(ff, x0, f_tol=0.05)
#     if verbose: print "Ending at", result, ff(result), cnt[0], "iterations"
#     return result 

# make sure tau doesn't go larger than tau0
# tau=logspace(-2,1.6,100) 
# model(tau0=50, nn=1, alpha=1, f1_cgs=10, k1=1, f2_cgs=0, k2=1, fint_cgs=10, gamma=1.67, dd=1.5)
# plots(tau, tau0=50.0, sig0=9.0, nn=1.0, alpha=1.0, f1_cgs=10.0, k1=1.0, f2_cgs=0.0, k2=1.0, fint_cgs=10.0, gamma=1.67, dd=1.5, p0=None)
def plots(tau, tau0, sig0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd, p0=None, t0_cgs=None):

    # if t0_cgs is specified, use it and solve for tau0 (ignoring the
    # value given).  If not, use tau0 and solve for t0_cgs.  This
    # allows you to specify the model differently and the rest of the
    # plotting stuff still works.  In principle maybe I shouldn't be
    # passing all these parameters around, but should define a "model"
    # in a more generic fashion which is then passed to the plotting
    # routine.
    if t0_cgs:
        tau_rc, tau0 = rc_model(t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd)
    else:
        tau_rc, t0_cgs = model(tau0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd)

    print tau_rc, t0_cgs, tau0
    #tau_rc, t0_cgs = 4.36210102405, 78.4106542356
    print tau_rc, t0_cgs, tau0

    pl.clf()

    def flux_marks():
        # mark rad/conv boundary
        pl.semilogy([-100, 100], [tau_rc, tau_rc], 'k')    
        # mark zero
        pl.semilogy([0, 0], [tau[0], tau[-1]], 'k')
        pl.xlim(-50,100)

    def temp_marks():
        # mark rad/conv boundary
        pl.loglog([1, 100], [tau_rc, tau_rc], 'k')    

    def flux_to_temp(xx):
        xx = asarray(xx)
        return sign(xx)*(abs(xx)/sigma_cgs)**0.25
    
    pl.subplot(2,2,1)
    pl.semilogy(flux_to_temp(frad_up_rad(tau, t1_cgs, k1, t2_cgs, k2, tint_cgs, dd)), tau, 'b:')
    pl.semilogy(flux_to_temp(frad_down_rad(tau, t1_cgs, k1, t2_cgs, k2, tint_cgs, dd)), tau, 'b--')
    pl.semilogy(flux_to_temp(frad_net_rad(tau, t1_cgs, k1, t2_cgs, k2, tint_cgs, dd)), tau, 'b')
    pl.ylim(tau[-1], tau[0])
    pl.legend(('up', 'down', 'net'), loc='upper left')
    pl.title('Rad teff in rad zone')
    pl.semilogy(flux_to_temp(fstar_net(tau, t1_cgs, k1, t2_cgs, k2)), tau, 'g')
    flux_marks()

    pl.subplot(2,2,2)
    pl.semilogy(flux_to_temp(frad_up_conv(tau, tau0, t0_cgs, nn, alpha, gamma, dd)), tau, 'r:')
    pl.semilogy(flux_to_temp(frad_down_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2,
                                   tint_cgs, gamma, dd)), tau, 'r--')
    pl.semilogy(flux_to_temp(frad_net_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2,
                                   tint_cgs, gamma, dd)), tau, 'r')
    pl.ylim(tau[-1], tau[0])
    pl.legend(('up', 'down', 'net'), loc='upper left')
    pl.title('Rad teff in conv zone')
    pl.semilogy(flux_to_temp(fstar_net(tau, t1_cgs, k1, t2_cgs, k2)), tau, 'g')
    flux_marks()
    
    pl.subplot(2,2,3)
    pl.title('Net teff')
    pl.semilogy(flux_to_temp(fstar_net(tau, t1_cgs, k1, t2_cgs, k2)), tau, 'g')
    pl.semilogy(flux_to_temp(frad_net_rad(tau, t1_cgs, k1, t2_cgs, k2, tint_cgs, dd)), tau, 'b')
    pl.semilogy(flux_to_temp(frad_net_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2,
                                   tint_cgs, gamma, dd)), tau, 'r')
    pl.semilogy(flux_to_temp(fconv_up_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, t1_cgs, k1,
                                   t2_cgs, k2, tint_cgs, gamma, dd)), tau, 'm')
    pl.legend(('Star', 'Rad (rad)', 'Rad (conv)', 'Conv'), loc='upper left')
    pl.ylim(tau[-1], tau[0])
    flux_marks()

    pl.subplot(2,2,4)
    pl.loglog(temp_rad(tau, t1_cgs, k1, t2_cgs, k2, tint_cgs, dd), tau, 'b')
    pl.loglog(temp_conv(tau, tau0, t0_cgs, nn, alpha, gamma), tau, 'r')
    pl.legend(('Rad', 'Conv'), loc='lower left')
    pl.title('Temperature')
    pl.ylim(tau[-1], tau[0])
    temp_marks()

    pl.draw()

def plots_check(tau, tau0, sig0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd, p0=None):    
    """Plot total fluxes.  Should be zero..."""
    tau_rc, t0_cgs = model(tau0, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd)
    pl.clf()
    pl.semilogx(tau, fcheck_rad(tau, t1_cgs, k1, t2_cgs, k2, tint_cgs, dd), 'b')
    pl.semilogx(tau, fcheck_conv(tau, tau_rc, tau0, t0_cgs, nn, alpha, t1_cgs, k1, t2_cgs, k2, tint_cgs, gamma, dd), 'r')
    pl.semilogx([tau_rc, tau_rc], [-20, 20], 'k')    
    pl.legend(('rad zone','conv zone'))

def fig1():
    # Their fig 1.  I have no doubt that they plotted the function
    # correctly, but try to get it from the model directly.
    #
    # no attenuation, k1 = k2 = 0; 
    #
    # - you can put the flux into channel 1, channel 2, or internal
    # channel, it should all be the same;  Verified
    # 
    # - tau_rc should be independent of absolute value of flux; verified
    # 
    # - all taus are mutiplied by d, so should be able to change it
    # and only scale taus; verified
    # 
    # - should depend on alpha, gamma, n only through combination 4
    # alpha(gamma-1)/n gamma. verified

    fbons = linspace(0.2, 1.0, 30)
    tau0s = logspace(-2,1,30)

    # these shouldn't matter
    gamma=1.67
    nn = 3.0
    teff = 1000.0
    dd=1.0

    # edges of pixels
    X,Y = structure.make_grid(fbons, tau0s)

    result = [] 
    for fbon in fbons:
        row = [] 
        for tau0 in tau0s:
            alpha = fbon*nn*gamma/(4.0*(gamma-1))
            trc, t0_cgs = model(tau0, nn=nn, alpha=alpha, t1_cgs=teff, k1=0,
                                t2_cgs=0, k2=0, tint_cgs=0, gamma=gamma, dd=dd)
            row.append(trc)
        result.append(row)

    levels = array([0.01, .1, 0.5, 1.0, 2.0])/dd
    pl.contour(X,dd*Y,array(result), levels)
    pl.gca().set_yscale('log')
    pl.ylim(10,0.01)
    pl.draw()

def fig2():
    # Says that they use "a range of values of t0" but that for large
    # values it's independent of everything but 4beta/n.  So I guess
    # they picked a large value to plot?

    fbons = linspace(0.2, 1.0, 50)

    # these shouldn't matter
    gamma=1.67
    nn = 3.0
    teff = 1000.0
    dd=1.0
    tau0 = 100

    result = [] 
    for fbon in fbons:
        alpha = fbon*nn*gamma/(4.0*(gamma-1))
        trc, t0_cgs = model(tau0, nn=nn, alpha=alpha, t1_cgs=teff, k1=0,
                            t2_cgs=0, k2=0, tint_cgs=0, gamma=gamma, dd=dd)
        result.append(trc)
    result = asarray(result)

    pl.semilogy(fbons, dd*result)
    pl.ylim(20, 0.01)
    pl.draw()

def fig10():
    f1_mks = 1.3
    f2_mks = 7.0
    fi_mks = 5.0
    t1_cgs = ((1e3*f1_mks/sigma_cgs))**0.25
    t2_cgs = ((1e3*f2_mks/sigma_cgs))**0.25
    ti_cgs = ((1e3*fi_mks/sigma_cgs))**0.25
    print t1_cgs, t2_cgs, ti_cgs

    # params
    alpha = 0.85
    k1 = 100
    k2 = 0.06
    gamma = 7/5.0
    # can't find where they specify nn
    dd = 1.5
    nn=1.0
    tau0 = 1e5
    tau_rc, t0_cgs = model(tau0, nn=nn, alpha=alpha, t1_cgs=t1_cgs, k1=k1,
                           t2_cgs=t2_cgs, k2=k2, tint_cgs=ti_cgs, gamma=gamma, dd=dd)


    print tau_rc
    pl.clf()
    tau = logspace(-3,5,100)
    pl.semilogy(temp_rad(tau, t1_cgs, k1, t2_cgs, k2, ti_cgs, dd), tau, 'b')
    pl.semilogy(temp_conv(tau, tau0, t0_cgs, nn, alpha, gamma), tau, 'r')
    pl.legend(('Rad', 'Conv'), loc='lower left')
    pl.title('Temperature')
    pl.ylim(tau[-1], tau[0])
    pl.xlim(10,200)
    pl.plot([10, 1000], [tau_rc, tau_rc], 'k')    
        
##############################
## Utility stuff

def ave(a, n=1, axis=-1):
    """average n adjacent points of array a along axis axis
    ave([1,2,5], 1) = [1.5, 3.5] """
    
    if n==0: return array(a)
    if n < 0: raise ValueError, 'order must be non-negative but got ' + repr(n)
    a = asarray(a)
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

def lave(a, n=1, axis=-1):    
    """As ave(), but do the averaging in log space (so that you get
    geometric means instead of arithmetic means)"""
    return exp(ave(log(a), n=n, axis=axis))

##############################
## playing around
def factorial(nn):
    assert nn >= 0
    result = 1
    for ii in range(1,nn+1):
        result *= ii
    return result

def gamma_approx_plot():
    pl.clf()
    xs = logspace(-1,1,100)
    aas = (1,2,3,4,5)
    cs = ('b', 'r', 'g', 'c', 'm')
    for aa,cc in zip(aas, cs):
        # actual function
        pl.loglog(xs, scipy.special.gamma(aa)*scipy.special.gammaincc(aa,xs), c=cc)
        # small x
        pl.loglog(xs, 0*xs + factorial(aa-1), c=cc, ls=':')
        #pl.loglog(xs, 0*xs + exp((aa-1)*(log(aa-1)-1)), c=cc, ls='-.')
        # large x
        pl.loglog(xs, xs**(aa-1)*exp(-xs), c=cc, ls='--')
    pl.ylim(1e-3,None)
    pl.draw()
