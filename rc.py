# GSN Sept 18
#
# Implementation of analytic model of radiative/convective planet
# following Robinson + Catling ApJ 757:104 (2012)
#
# I've extended this with functions that solve for the pressure in the
# radiative region.  This is evidently done in the paper but not
# discussed, evidently because it's bread and butter for atmospheres
# people.  The case where the opacity is separable into a function of
# pressure and a function of temperature is treated by numerical
# integration (no iteration to match a surface pressure=0 boundary
# condition) so it's fast.  General opacities are allowed, but be
# prepared to wait if you want a lot of models.
#
# Can't reproduce their fig 4: I get the same log derivs indep of n
# Many of the figures look pretty good but are not the same
# quantitatively.  Not sure what's going on with this.
#
# The procedure for solving for continuity at the rad/conv boundary,
# etc, is different depending on what you specify, so the different
# possibilities are split into different objects.  Code common to all
# possibilities (e.g. the RC flux formulae) are the "roots" of the
# inheritence tree.  "User interface" objects are the leaves.
# 
# Planet: Contains the RC formulae for fluxes, temps, etc
#   PlanetFromFlux : Specify fluxes, solve for surf. grav
#     PlanetFromFluxTau : Spec. tau0, solve for T0
#     PlanetFromFluxTemp : Spec. T0, solve for tau0
#   PlanetFromGrav : Specify surf. grav, tau0, solve for Tint, T0
#
# To specify a model, _all_ parameters should be passed as keyword
# args.  These are the parameters:
# 
#   Parameters defined by RC, must be scalar floating point values: 
#   k1, k2, p0_cgs, nn, alpha, gamma, dd
#   
#   Parameters defined by RC, you specify one and solve for the other
#   t0_cgs, tau0
# 
#   Parameters defined by RC, always solved for:
#   tau_rc
# 
#   Specify RC fluxes via temperature via f = sigma T^4: 
#   tint_cgs, t1_cgs, t2_cgs
# 
#   You may specify the entropy at the reference point instead of p0_cgs
#   It's entropy per baryon in dimensionless form:
#   sig0
#
#   Precision with which to find roots when solving for continuity:
#   dtau0, dt0, dgg, 
#
#   Opacity, needed to solve for pressure in the radiative region:
#   kappa_cgs
#     May be specified as:
#     1) a constant
#     2) a sequence of two functions A(), B() such that kappa = A(p)*B(T(tau))
#     3) a function of p and T
#   
#   Whether or not to try to solve for surf. grav:
#   gravity = True/False  (can be time consuming)
#
# Flat list of parameters (for reminding myself)
#   gg_cgs, k1, k2, p0_cgs, nn, alpha, gamma, dd t0_cgs, tau0 tau_rc tint_cgs, t1_cgs, t2_cgs sig0 dtau0, dt0, dgg, kappa_cgs,  gravity, tau_rc
#
# Some example models
### Venus 
# PlanetFromFluxTemp(nn=1, tint_cgs=0, t1_cgs=(160*1e3/sigma_cgs)**0.25, k1=0, t0_cgs=730, p0_cgs=92*1e6, alpha=0.8, gamma=1.3, dd=1.5, kappa_cgs=0.2, gravity=True)
# PlanetFromFluxTemp(nn=2, tint_cgs=0, t1_cgs=(160*1e3/sigma_cgs)**0.25, k1=0, t0_cgs=730, p0_cgs=92*1e6, alpha=0.8, gamma=1.3, dd=1.5, kappa_cgs=0.2, gravity=True)
#
### Titan 
# PlanetFromFluxTemp(tint_cgs=0, t1_cgs=72, k1=120, t2_cgs=66, k2=0.2, t0_cgs=94, p0_cgs=1.5*1e6, nn=1.33, alpha=0.77, gamma=1.4, dd=1.5, kappa_cgs=0.2, gravity=True)
#
### Jupiter 
# PlanetFromFluxTemp(t1_cgs=69, k1=100, t2_cgs=105, k2=0.06, t0_cgs=191, tint_cgs=99, p0_cgs=1.1*1e6, nn=2, alpha=0.85, gamma=1.4, dd=1.5, kappa_cgs=0.2, gravity=True)
#
### GSN Favorite model
# PlanetFromFluxTau(t1_cgs=150, k1=100, t2_cgs=105, k2=0.06, tau0=1000, tint_cgs=150, p0_cgs=1.1*1e6, nn=2, alpha=0.85, gamma=1.4, dd=1.5, kappa_cgs=0.2, gravity=True)
### GSN Favorite, specifying entropy
# PlanetFromFluxTau(t1_cgs=150, k1=100, t2_cgs=105, k2=0.06, tau0=1000, tint_cgs=150, sig0=13.75, nn=2, alpha=0.85, gamma=1.4, dd=1.5, kappa_cgs=0.2, gravity=True)


from numpy import *
import scipy.optimize, scipy.integrate
import pylab as pl

import structure

sigma_cgs = 5.67e-5
G_cgs = 6.67e-8  # Only enters into calculating the luminosity

def test():
    # quick test to exercise all of the code paths
    # spec tau0, sigma, 
        
    mm = PlanetFromFluxTau(tau0=1000, sig0=9, nn=1, alpha=1,
                t1_cgs=75, k1=1, t2_cgs=0, k2=0, tint_cgs=75,
                gamma=1.67, dd=1.5, kappa_cgs=0.2, gravity=True)
    # spec  tau0, p0
    mm = PlanetFromFluxTau(tau0=1000, sig0=None, p0_cgs=1e2*1e6, nn=1, alpha=1,
                t1_cgs=75, k1=1, t2_cgs=0, k2=0, tint_cgs=75,
                gamma=1.67, dd=1.5, kappa_cgs=0.2, gravity=True)

    # spec  t0, sigma
    mm = PlanetFromFluxTemp(t0_cgs=1000, sig0=9, nn=1, alpha=1,
               t1_cgs=75, k1=1, t2_cgs=0, k2=0, tint_cgs=75,
               gamma=1.67, dd=1.5, kappa_cgs=0.2, gravity=True)

    # spec  t0, p0
    mm = PlanetFromFluxTemp(t0_cgs=1000, sig0=None, p0_cgs=1e2*1e8,nn=1, alpha=1,
               t1_cgs=75, k1=1, t2_cgs=0, k2=0, tint_cgs=75,
               gamma=1.67, dd=1.5, kappa_cgs=0.2, gravity=True)

    # these are not used in solving for a model, just check them to make sure they run
    mm.frad_down_conv(1.1)

    mm = PlanetFromGravDirect(gg_cgs = 58600, tau0=100, sig0=9, nn=1, alpha=1,
               t1_cgs=75, k1=1, t2_cgs=0, k2=0, 
               gamma=1.67, dd=1.5, kappa_cgs=0.2)

def all_figs():
    # This is just to exercise code
    fig1()
    fig2()
    fig3()
    fig4()
    fig5(1.5)
    fig6()
    fig7()
    fig8_9()
    fig10_11()
    fig12_13()

class Planet:
    """Just contains the expressions for fluxes, etc, in RC.  Doesn't
    solve for tau_rc, tau0, t0, or gg"""

    # finding the surface gravity takes a long time and isn't always necessary, so skip it by default.
    def __init__(self, tint_cgs=0,      # Probably need some value for this
                 t1_cgs=0, k1=0, t2_cgs=0, k2=0,  # Might need some of these
                 nn=1, alpha=1, gamma=1.66, dd=1.5, # These have good defaults
                 kappa_cgs=None):  

        # fill in values
        self.nn = float(nn)
        self.alpha = alpha
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.gamma = float(gamma)
        self.dd = float(dd)

        # Don't store these in the object, only store fluxes, to avoid redundant info.
        #self.t1_cgs = t1_cgs
        #self.t2_cgs = t2_cgs
        #self.tint_cgs = tint_cgs

        # When the opacity is a separable function of pressure and
        # temperature, you can find the surface gravity analtyically.
        # Allow this as a special case.  This maybe should go under
        # PlanetFlux b/c you don't care about this if the surface
        # gravity is specified.
        if iterable(kappa_cgs):
            self._kappa_separate_p = kappa_cgs[0]
            self._kappa_separate_t = kappa_cgs[1]
            self.kappa = lambda x,y: kappa_cgs[0](x)*kappa_cgs[1](y)
        elif not callable(kappa_cgs): 
            # allow just constants, too, pressed into separable form.
            self._kappa_separate_p = lambda x: 0*asarray(x) + kappa_cgs
            self._kappa_separate_t = lambda x: 0*asarray(x) + 1.0
            self.kappa = lambda x,y: self._kappa_separate_p(x)*self._kappa_separate_t(y)
        else: 
            self.kappa = kappa_cgs

        # simple derived quantities
        self.f1_cgs, self.f2_cgs, self.fint_cgs = [sigma_cgs*tt**4 for tt in t1_cgs, t2_cgs, tint_cgs]
        self.beta = alpha*(gamma-1)/gamma

        # Not so important parameters
        self.kmin = 1e-3
        self.verbose = True

##############################
## Convective region

    def frad_up_conv(s, tau, t0=None, tau0=None):
        """Upward radiative flux in the convective region, RC eq 13"""

        # Gamma from their paper is defined thusly ito scipy functions
        def Gamma(a,x):
            return scipy.special.gamma(a)*scipy.special.gammaincc(a,x)

        # Allow temp to be an argument in order to carry out solution
        # for tau_rc.  If it's not specified, use the value in the
        # object.
        if t0 is None: t0 = s.t0_cgs
        if tau0 is None: tau0 = s.tau0

        ex = 4*s.beta/s.nn

        prefactor = sigma_cgs*t0**4
        gamfactor = exp(s.dd*tau)*(s.dd*tau0)**(-ex)
        expterm = exp(s.dd*(tau-tau0))
        gammadiff = (Gamma(1+ex, s.dd*tau) - Gamma(1+ex, s.dd*tau0))
        return prefactor*(expterm + gamfactor*gammadiff)

    def frad_down_conv(s, tau):    
        """Downward radiative flux in the convective region, RC eq 14"""    
        # This is very likely messed up, and is in turn messing up the
        # computation of the convective flux.  However, it's hard to
        # see how it's so messed up since the soln satisfies the given
        # flux constraints.  On the other hand, that includes the
        # convective flux, which is computed to make the flux
        # constraints correct.  So...  functions to compute
        # convection, etc, depend in the correct way on this, which is
        # messed up.

        if iterable(tau): return array([s.frad_down_conv(the_tau)
                                  for the_tau in tau])
        def integrand(xx):
            return (xx/s.tau0)**ex * exp(-s.dd*(tau-xx))
        
        ex = 4*s.beta/s.nn
        factor = s.dd*sigma_cgs*s.t0_cgs**4
        term1 = s.frad_down_rad(s.tau_rc)*exp(-s.dd*(tau-s.tau_rc))

        # do the whole integral every time.  Extremely dumb.  Fix this
        # later.  This is not needed and not typically interesting,
        # though, so don't worry about it for now.
        integ, err = scipy.integrate.quad(integrand, s.tau_rc, tau)

        return term1 + factor*integ

    def frad_net_conv(s, tau):
        return (s.frad_up_conv(tau) - s.frad_down_conv(tau))

    def fconv_up_conv(s, tau):
        """Upward convective flux in the convective region, RC eq 22"""        
        return (s.fint_cgs + s.fstar_net(tau) 
                - s.frad_up_conv(tau) + s.frad_down_conv(tau))

    def temp_conv(s, tau):
        """Temp profile in convective region, RC eq 11"""

        ex = s.beta/s.nn
        return s.t0_cgs*(tau/s.tau0)**ex

    def pressure_conv(s, tau):
        """pressure profile in convective region, RC eq 6"""
        return s.p0_cgs*(tau/s.tau0)**(1/s.nn)

    def fcheck_conv(s, tau):
        """Sum of fluxes, should be zero..."""
        return (s.frad_net_conv(tau) + s.fconv_up_conv(tau) 
                - s.fint_cgs - s.fstar_net(tau))

    def fcheck_rad(s, tau):
        """Sum of fluxes, should be zero..."""
        return s.frad_net_rad(tau) - s.fint_cgs - s.fstar_net(tau)

##############################
## Radiative region

    def temp_rad(s, tau, relaxed=False):
        """Temp profile in radiative region, RC eq 18"""

        tau = asarray(tau)

        # take the limit as k->0 by hand
        term1 = ((1 + s.dd*tau + s.k1/s.dd) if s.k1 < s.kmin
                 else 1+s.dd/s.k1 + (s.k1/s.dd - s.dd/s.k1)*exp(-s.k1*tau))
        term2 = ((1 + s.dd*tau + s.k2/s.dd) if s.k2 < s.kmin
                 else 1+s.dd/s.k2 + (s.k2/s.dd - s.dd/s.k2)*exp(-s.k2*tau))
        sigt4 = 0.5*(s.f1_cgs*term1 + s.f2_cgs*term2 + s.fint_cgs*(1+s.dd*tau))

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
            frac = 0.8  # allowed drop before constant kicks in
            ff = s.temp_rad(0, relaxed=False)
            f1 = s.temp_rad(dtau, relaxed=False)
            fp = (f1-ff)/dtau
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

    def frad_up_rad(s, tau):
        """Upward radiative flux in the radiative region, RC eq 19"""

        # take the limit as k->0 by hand
        term1 = (2+(s.dd-s.k1)*tau if s.k1 < s.kmin
                 else 1 + s.dd/s.k1 + (1-s.dd/s.k1)*exp(-s.k1*tau))
        term2 = (2+(s.dd-s.k2)*tau if s.k2 < s.kmin
                 else 1 + s.dd/s.k2 + (1-s.dd/s.k2)*exp(-s.k2*tau))
        return 0.5*(s.f1_cgs*term1 + s.f2_cgs*term2 + s.fint_cgs*(2+s.dd*tau))

    def frad_down_rad(s, tau):
        """Downward radiative flux in the radiative region, RC eq 20"""

        # take the limit as k->0 by hand
        term1 = ( (s.dd + s.k1)*tau if s.k1 < s.kmin
                  else 1 + s.dd/s.k1 - (1+s.dd/s.k1)*exp(-s.k1*tau))
        term2 = ( (s.dd + s.k2)*tau if s.k2 < s.kmin
                  else 1 + s.dd/s.k2 - (1+s.dd/s.k2)*exp(-s.k2*tau))
        return 0.5*(s.f1_cgs*term1 + s.f2_cgs*term2 + s.fint_cgs*s.dd*tau)

    def frad_net_rad(s, tau):
        return (s.frad_up_rad(tau) - s.frad_down_rad(tau))

    def pressure_rad(s, tau):
        return s._simple_pressure_rad(tau, s.gg_cgs)

    def pressure_rad_hypothetical(s, tau):
        """Forget about the rad/conv boundary and give what the
        radiative pressure would be if there were no convection."""
        # this requires splitting the input array in two b/c of the
        # requirement that the integration start at the rad/conv
        # boundary.
        trad = concatenate(([s.tau_rc], tau[tau<=s.tau_rc][::-1]))
        tconv = concatenate(([s.tau_rc], tau[tau>s.tau_rc]))
        prad = s._simple_pressure_rad(trad, s.gg_cgs)[1:][::-1]
        pconv = s._simple_pressure_rad(tconv, s.gg_cgs)[1:]
        pp = concatenate((prad, pconv))
        return pp[:,0]
    
    def _simple_pressure_rad(s, taus, gg_cgs):
        """pressure profile in radiative region.  This is not explicitly
        computed in RC.  This assumes you already know the surface gravity"""
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
            trad = s.temp_rad(tau, relaxed=True)
            result = [gg_cgs/s.kappa(pp, trad)]
            return result

        # don't rely on the user to give a decreasing array of optical
        # depths, starting at the right value, etc.: give sensible answers
        # even if they mess up the input.
        # flip = True if taus[0] < taus[-1] else False
        # if flip: taus = taus[::-1]
        
        assert abs(taus[0]/s.tau_rc - 1) < 1e-4

        p_rc = s.pressure_conv(s.tau_rc)[0]
        return scipy.integrate.odeint(derivs, [p_rc], taus, hmax=0.1, mxstep=5000)
                
##############################
## Apply everywhere

    def fstar_net(s, tau):
        """Net absorbed stellar flux, RC eq 15"""
        return s.f1_cgs*exp(-s.k1*tau) + s.f2_cgs*exp(-s.k2*tau)

    def temp(s, tau):
        tau = asarray(tau)
        result = s.temp_rad(tau)
        result[tau>s.tau_rc] = s.temp_conv(tau)[tau>s.tau_rc]
        return result
    
    def pressure(s, tau):        
        # handle scalars
        if not iterable(tau):
            if tau >= s.tau_rc:
                return s.pressure_conv(tau)
            else:
                return s.pressure_rad([s.tau_rc, tau])[1,0]
        else:
            # taus must be in increasing order for the moment
            tau = asarray(tau)
            tau_rad = concatenate(([s.tau_rc], tau[tau<s.tau_rc][::-1]))
            pp_rad = s.pressure_rad(tau_rad)[1:][::-1]
            pp = s.pressure_conv(tau)
            pp[tau<s.tau_rc] = pp_rad
            return pp

## My Extensions
    def lum(s, mm_cgs):
        """Luminosity from all sources at the top of the atmosphere.
        This is the only place that mass enters into the model, so
        just pass it as an argument here."""

        # probably only need the upward radiative flux here
        return 4*pi*G_cgs*mm_cgs*s.frad_net_rad(0.0)

    def lum_int(s, mm_cgs):
        """For cooling, care about how much internal energy is leaking
        out, not luminosity of object."""
        return 4*pi*G_cgs*mm_cgs*s.fint_cgs
        
    def _consistency_check(s):
        """Run after solving for the model to check for problems."""
        if s.tau0 < 2*s.tau_rc:
            raise ValueError, "Reference optical depth is too low"
        if s.tau0 < 5.0:
            raise ValueError, "Reference optical depth is too low"

class PlanetFromFlux(Planet):
    """Make planet specifying fluxes and finding surface gravity"""

    # finding the surface gravity takes a long time and isn't always necessary, so skip it by default.
    def __init__(self, **kw):  
        Planet.__init__(self, **kw)
        # Can't actually solve for the surface gravity here b/c it's
        # the children that solve for tau_rc, etc.

    def _analytic_surface_gravity(s):
        """When the opacity is a separable function of pressure and
        temperature, you can find the surface gravity analtyically.
        Allow this as a special case."""
        p_rc = s.pressure(s.tau_rc)
        num, err = scipy.integrate.quad(s._kappa_separate_p, 0, p_rc)
        igrand = lambda xx: s._kappa_separate_t(s.temp_rad(xx, relaxed=True))
        denom, err = scipy.integrate.quad(igrand, 0, s.tau_rc)
        return num / denom

    def _surface_gravity(s, dgg):
        """find surface gravity by requiring that p=0 at tau=0"""
        # Should include a special case for kappa = const since that's
        # what I'm mostly using and it's analytic.

        def ff(xx):
            result = s._simple_pressure_rad([s.tau_rc, 0], xx)
            if s.verbose >= 3: print "Finding surface gravity", xx, result[1]
            return result[1]

        # Take the analytic shortcut if it's available
        if hasattr(s, '_kappa_separate_t'):
            return s._analytic_surface_gravity()
        gl, gh = 0.0, 1.0
        while ff(gh) > 0 and 2*gh != gh: gh *= 2
        return scipy.optimize.bisect(ff, gl, gh, xtol=dgg)

class PlanetFromFluxTau(PlanetFromFlux):
    """Make planet specifying fluxes and reference optical depth"""

    def __init__(self, tau0=None, dtau0=1e-4, 
                 sig0=None, p0_cgs=None,   
                 dgg=1e-2, gravity=False, relaxed=False, **kw):

        PlanetFromFlux.__init__(self, **kw)

        self.tau0 = tau0

        self.tau_rc, self.t0_cgs = self._model(dtau0)
                    
        if p0_cgs and sig0: raise ValueError, "Can't specify both p0 and sig0"
        # Pressure isn't used very much, you can get away with not specifying it 
        self.p0_cgs = p0_cgs or (sig0 and find_pressure(sig0, self.t0_cgs))

        if gravity:
            self.gg_cgs = self._surface_gravity(dgg)

        # Usually if you're specifying tau, you want to make sure
        # you've gotten the R/C boundary.
        if not relaxed:
            self._consistency_check()


    def _model(s, dtau0):
        """Solve for temperature and flux continuity at the
        radiative-convective boundary.  This is where the big money is.

        For planets with surfaces, you may want to fix the reference
        temperature and find the optical depth.  For planets without
        surfaces I think it makes more sense to fix the reference optical
        depth and find the temperature there.

        Follow suggestions from DSP and specify fluxes via effective temps."""

        def t0_taurc(xx):
            return s.temp_rad(xx)*(s.tau0/xx)**(s.beta/s.nn)

        def ff(xx):
            cnt[0] += 1
            t0 = t0_taurc(xx)
            value =  (s.frad_up_conv(xx, t0=t0) - s.frad_up_rad(xx))
            if s.verbose >= 3: print "Finding root given tau0: ", xx, value/ftot
            return value/ftot

        cnt = [0] 
        ftot = s.f1_cgs + s.f2_cgs + s.fint_cgs

        # per DSP's complaint regarding tlusty, try to make this
        # bulletproof (but don't allow an infinite loop)
        taul, tauh = 1.0, 1.0
        while ff(taul) < 0 and 2*taul != taul: taul /= 2.0
        while ff(tauh) > 0 and 2*tauh != tauh: tauh *= 2.0
        if s.verbose >=2: print "Starting at", taul, tauh, ff(taul), ff(tauh)
        tau_rc = scipy.optimize.bisect(ff, taul, tauh, xtol=dtau0)
        if s.verbose >=2: print "Ending at", tau_rc, ff(tau_rc), cnt[0], "iterations"

        # calculate t0_cgs
        t0_cgs = s.temp_rad(tau_rc)*(s.tau0/tau_rc)**(s.beta/s.nn)
        return [tau_rc, t0_cgs]

class PlanetFromFluxTemp(PlanetFromFlux):
    """Make planet specifying fluxes and reference temperature"""

    def __init__(self, t0_cgs=None, dt0=1e-2,
                 sig0=None, p0_cgs=None,   
                 dgg=1e-2,
                 gravity=False, **kw):

        PlanetFromFlux.__init__(self, **kw)

        self.t0_cgs = t0_cgs

        self.tau_rc, self.tau0 = self._model(dt0)
                    
        if p0_cgs and sig0: raise ValueError, "Can't specify both p0 and sig0"

        self.p0_cgs = p0_cgs or find_pressure(sig0, self.t0_cgs)

        if gravity:
            self.gg_cgs = self._surface_gravity(dgg)

        # Usually if you're specifying T0, you're giving a surface
        # temp and are not obligated to have a large optical depth.
        # self._consistency_check()

    def _model(s, dt0):
        """Solve for temperature and flux continuity at the
        radiative-convective boundary.  This is where the big money is.

        Follow RC and make T0 a model parameter, then solve for tau_rc and
        tau_0.  This is to facilitate comparison with their plots.

        Follow suggestions from DSP and specify fluxes via effective temps."""

        def tau0_taurc(xx):
            return xx*(s.t0_cgs/s.temp_rad(xx))**(s.nn/s.beta)

        def ff(xx):
            cnt[0] += 1
            tau0 = tau0_taurc(xx)
            value =  s.frad_up_conv(xx, tau0=tau0) - s.frad_up_rad(xx)
            if s.verbose >= 3: print "Finding root given t0: ", xx, value/ftot
            return value/ftot

        cnt = [0] 
        ftot = s.f1_cgs + s.f2_cgs + s.fint_cgs

        # per DSP's complaint, try to make this bulletproof (but don't
        # allow an infinite loop)
        taul, tauh = 1.0, 1.0
        while ff(taul) < 0 and 2*taul != taul: taul /= 2.0
        while ff(tauh) > 0 and 2*tauh != tauh: tauh *= 2.0

        if s.verbose >=2: print "Starting at", taul, tauh, ff(taul), ff(tauh)
        tau_rc = scipy.optimize.bisect(ff, taul, tauh, xtol=dt0)
        if s.verbose >=2: print "Ending at", tau_rc, ff(tau_rc), cnt[0], "iterations"

        return [tau_rc, tau0_taurc(tau_rc)]

###############
#### Evolution
# 
# I want to do evolution.  Assuming an EOS, fixing the mass, I can
# _solve_ HSE for the radius of the object.  This means that surface
# gravity is _fixed_.  What I don't know is the internal flux.  Before
# I specified tau_0, t_int and sig0, then solved for tau_rc, T_0, and
# g.  Now I want to swap t_int for g.
#
# So the goal is to specify tau_0 sig0, and g, then solve for T_0,
# tau_rc, and T_int.
#
# I have two algebraic equations and one differential equation that
# define the model.  Luckily the two algebraic equations can be solved
# for T0 and T_int.  So the following should work:
# 
# 1) Solve one algebraic eqn for T0 and eliminate
# 2) Solve the other algebraic equation for T_int(tau_rc)
# 3) Guess tau_rc
# 4) Evaluate pressure_conv(tau_rc) to get BC for integration.
# 5) Integrate HSE from p = p_rc, tau=tau_rc to tau=0.
# 6) Is p(tau=0) = 0?
#
# Good.  Now I know how much energy I'm losing.  But what I want is
# dsigma_dt rather than de/dt.  If I were better at thermo I'd
# probably see how to do this, but take a dumb appoach: I know rho(r),
# sigma => temp(r) => total thermal energy.  Can subtract off
# derivative and find thermal energy at the new time.  In the
# degenerate case, EOS, therefore density don't change, and convection
# mixes things at least in the core, so I should be able to reverse
# this and find the new entropy.  Then solve for a new model and repeat.

def PlanetFromGrav(gg_cgs=None, dtint=1e-2, **kw):
    def make_obj(tint):
        if 'tau0' in kw: 
            mm = PlanetFromFluxTau(tint_cgs=tint, gravity=True, **kw)
        elif 't0_cgs' in kw:
            mm = PlanetFromFluxTemp(tint_cgs=tint, gravity=True, **kw)
        else:
            raise ValueError, "Don't know how to make the model"
        return mm
    
    def ff(tint):
        mm = make_obj(tint)
        result = mm.gg_cgs - gg_cgs
        return result

    # It's an error to try to specify tint:
    if 'tint_cgs' in kw: raise ValueError, "This model solves for Tint"

    tl, th = 100, 100

    while ff(tl) >= 0 and 2*tl != tl: tl /= 2.0
    while ff(th) <= 0 and 2*th != th: th *= 2.0
    tint = scipy.optimize.bisect(ff, tl, th, xtol=dtint)

    return make_obj(tint)
        
class PlanetFromGravDirect(Planet): 
    # Well, I think this is in principle working, and the values look
    # sort of reasonable if you know that you're starting near the
    # correct value.  But it's _very_ touchy.  Starting tau_rc far
    # from the correct value gives nans, small changes make giant
    # differences.  So maybe this is not the way to do this after all.
    # Maybe can define a bunch of models, solve for surface grav, and
    # then interpolate?

    def __init__(self, tau0=None, gg_cgs=None, dtau_rc=1e-2, 
                 sig0=None, p0_cgs=None, **kw):
        Planet.__init__(self, **kw)

        # It's an error to try to specify tint:
        if 'tint_cgs' in kw: raise ValueError, "This model solves for Tint"

        self.tau0 = tau0
        self.sig0 = sig0
        self.gg_cgs = gg_cgs
        self._model(dtau_rc, p0_cgs, sig0)

        self._consistency_check()

    def _find_fint_t0(s, tau_rc):
        # This is fantastically ugly.  

        # Gamma from their paper is defined thusly ito scipy functions
        def Gamma(a,x):
            return scipy.special.gamma(a)*scipy.special.gammaincc(a,x)

        # From temp_rad
        A1 = 0.5*((1 + s.dd*tau_rc + s.k1/s.dd) if s.k1 < s.kmin
              else 1+s.dd/s.k1 + (s.k1/s.dd - s.dd/s.k1)*exp(-s.k1*tau_rc))
        A2 = 0.5*((1 + s.dd*tau_rc + s.k2/s.dd) if s.k2 < s.kmin
              else 1+s.dd/s.k2 + (s.k2/s.dd - s.dd/s.k2)*exp(-s.k2*tau_rc))

        # From frad_up_conv
        ex = 4*s.beta/s.nn
        gamfactor = exp(s.dd*tau_rc)*(s.dd*s.tau0)**(-ex)
        expterm = exp(s.dd*(tau_rc-s.tau0))
        gammadiff = (Gamma(1+ex, s.dd*tau_rc) - Gamma(1+ex, s.dd*s.tau0))
        B = (expterm + gamfactor*gammadiff)

        # From frad_up_rad
        # take the limit as k->0 by hand
        C1 = 0.5*(2+(s.dd-s.k1)*tau_rc if s.k1 < s.kmin
                  else 1 + s.dd/s.k1 + (1-s.dd/s.k1)*exp(-s.k1*tau_rc))
        C2 = 0.5*(2+(s.dd-s.k2)*tau_rc if s.k2 < s.kmin
                  else 1 + s.dd/s.k2 + (1-s.dd/s.k2)*exp(-s.k2*tau_rc))

        # ratio of optical depths that appears...
        R = (s.tau0/tau_rc)**ex

        f_int = (2*B*R*(A1+A2) - 2*(C1+C2))/(2+s.dd*tau_rc - B*R*(1+s.dd*tau_rc))

        sigt04 = A1 + A2 + 0.5*f_int*(1+s.dd*tau_rc)
        t0 = (sigt04/sigma_cgs)**0.25
        return f_int, t0

    def _model(s, dtau_rc, p0_cgs, sig0):

        def ff(tau_rc):
            # Fill out object with trial values
            s.tau_rc = tau_rc
            s.fint_cgs, s.t0_cgs = s._find_fint_t0(tau_rc)
            # I'm trying to solve for t0, so if I've specified sig0, I
            # need to keep updating as t0 changes
            s.p0_cgs = p0_cgs or find_pressure(sig0, s.t0_cgs)
            ps = s.pressure_rad([tau_rc, 0.0])
            return ps[1]

        taul, tauh = 1, 1
        ff(1.1)

        #while ff(taul) < 0 and 2*taul != taul: taul /= 2.0
        #while ff(tauh) > 0 and 2*tauh != tauh: tauh *= 2.0
        #tau_rc = scipy.optimize.bisect(ff, taul, tauh, xtol=dtau_rc)

        # function itself sets params in the object, so call it one
        # more time for good measure, although it's probably not
        # necessary.
        #ff(tau_rc)

# Example parameters
# evolve([0, 1e-3], 2e30, gg_cgs=9053, t1_cgs=150, k1=100, t2_cgs=105, k2=0.06, tau0=1000, sig0=13.75, nn=2, alpha=0.85, gamma=1.4, dd=1.5, kappa_cgs=0.2)
def evolve(ts_gyr, mass, gg_cgs=None, sig0=None, gamma=1.67, **kw):
    """Idea here is that solution to lane-emden equation have cores,
    not cusps, so the mean density is an ok approx to the density
    throughout the object.  Then the entropy gives the temperature,
    which gives the thermal energy of the whole thing in terms of
    entropy per baryon and density.  Now you can take the derivative
    and get dsigma/dt in terms of dE/dt, which is given by the model.
    This requires that the pressure is coming from degenerate
    electrons and the entropy comes from non-degenerate protons.  As
    the entropy becomes less than 1 or the electrons become
    non-degenerate, this breaks down.  For now, you must specify the
    radius, but the idea is that this will come from a solution to the
    lane-emden equation (at first) or a more sophisticated HSE
    solution (later)."""
    
    mp = 1.67e-24
    mu = 1.22 # non-ionized primordial gas
    hbar = 1.05e-27
    bb = mu*mp*(mp/(2*pi*hbar**2))**1.5
    
    rr = sqrt(G_cgs*mass/gg_cgs)
    print "Using r = ", rr/7.1e9, "r_j"
    rho = 3*mass/(4*pi*rr**3)
    efactor = mass*(rho/bb)**(2/3.0)/(mu*mp*(gamma-1))
                    
    def derivs(yy,tt):
        sig = yy[0]
        en = efactor*exp(2*sig/3.0 - 5/3.0)
        mm = PlanetFromGrav(gg_cgs=gg_cgs, gamma=gamma, sig0=sig, **kw)
        dsdt = -3*mm.lum_int(mass)/(2*en)
        return [dsdt]

    ts = 1e9*3.15e7*asarray(ts_gyr)
    return scipy.integrate.odeint(derivs, [sig0], ts)
    
def evolve_plot():
    ts = linspace(0,1e-1,25)
    gg=8337
    mass = 2e30
    
    pl.clf()
    params = dict(t1_cgs=150, k1=100, t2_cgs=105, k2=0.06, tau0=1000,
                  nn=2, alpha=0.85, gamma=1.4, dd=1.5, kappa_cgs=0.2)
    sigs = [12.75,13.75,14.75]

    for sig in sigs:
        sig_time = evolve(ts, mass, sig0=sig, gg_cgs=gg, **params)

        mms = [PlanetFromGrav(sig0=the_sig, gg_cgs=gg, **params)
              for the_sig in sig_time]
        
        pl.subplot(2,3,1)
        pl.plot(ts, sig_time)
        pl.subplot(2,3,2)
        pl.semilogy(ts, [the_m.lum_int(mass) for the_m in mms])
        pl.subplot(2,3,3)
        pl.plot(ts, [the_m.tau_rc for the_m in mms])
        pl.subplot(2,3,4)
        pl.plot(ts, [the_m.temp([1.0])[0] for the_m in mms])
        pl.subplot(2,3,5)
        pl.plot(ts, 1e-3*array([the_m.pressure(1.0) for the_m in mms]))

    pl.subplot(2,3,1)
    pl.xlabel('t (gyr)')        
    pl.ylabel('sigma')
    pl.subplot(2,3,2)
    pl.xlabel('t (gyr)')        
    pl.ylabel('L (erg/s)')
    pl.subplot(2,3,3)
    pl.xlabel('t (gyr)')        
    pl.ylabel('tau_rc')
    pl.subplot(2,3,4)
    pl.xlabel('t (gyr)')        
    pl.ylabel('T(tau=1) (K)')
    pl.subplot(2,3,5)
    pl.ylabel('P(tau=1) (mbar)')    
    pl.xlabel('t (gyr)')
    
    pl.draw()
    

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
                                            
def plot_grav(tints=logspace(1,3,30)):
    
    result = []
    for tint in tints:
        mm = PlanetFromFluxTemp(t1_cgs=69, k1=100, t2_cgs=105, k2=0.06, t0_cgs=191, tint_cgs=tint, p0_cgs=1.1*1e6, nn=2, alpha=0.85, gamma=1.4, dd=1.5, kappa_cgs=0.2, gravity=True)
        mm = PlanetFromFluxTau(tint_cgs=tint, tau0=1000, sig0=9, kappa_cgs=1.0, gravity=True)
        result.append(mm.gg_cgs)

    pl.clf()
    pl.loglog(tints, result)
    pl.draw()

def plot_t0(tints=logspace(1,4,30)):
    
    result = []
    for tint in tints:
        mm = PlanetFromFluxTau(tint_cgs=tint, tau0=1000, sig0=9, kappa_cgs=1.0, gravity=True)
        result.append(mm.t0_cgs)

    pl.clf()
    pl.loglog(tints, result)
    pl.draw()
        
def plot_model(tau, mm, pressure=False):

    def flux_marks():
        # mark rad/conv boundary
        pl.semilogy([-100, 100], [rc_bnd, rc_bnd], 'k')    
        # mark zero
        pl.semilogy([0, 0], [yy[0], yy[-1]], 'k')
        pl.xlim(-50,300)

    def temp_marks():
        # mark rad/conv boundary
        pl.loglog([1, 100], [rc_bnd, rc_bnd], 'k')    

    def flux_to_temp(xx):
        xx = asarray(xx)
        return sign(xx)*(abs(xx)/sigma_cgs)**0.25

    if pressure:
        yy = mm.pressure(tau)
        rc_bnd = mm.pressure(mm.tau_rc)
    else:
        yy = tau
        rc_bnd = mm.tau_rc

    pl.clf()

    pl.subplot(2,2,1)
    pl.semilogy(flux_to_temp(mm.frad_up_rad(tau)), yy, 'b:')
    pl.semilogy(flux_to_temp(mm.frad_down_rad(tau)), yy, 'b--')
    pl.semilogy(flux_to_temp(mm.frad_net_rad(tau)), yy, 'b')
    pl.ylim(yy[-1], yy[0])
    pl.legend(('up', 'down', 'net'), loc='upper left')
    pl.title('Rad teff in rad zone')
    pl.semilogy(flux_to_temp(mm.fstar_net(tau)), yy, 'g')
    flux_marks()

    pl.subplot(2,2,2)
    pl.semilogy(flux_to_temp(mm.frad_up_conv(tau)), yy, 'r:')
    pl.semilogy(flux_to_temp(mm.frad_down_conv(tau)), yy, 'r--')
    pl.semilogy(flux_to_temp(mm.frad_net_conv(tau)), yy, 'r')
    pl.ylim(yy[-1], yy[0])
    pl.legend(('up', 'down', 'net'), loc='upper left')
    pl.title('Rad teff in conv zone')
    pl.semilogy(flux_to_temp(mm.fstar_net(tau)), yy, 'g')
    flux_marks()
    
    pl.subplot(2,2,3)
    pl.title('Net teff')
    pl.semilogy(flux_to_temp(mm.fstar_net(tau)), yy, 'g')
    pl.semilogy(flux_to_temp(mm.frad_net_rad(tau)), yy, 'b')
    pl.semilogy(flux_to_temp(mm.frad_net_conv(tau)), yy, 'r')
    pl.semilogy(flux_to_temp(mm.fconv_up_conv(tau)), yy, 'm')
    pl.legend(('Star', 'Rad (rad)', 'Rad (conv)', 'Conv'), loc='upper left')
    pl.ylim(yy[-1], yy[0])
    flux_marks()

    pl.subplot(2,2,4)
    pl.loglog(mm.temp_rad(tau), yy, 'b')
    pl.loglog(mm.temp_conv(tau), yy, 'r')
    pl.legend(('Rad', 'Conv'), loc='lower left')
    pl.title('Temperature')
    pl.ylim(yy[-1], yy[0])
    temp_marks()

    pl.draw()

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
            mm = PlanetFromFluxTau(tint_cgs=teff, t1_cgs=0, t2_cgs=0, k1=0, k2=0, 
                        tau0=tau0, nn=nn, alpha=alpha, gamma=gamma, dd=dd, relaxed=True)

            row.append(mm.tau_rc)
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
    tau0 = 2000

    result = [] 
    for fbon in fbons:
        alpha = fbon*nn*gamma/(4.0*(gamma-1))        
        mm = PlanetFromFluxTau(tint_cgs=teff, t1_cgs=0, t2_cgs=0, k1=0, k2=0, tau0=tau0, nn=nn, alpha=alpha, gamma=gamma, dd=dd)
        result.append(mm.tau_rc)
    result = asarray(result)

    pl.semilogy(fbons, dd*result)
    pl.ylim(20, 0.01)
    pl.draw()

def fig3():
    # This looks grossly like their plot, but not quantitatively the
    # same.  There are some parameters that they didn't specify,
    # though.

    # their plot actually shows the radiative temp only, I include the
    # overall temp as dotted lines.
    
    nn=2
    gamma=1.4
    tint=0
    tau0=2
    
    # They do not seem to appreciate that tau_rc = tau_0 when you have
    # no internal energy source.  On the other hand, maybe they're
    # thinking of terrestrial planets w/ a surface.
    
    # Value shouldn't matter?
    t1=1000
    p0 = 1e6
    dd=1.0  # actually this one does matter
    taus = logspace(-2,2,100)

    # value unspecified in paper?
    alpha = 1.0  # this makes a difference
    kappa = 0.2
    
    pl.clf()

    cols = ['b', 'r', 'g', 'k']
    kods = [0, 0.1, 0.5, 10]
    cols = ['k']
    kods = [10]
    
    for kod,cc in zip(kods, cols):
        kk = dd*kod

        mm = PlanetFromFluxTau(tint_cgs=tint, t1_cgs=t1, k1=kk, tau0=tau0, p0_cgs=p0, nn=nn,
                    alpha=alpha, gamma=gamma, dd=dd, kappa_cgs=kappa, gravity=True, relaxed=True)
        xx = (mm.temp_rad(taus)/t1)**4
        xx2 = (mm.temp(taus)/t1)**4
        yy = mm.pressure(taus)/p0
        pl.loglog(xx,yy, c=cc)
        pl.loglog(xx2,yy, c=cc, ls=':')

    pl.xlim(0.4, 10)
    pl.ylim(2, 0.1)
    pl.draw()
    
def fig4():
    # Looks good but I only get n=1
    
    # shouldn't matter?
    tint=0
    dd = 2.0
    t1 = 10
    tau0=1000
    p0=1e4
    gamma = 1.2
    kappa = 2.0
    alpha = 3.0
    
    kods = logspace(-3,-0.01,4)
    nns = [1,2,4]
    
    taus = logspace(-2,2,100)

    pl.clf()
    
    for nn in nns:        
        result = []
        for kod in kods:
            kk = kod*dd
            mm = PlanetFromFluxTau(tint_cgs=tint, k1=kk, nn=nn,
                        t1_cgs=t1, tau0=tau0, p0_cgs=p0,
                        alpha=alpha, gamma=gamma, dd=dd, kappa_cgs=kappa, gravity=True)
            # I think I'm supposed to just find the pressure if it
            # were radiative, to find out if the region is convective
            # or not.
            tt = mm.temp_rad(taus)
            pp = mm.pressure_rad_hypothetical(taus)
            log_deriv = diff(log(tt))/diff(log(pp))
            result.append(log_deriv.max())
        pl.semilogx(kods, result)
        print result
        
    pl.ylim(0,0.4)
    pl.draw()

def fig5(fbon):
    """fbon = 0.46 for upper panel, 0.57 for lower panel"""

    # value shouldn't matter?
    t1 = 1000
    alpha = 1.0
    gamma = 1.4
    dd = 1.5

    # axes
    kods = logspace(-3,-0.01,30)
    dtau0s = logspace(-1,1,30)
    levels = [0.05, 0.1, 0.2, 0.4, 1.0, 5.0]
    
    # specified
    tint=0
    nn=2
    alpha = fbon*nn*gamma/((gamma-1)*4)
    
    result = []
    for kod in kods:
        kk = kod*dd
        row = [] 
        for dtau0 in dtau0s:
            tau0 = dtau0/dd
            mm = PlanetFromFluxTau(tint_cgs=tint, t1_cgs=t1, k1=kk, tau0=tau0,
                        nn=nn, alpha=alpha, gamma=gamma, dd=dd, relaxed=True)
            row.append(dd*mm.tau_rc)
        result.append(row)

    X,Y = structure.make_grid(kods, dtau0s)
    pl.clf()
    pl.contour(X,Y,result, levels)
    pl.gca().set_xscale('log')
    pl.gca().set_yscale('log')
    pl.ylim(10, 0.1)
    pl.draw()
    
def fig6():

    # Shouldn't matter?
    kappa = 0.2
    ti = 100
    p0 = 1e6
    dd = 1.5
    gamma = 1.4
    alpha = 1
    tau0 = 1000
    tau = logspace(-2,3,100)
    
    # Specified
    t1 = 10*ti
    nn = 2
    kods = [0.1, 2]

    pl.clf()
    for kod in kods:
        kk = kod*dd
        mm = PlanetFromFluxTau(tint_cgs=ti, t1_cgs=t1, k1=kk, tau0=tau0, p0_cgs=p0, nn=nn,
                    alpha=alpha, gamma=gamma, dd=dd, kappa_cgs=kappa, gravity=True)
        xx = (mm.temp(tau)/t1)**4
        yy = mm.pressure(tau)/p0
        #yy = mm.pressure_rad_hypothetical(tau)/p0
        pl.loglog(xx,yy)
        
    pl.ylim(1e3,0.1)
    pl.ylim(1e3,0.001)
    pl.xlim(0.3,200)
    pl.draw()

def fig7():
    # axes
    kods = logspace(-1, 1, 4)
    fratios = logspace(-1, 5, 5)
    tau = logspace(0,4,30)
    levels = [2,10,1e2,1e3,1e4,1e5]

    # shouldn't matter
    tint = 100
    gamma = 1.4
    tau0 = 1000
    p0 = 1e6
    alpha = 1
    dd = 1.5
    kappa = 0.2
    
    # specified
    nn=2

    avetau = ave(tau)
    
    result = [] 
    for kod in kods:
        kk = kod*dd
        row = []
        for fratio in fratios:
            t1 = tint*fratio**0.25
            mm = PlanetFromFluxTau(tint_cgs=tint, t1_cgs=t1, k1=kk, tau0=tau0, sig0=None, p0_cgs=p0,
                        nn=nn, alpha=alpha, gamma=gamma, dd=dd, kappa_cgs=kappa, gravity=True)
            tt = mm.temp_rad(tau)
            pp = mm.pressure_rad_hypothetical(tau)
            limit = (gamma-1)/gamma
            lderiv = diff(log(tt))/diff(log(pp))
            idx = lderiv.searchsorted(limit)
            if idx != len(avetau):
                row.append(dd*avetau[idx])
            else:
                row.append(dd*tau[-1])
            
        result.append(row)
        
    print result
    X,Y = structure.make_grid(kods, fratios)

    pl.clf()
    pl.contour(X,Y,result,levels)
    pl.gca().set_xscale('log')
    pl.gca().set_yscale('log')
    pl.xlim(0.1, 10)
    pl.ylim(1e5, 0.1)
    pl.draw()
    
def fig8_9():
    t1 = (160*1e3/sigma_cgs)**0.25

    # unspecified
    dd=1.5
    kappa=0.2
    
    common = dict(tint_cgs=0, t1_cgs=t1, k1=0, t0_cgs=730, p0_cgs=92*1e6, alpha=0.8,
                  gamma=1.3, dd=dd, kappa_cgs=kappa, gravity=True)

    m1 = PlanetFromFluxTemp(nn=1, **common)
    m2 = PlanetFromFluxTemp(nn=2, **common)

    print "Model 1", m1.tau_rc, m1.tau0
    print "Model 2", m2.tau_rc, m2.tau0
    
    pl.figure(1)
    pl.clf()
    tau1 = logspace(-2, 3, 100)
    tau2 = logspace(-2, 5, 100)
    pl.semilogy(m1.temp(tau1), 1e-6*m1.pressure(tau1))
    pl.semilogy(m2.temp(tau2), 1e-6*m2.pressure(tau2))
    pl.ylim(1e2, 1e-2)
    pl.xlim(150, 800)
    pl.draw()

    pl.figure(2)
    pl.clf()

    pp = 1e-6*m1.pressure(tau1)
    # red for stuff in the convective zone
    # blue for stuff in the radiative zone
    # solid for flux up
    # dashed for flux down
    # dotted for temperature
    pl.semilogy(1e-3*sigma_cgs*m1.temp_conv(tau1)**4, pp, 'r:')    
    pl.semilogy(1e-3*sigma_cgs*m1.temp_rad(tau1)**4, pp, 'b:')
    pl.semilogy(1e-3*m1.frad_down_conv(tau1), pp, 'r--')
    pl.semilogy(1e-3*m1.frad_up_conv(tau1), pp, 'r')
    pl.semilogy(1e-3*m1.frad_up_rad(tau1), pp, 'b-')
    pl.semilogy(1e-3*m1.frad_down_rad(tau1), pp, 'b--')

    p_rc = 1e-6*m1.pressure(m1.tau_rc)
    pl.plot([0, 800], [p_rc, p_rc], 'k')
    pl.xlim(0, 800)
    pl.ylim(2, 0.01)
    pl.draw()

def fig10_11():
    # not specified
    kappa = 1.0
    dd = 1.5

    def to_temp(xx):
        return (xx*1e3/sigma_cgs)**0.25

    common = dict(tint_cgs=to_temp(5.4), p0_cgs=1.1*1e6, nn=2, alpha=0.85, gamma=1.4, dd=dd, kappa_cgs=kappa, gravity=True)
    m1 = PlanetFromFluxTemp(t1_cgs=0, k1=0, t2_cgs=to_temp(8.3), k2=0, t0_cgs=165, **common)
    m2 = PlanetFromFluxTemp(t1_cgs=0, k1=0, t2_cgs=to_temp(8.3), k2=0, t0_cgs=168, **common)
    m3 = PlanetFromFluxTemp(t1_cgs=to_temp(1.3), k1=100, t2_cgs=to_temp(7.0), k2=0.06, t0_cgs=191, **common)
    
    print "Model 1", m1.tau_rc, m1.tau0
    print "Model 2", m2.tau_rc, m2.tau0
    print "Model 3", m3.tau_rc, m3.tau0

    pl.figure(1)
    pl.clf()
    tau = logspace(-3, 2, 100)
    pl.semilogy(m1.temp(tau), 1e-6*m1.pressure(tau))
    pl.semilogy(m2.temp(tau), 1e-6*m2.pressure(tau))
    pl.semilogy(m3.temp(tau), 1e-6*m3.pressure(tau))
    pl.ylim(1e0, 1e-3)
    pl.xlim(100, 200)
    pl.draw()

    pl.figure(2)
    pl.clf()

    pp = 1e-6*m1.pressure(tau)
    # red for stuff in the convective zone
    # blue for stuff in the radiative zone
    # solid for flux up
    # dashed for flux down
    # dotted for temperature
    pl.semilogy(1e-3*sigma_cgs*m1.temp_conv(tau)**4, pp, 'r:')    
    pl.semilogy(1e-3*sigma_cgs*m1.temp_rad(tau)**4, pp, 'b:')
    pl.semilogy(1e-3*m1.frad_down_conv(tau), pp, 'r--')
    pl.semilogy(1e-3*m1.frad_up_conv(tau), pp, 'r')
    pl.semilogy(1e-3*m1.frad_up_rad(tau), pp, 'b-')
    pl.semilogy(1e-3*m1.frad_down_rad(tau), pp, 'b--')

    pl.semilogy(1e-3*m1.fstar_net(tau), pp, 'g')
    pl.semilogy(1e-3*m1.fconv_up_conv(tau), pp, 'm')

    # cyan lines are net fluxes, solid applies in radiative zone,
    # dashed applies in convective zone.
    pl.semilogy(1e-3*m1.frad_net_rad(tau), pp, 'c')
    pl.semilogy(1e-3*m1.frad_net_conv(tau), pp, 'c--')
                
    p_rc = 1e-6*m1.pressure(m1.tau_rc)
    pl.plot([0,20], [p_rc, p_rc], 'k')
    pl.xlim(0, 20)
    pl.ylim(1, 0.001)
    pl.draw()


def fig12_13():

    def to_temp(xx):
        return (xx*1e3/sigma_cgs)**0.25

    # didn't specify
    dd = 1.5
    kappa = 0.2
    
    mm = PlanetFromFluxTemp(tint_cgs=0, t1_cgs=to_temp(1.5), k1=120, t2_cgs=to_temp(1.1), k2=0.2, t0_cgs=94,
           p0_cgs=1.5*1e6, nn=1.33, alpha=0.77, gamma=1.4, dd=dd, kappa_cgs=kappa, gravity=True)

    print "Model", mm.tau_rc, mm.tau0

    pl.clf()
    tau = logspace(-3, 2, 100)
    pl.semilogy(mm.temp(tau), 1e-6*mm.pressure(tau))
    pl.ylim(1e0, 1e-3)
    pl.xlim(60, 175)
    pl.draw()

    pl.figure(2)
    pl.clf()

    pp = 1e-6*mm.pressure(tau)
    # red for stuff in the convective zone
    # blue for stuff in the radiative zone
    # solid for flux up
    # dashed for flux down
    # dotted for temperature
    pl.semilogy(1e-3*sigma_cgs*mm.temp_conv(tau)**4, pp, 'r:')    
    pl.semilogy(1e-3*sigma_cgs*mm.temp_rad(tau)**4, pp, 'b:')
    pl.semilogy(1e-3*mm.frad_down_conv(tau), pp, 'r--')
    pl.semilogy(1e-3*mm.frad_up_conv(tau), pp, 'r')
    pl.semilogy(1e-3*mm.frad_up_rad(tau), pp, 'b-')
    pl.semilogy(1e-3*mm.frad_down_rad(tau), pp, 'b--')

    pl.semilogy(1e-3*mm.fstar_net(tau), pp, 'g')
    pl.semilogy(1e-3*mm.fconv_up_conv(tau), pp, 'm')

    # cyan lines are net fluxes, solid applies in radiative zone,
    # dashed applies in convective zone.
    pl.semilogy(1e-3*mm.frad_net_rad(tau), pp, 'c')
    pl.semilogy(1e-3*mm.frad_net_conv(tau), pp, 'c--')
    
    p_rc = 1e-6*mm.pressure(mm.tau_rc)
    pl.plot([0,20], [p_rc, p_rc], 'k')
    pl.xlim(0, 4)
    pl.ylim(1.5, 0.001)
    pl.draw()

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
