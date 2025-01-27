# import modules
import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy import constants, units
from dataclasses import dataclass
from scipy.signal import convolve

from imfits import Imfits
from imfits.drawmaps import AstroCanvas


### constants
Ggrav  = constants.G.cgs.value        # Gravitational constant
Msun   = constants.M_sun.cgs.value    # Solar mass (g)
Lsun   = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
Rsun   = constants.R_sun.cgs.value    # Solar radius (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mH     = constants.m_p.cgs.value      # Proton mass (g)
hp     = 6.626070040e-27 # Planck constant [erg s]
sigsb  = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
NA     = 6.022140857e23  # mol^-1
Mearth = 3.0404e-6*Msun  # Earth mass [g]
Mjup   = 9.5479e-4*Msun  # Jupiter mass [g]

# unit
auTOcm = units.au.to('cm')            # 1 au (cm)
pcTOcm = units.pc.to('cm')            # 1 pc (cm)
auTOkm = 1.495978707e8  # AU --> km
auTOpc = 4.85e-6        # au --> pc
pcTOau = 2.06e5         # pc --> au


# Radial profiles
def powerlaw_profile(r, p, I0, r0=1.):
    return I0*(r/r0)**(-p)

def ssdisk(r, Ic, rc, gamma, beta = None):
    beta_p = gamma if beta is None else beta # - beta = - gamma - q
    return Ic * (r/rc)**(- beta_p) * np.exp(-(r/rc)**(2. - gamma))

def gaussian_profile(r, I0, sigr):
    return I0*np.exp(-r**2./(2.*sigr**2))

# Gaussians
# 1D
def gaussian1d(x, amp, mx, sig):
    return amp*np.exp(-(x - mx)**2/(2*sig**2)) #+ offset

# 2D
def gaussian2d(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
    '''
    Generate normalized 2D Gaussian

    Parameters
    ----------
     x: x value (coordinate)
     y: y value
     A: Amplitude. Not a peak value, but the integrated value.
     mx, my: mean values
     sigx, sigy: standard deviations
     pa: position angle [deg]. Counterclockwise is positive.
    '''
    shape = x.shape
    if pa: # skip if pa == 0.
        x, y = rotate2d(x.ravel(), y.ravel(), pa)

    coeff = A if peak else A/(2.0*np.pi*sigx*sigy)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    return (coeff*expx*expy).reshape(shape)


# 2D rotation
def rotate2d(x, y, angle):
    '''
    Rotate Cartesian coordinates.
    Right hand direction will be positive.

    Parameters
    ----------
    x, y (float or 1D ndarray): coordinates.
    angle (float): rotational angle (radian)
    '''

    rot = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]]
        )

    return rot @ np.array([x, y])


def vkep(r, ms, z = 0.):
    return np.sqrt(Ggrav * ms * r * r / (r*r + z*z)**(1.5))


# Planck function
def Bv(T,v):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    v = v * 1.e9 # GHz --> Hz
    #print((hp*v)/(kb*T))
    exp=np.exp((hp*v)/(kb*T)) - 1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp
    #print(exp, T, v)
    return Bv


# Jy/beam
def Bv_Jybeam(T,v,bmaj,bmin):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    bmaj, bmin: beamsize [arcsec]
    '''

    # units
    bmaj = np.radians(bmaj / 3600.) # arcsec --> radian
    bmin = np.radians(bmin / 3600.) # arcsec --> radian
    v = v * 1.e9 # GHz --> Hz

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj * bmin * C2  # beam --> str


    #print(np.nanmax((hp*v)/(kb*T)), np.nanmin(T))
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    #print(np.nanmax(exp))
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # cgs --> Jy/beam
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv*bTOstr     # Jy/str --> Jy/beam
    return Bv



@dataclass(slots=True)
class SSDisk:

    Ic: float = 1.
    rc: float = 1.
    beta: float = 0.
    gamma: float = 0.
    inc: float = 0.
    pa: float = 0.
    ms: float = 0.
    vsys: float = 0.

    def set_params(self, Ic = 0, rc = 0, beta = 0, gamma = 0, 
        inc = 0, pa = 0, ms = 0, vsys = 0):
        '''

        Parameters
        ----------
         Ic (float): 
         rc (float): au
         inc (float): deg
         pa (float): deg
         z0 (float): au
         r0 (float): au
         ms (float): Msun
        '''
        # initialize parameters
        self.Ic = Ic
        self.rc = rc
        self.beta = beta
        self.gamma = gamma
        self.inc = inc
        self.pa = pa
        self.ms = ms
        self.vsys = vsys


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def build(self, xx_sky, yy_sky):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)

        # deprojection from sky to disk coordinates
        x, y = xx_sky.ravel(), yy_sky.ravel()
        # rotate by pa
        xp, yp = rotate2d(x, y, _pa_rad + 0.5 * np.pi)
        yp /= np.cos(_inc_rad)

        # local coordinates
        r = np.sqrt(xp * xp + yp * yp) # radius
        th = np.arctan2(yp, xp) # azimuthal angle (rad)

        # take y-axis as the line of sight
        vlos = vkep(r * auTOcm, self.ms * Msun) \
        * np.cos(th) * np.sin(_inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s
        I_int = ssdisk(r, self.Ic, self.rc, self.gamma, self.beta)

        return I_int.reshape(xx_sky.shape), vlos.reshape(xx_sky.shape)


    def build_cube(self, xx, yy, v, beam = None, linewidth = 0., dist = 140.):
        # get intensity and velocity fields
        I_int, vlos = self.build(xx, yy)
        
        # vaxes
        ny, nx = xx.shape
        nv = len(v)
        delv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])
        I_cube = np.zeros((nv, ny, nx))

        # making a cube
        for i in range(nv):
            vindx = np.where((ve[i] <= vlos) & (vlos < ve[i+1]))
            I_cube[i,vindx[0], vindx[1]] = I_int[vindx]

        # Convolve beam if given
        if beam is not None:
            gaussbeam = gaussian2d(xx, yy, 1., 0., 0., 
            beam[1] * dist / 2.35, beam[0] * dist / 2.35, beam[2], peak=True)
            gaussbeam /= np.nansum(gaussbeam)

            I_cube /= np.abs((xx[0,0] - xx[0,1])*(yy[1,0] - yy[0,0])) # per pixel to per arcsec^2
            I_cube *= np.pi/(4.*np.log(2.)) * beam[0] * beam[1] # per arcsec^2 --> per beam

            # beam convolution
            I_cube = np.where(np.isnan(I_cube), 0., I_cube)
            I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')

        # line broadening
        if linewidth is not None:
            gaussbeam = np.exp(-( (v - self.vsys) /(2. * linewidth / 2.35))**2.)
            I_cube = convolve(I_cube, np.array([[gaussbeam]]).T, mode='same')

        return I_cube


def main():
    # --------- input ---------
    # model params
    Ic, rc, beta, gamma = [1., 600., 1.5, 1.]
    inc = 70.
    pa = 69.
    ms = 1.6
    vsys = 7.4

    # object
    f = 'l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.croped.fits'
    dist = 140.
    # -------------------------


    # --------- main ----------
    # read fits file
    cube = Imfits(f)
    cube.trim_data([-5., 5.,], [-5.,5.], [4.4, 10.4])
    xx = cube.xx * 3600. * dist # in au
    yy = cube.yy * 3600. * dist # in au
    v = cube.vaxis # km/s


    # model
    model = SSDisk(Ic, rc, beta, gamma, inc, pa, ms, vsys)
    modelcube = model.build_cube(xx, yy, v, cube.beam, 0.5, dist)
    vmin, vmax = np.nanmin(modelcube)*0.5, np.nanmax(modelcube)*0.5


    # plot
    canvas = AstroCanvas((4,7),(0,0), imagegrid=True)
    canvas.channelmaps(cube, contour=True, color=False,
        clevels = np.array([-3,3.,6.,9.,12.,15])*5e-3)
    for i, im in enumerate(modelcube):
        if i < len(canvas.axes):
            ax = canvas.axes[i]
            ax.pcolor(xx / dist, yy / dist, im, shading='auto', rasterized=True,
                vmin = vmin, vmax = vmax, cmap='PuBuGn')
        else:
            break

    plt.show()
    # -------------------------



if __name__ == '__main__':
    main()