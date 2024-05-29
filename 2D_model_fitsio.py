# import modules
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from astropy import constants, units
from dataclasses import dataclass
from scipy.signal import convolve
from astropy.io import fits
import matplotlib.colors as colors

current_path = os.getcwd()
print(current_path)
# Check if it is office computer or laptop and set path of imfits accordingly
if current_path.split(sep=':')[0] == 'D':                           # Office computer
    print("In office")
    sys.path.append("D:\L1489_IRS_ssp\imfits")
else:                                                               # Laptop N
    sys.path.append("E:/Mihir_new/ASIAA-SSP/imfits/")

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

def ssdisk_gaussian_ring(r, theta, Ic, rc, gamma, beta = None, ring_height = None, 
                         ring_loc = 600, ring_width = 1):
    
    if ring_height == None:
        ring_height = 0.

    beta_p = gamma if beta is None else beta # - beta = - gamma - q

    profile_ssdisk = Ic * (r/rc)**(- beta_p) * np.exp(-(r/rc)**(2. - gamma)) 
    profile_gauss_out = np.max(profile_ssdisk)*ring_height*np.exp(-((r-ring_loc)/ring_width)**2)
    profile_gauss_in = np.max(profile_ssdisk)*(2.5)*ring_height*np.exp(-((r-63.)/190.)**2)

    return  profile_gauss_in + profile_gauss_out

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
    '''
    Returns kepler velocity at a given point in a protoplanetary disk. Assumes mass of the cental star (M_star) 
    is much greater then disk mass, and hence only M_star determines the kepler velocity.

    Parameters:
    r       : `float`, radial distance of the point wrt the star
    M_star  : `float`, Mass of the star
    z       : `float`, vertical height of point from disk midplane
    '''
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

def write_fits(model_cube, modelcube_header ):
    # Now let's convert model cube into a fits file using wcs axis of the template
    #print(repr(cube.header))
    model_hdu = fits.PrimaryHDU(data = model_cube, header= modelcube_header )
    model_hdu.writeto("L1489_irs_model_3.fits", overwrite=True)
    return(0)

@dataclass()
class SSDisk:

    Ic: float = 1.              #  Central Intensity
    rc: float = 1.              #  Outer radius of disk model in AU
    beta: float = 0.
    gamma: float = 0.
    inc: float = 73.
    pa: float = 69.
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

    def build(self, xx_sky, yy_sky, intensity_function = None, rp_kwargs = None):
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
        print(r)
        print(max(r), min(r))
        theta = np.arctan2(yp, xp) # azimuthal angle (rad)

        # take y-axis as the line of sight
        vlos = vkep(r * auTOcm, self.ms * Msun)*np.cos(theta) * np.sin(_inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s
        
        # Calculate intensity
        if intensity_function == None:
            I_int = ssdisk(r, self.Ic, self.rc, self.gamma, self.beta)

        else:
            I_int = intensity_function(r, theta, self.Ic, self.rc, 
                                       self.gamma, self.beta, **rp_kwargs)

        plot_intensity_radii = True

        if plot_intensity_radii:

            fig, axes = plt.subplots()
            axes.scatter(r/(140.), I_int, marker = 'o', s = 2.)
            print("plotting intensity")
            axes.set_xlim(0.1, 3.)
            axes.set_ylim(1,100)
            axes.set_yscale('log')
            axes.set_xscale("log")
            axes.grid()
            plt.show()
            plt.close()



        return I_int.reshape(xx_sky.shape), vlos.reshape(xx_sky.shape)

    def build_cube(self, xx, yy, v, beam = None, linewidth = 0., 
                   dist = 140., radial_profile = None, rp_kwargs = None):

        """
        Builds an intensity cube given meshgrids of x,y and v axes. Convolves intesity with beam and line broadening.

        Parameters:
        xx              : `np.meshgrid` object for x coordinate
        yy              : `np.meshgrid` object for y coordinate
        v               : `np.ndarray` of v (velocity)
        beam            : ``, convolving beam 
        linewidth       : `float`, Line width
        dist            : `float`, Distance of star in AU
        radial_profile  : A function of `r` and `theta` giving intensity profile. The function must be of the form fun(r,theta,Ic,Rc,**kwargs)
        **rp_kwargs     : `dict`, Dictionary specifying constants and their values as key, value pair required for radial_profile function

        Returns:
        I_cube          : `np.ndarray` of shape (len(v), len(y), len(x))
        I_int           : `np.ndarray` of shape (len(x_sky), len(y_sky)), gives integrated intensity over whole spectral cube
        """
        # get intensity and velocity fields
        I_int, vlos = self.build(xx, yy, intensity_function = radial_profile, 
                                 rp_kwargs = rp_kwargs)
        
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

            I_cube /= np.abs((xx[0,0] - xx[0,1])*(yy[1,0] - yy[0,0])) # per pixel to per arcsec^2
            I_cube *= np.pi/(4.*np.log(2.)) * beam[0] * beam[1] # per arcsec^2 --> per beam

            # beam convolution
            I_cube = np.where(np.isnan(I_cube), 0., I_cube)
            I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')

        # line broadening
        if linewidth is not None:
            gaussbeam = np.exp(- 0.5 * ( (v - self.vsys) /(linewidth / 2.35))**2.)
            I_cube = convolve(I_cube, np.array([[gaussbeam]]).T, mode='same')

        return I_cube, I_int

def main():
    # --------- input ---------
    # model params
    Ic, rc, beta, gamma = [1, 600., None, 0.9] # rc 
    inc = 10.
    pa = 0.
    ms = 1.6
    vsys = 7.3
    dist = 140.

    # Observed disk cube
    f_cube = 'uid___A002_b_6.cal.l1489_irs.spw_1_7.line.cube.clean.c_baseline_0.image.pbcor.Regridded.Smoothened.fits'
    
    # Observed disk PV
    f_PV = 'uid___A002_b_6.cal.l1489_irs.spw_1_7.line.cube.clean.c_baseline_0.image.pbcor.Regridded.Smoothened.PV_69_w1.fits'

    # --------- main ----------
    # read fits file
    cube = Imfits(f_cube)
    # shift coordinate center (the mesh grid will now bw centerd on pixel at this location)
    cube.shift_coord_center(coord_center = '04h04m43.07s 26d18m56.30s')
    cube.trim_data([-9., 9.,], [-9.,9.], [2.7, 11.9])
    # cube.trim_data(vlim = [1.19,13.5])   # trim_data([RA range in arcsec offset from center], [Dec range], [offset velocity range in kmps])
    
    xx = cube.xx * 3600. * dist # in au
    yy = cube.yy * 3600. * dist # in au
    v = cube.vaxis # km/s


    # model
    model = SSDisk(Ic, rc, beta, gamma, inc, pa, ms, vsys)
    # xx, yy, v, beam = None, linewidth = 0., dist = 140., radial_profile = None, **rp_kwargs
    modelcube, model_int_map = model.build_cube(xx, yy, v, cube.beam, 0.7, dist, 
                                 radial_profile = ssdisk_gaussian_ring, 
                                 rp_kwargs = {'ring_height' : 0.1, 'ring_loc' : 350., 
                                              'ring_width' : 50.})
    vmin, vmax = np.nanmin(modelcube)*0.5, np.nanmax(modelcube)*0.5


    write_fits(model_cube = modelcube, modelcube_header= cube.header)


    print(np.shape(modelcube))
    # Let's get PV plot out of the modelcube  
    pv_model = np.squeeze(modelcube[:, :, 150])

    plot_int_map = True

    if plot_int_map == True:
        
        fig, axes = plt.subplots()

        a = axes.pcolormesh(xx / dist, yy / dist, np.log(model_int_map), shading='auto', rasterized=True,
         cmap='PuBuGn', vmin = np.nanmin(model_int_map), vmax = np.nanmax(model_int_map))
        plt.colorbar(a,ax = axes)
        plt.show()
        plt.close()

    plot_cube = False

    if plot_cube:
        canvas = AstroCanvas((6,7),(0,0), imagegrid=True)
        canvas.channelmaps(cube, contour=True, color=False,
                           #coord_center='04h04m43.07s 26d18m56.30s',
                           #nskip=2,
                           imscale = [-7, 7, -7, 7],
            clevels = np.array([-3, 3.,6.,9.,12.,15.])*7e-3)
        for i, im in enumerate(modelcube):      #   Plotting model as image as raster
            if i < len(canvas.axes):
                ax = canvas.axes[i]
                ax.pcolormesh(xx / dist, yy / dist, im, shading='auto', rasterized=True,
                    vmin = vmin, vmax = vmax, cmap='PuBuGn')

            else:
                break
        plt.show()

    plot_PV = False
    if plot_PV:

        pv_obs = Imfits(f_PV, pv=True)

        print("Shape of observed pv", np.shape(pv_obs.data))
        rms_pv = pv_obs.estimate_noise()

        
        canvas = AstroCanvas((1,1))

        pv_plot = canvas.pvdiagram(pv_obs,
                    vrel = True,
                    color = False,
                    ccolor = 'green',
                    #cmap = 'inferno',
                    vmin = -2.0,
                    vmax = 14.0,
                    contour = True,
                    clip = 0.0000000,
                    #ylim = [-8.5,6.5],
                    clevels = np.array([3,7,10,15,25,35,45])*rms_pv,
                    x_offset = True, # If true, offset (radial distance from star) will be the x axis
                    vsys = 7.3, # systemic velocity in kmps
                    ln_var = True, # plot vertical center (systemic velocity)
                    ln_hor = True, # plot horizontal center (zero offset)
                    #cbaroptions = ('right', '3%', '3%'),
                    #cbarlabel = r'(Jy beam$^{-1})$',
                    colorbar = False 
                    )
        
        X, Y = np.meshgrid(xx[0,:]/(dist), v-7.4)
        ax = canvas.axes[0]
        

        pv_model[pv_model<0] = 1.e-18
        vmin = np.nanmin(pv_model)
        vmax = np.nanmax(pv_model)


        a = ax.pcolormesh(X, Y, pv_model, shading='auto', cmap='inferno', rasterized=True,
                    vmin = vmin, vmax = vmax*0.5)
        
        plt.colorbar(a, ax=ax)
        plt.show()


if __name__ == '__main__':
    main()
