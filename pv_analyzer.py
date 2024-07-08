import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import constants, units
from scipy.interpolate import CubicSpline
from typing import Union, Literal

current_path = os.getcwd()
# Check if it is office computer or laptop and set path of imfits accordingly
# Office computer
if current_path.split(sep=":")[0].lower() == "d":
    sys.path.append("D:\L1489_IRS_ssp\imfits")
    from imfits.drawmaps import AstroCanvas
    from imfits import Imfits
else:  # Laptop N
    sys.path.append("E:/Mihir_new/ASIAA-SSP/imfits/")
    from imfits.drawmaps import AstroCanvas
    from imfits import Imfits


class pv_analyze:
    """
    Analyzes a PV diagramm
    """

    G_grav = constants.G.value  # Gravitational constant (MKS)
    M_sun = constants.M_sun.value  # Solar mass (Kg)
    auTOm = units.au.to('m')            # 1 au (cm)


    def __init__(self, pv_path=None, is_Tb:bool=True, line_name:str="J_2_1",
                 obj_name:str = "L1489_irs", inclination:float = 73.0, PA:float = 69.0,
                 M_star:float = 1.6, distance_pc:float = 140.0, v_sys:float = 7.22):

        self.pv_path = pv_path
        self.is_Tb = is_Tb
        self.obj_name = obj_name
        self.inclination = inclination  # Inclination in degree
        self.PA = PA  # PA im degree
        self.M_star = M_star  # Mass of star in units of M_sun
        self.distance_pc = distance_pc  # Distance of object in parsec
        self.v_sys = v_sys  # Systemic velocity in kmps
        self.pv = None  # Imfits PV object
        self.rms = None  # Root mean squared error of PV
        self.pv_data = None  # 2D PV image data extracted from self.pv
        self.v_obs = None  # Observed velocities
        # Observed velocities converted to rotational velocities (v_rot = V_obs - Vsys)
        self.v_rot = None
        # Redshifted (positive) rotational velocities for points on curve, cropped till the extent of PV plot
        self.v_rot_redshifted = None
        # Blueshifted (positive) rotational velocities for points on curve, cropped till the extent of PV plot
        self.v_rot_blueshifted = None
        self.x_axis = None  # Observed x values (in arcec)
        # x coordinates of Keplerian or other Curve points in redshifted side (in arcsec)
        self.r_as_rs = None
        # x coordinates of Keplerian or other Curve points in blue shifted side (in arcsec)
        self.r_as_bs = None

    def read_pv(self, print_details=True):
        """
        Reads PV data and sets up attributes of velocities.
        parameters:
        -----------
        print_details:  `bool`, If True, prints name of imfits PV object and rms in units of Jy/beam and units of Kelvin if `is_Tb` = True
        """

        # Read data and convert intensity to Tb units
        pv = Imfits(self.pv_path, pv=True)
        self.rms = pv.estimate_noise()

        if print_details:
            print(f" rms = {pv.estimate_noise() :.2e} Jy beam$^-1$")

        if self.is_Tb:
            pv.convert_units(conversion="IvtoTb")
            self.rms = pv.estimate_noise()
            print(rf" rms = {self.rms :.2f} K")

        self.pv = pv  # PV imfits object

        self.v_obs = pv.vaxis
        self.v_rot = np.array(self.v_obs - self.v_sys)
        self.v_rot_redshifted = self.v_rot[self.v_rot > 0]
        self.v_rot_blueshifted = self.v_rot[self.v_rot <= 0]
        self.x_axis = pv.xaxis

        return

    def _get_pixel_ridx_on_curve(self, x_data=None, x_curve=None):
        # Find x indices of pixels in which the curve points fall
        _idx = [np.abs(x_data - b).argmin() for b in x_curve]
        return _idx

    def _get_pixel_vidx_on_curve(self, v_obs=None, v_rot=None, v_sys=None, v_tol=None):
        # Find velocity indices of rs and bs (we try to find v_obs which are equal to 
        # v_rot_bs + v_sys with a tolerance of 0.05 kmps as velocity channels are seperated by 0.2 kmps) )
        v_sum = v_rot + v_sys
        _idx = [np.abs(v_obs - v).argmin() for v in v_sum]
        # _idx = np.where(np.abs(v_obs[:, None] - (v_rot + v_sys)) <= v_tol)[0]
        return _idx

    def get_tb_on_curve(
        self,
        mode:Union[str, Literal['func', 'vals']] = 'func',
        curve_function:list = None,
        get_surrounding_pix:bool = False,
        get_pix_along: Union[str, Literal['r', 'v']] = 'r',
        num_pix:int = 4,
        return_avg:bool = True,
        return_interp_mean:bool = True,
        return_coords:bool = True,
        cf_kwargs = None,
        ) -> dict[np.ndarray, np.ndarray]:
        """
        Returns a pandas dataframe where each column is a temperature at pixels on/surrounding 
        the curve points in each velocity channel

        Parameters:
        -----------
        mode                :   `str`, {'func', 'vals'}, whether the curve function is a funcrion or list of rs and vs
        curve_function      :   `callable` or list, optional, default: None,
                                A function f(v_rotational) -> r, typically an inverse of kepler velocity function.
                                It provides angular distance `r` at which one should expect the emission comming from material rotating at
                                keplerian velocities. More generally, a function that returns r coordinates in units of AU
                                wrt center of disk for given velocity channels.
                                Default - a inverse kepler function GM/((v_rot*sin(i))**2) --> r

                                If mode = 'vals', then a list of lists, specifying r and v values in following order
                                [r_rs (au)] , [v_rs (kmps)], [r_bs (au)], [v_bs (kmps)]

        return_interp_mean  :   `bool`, If true then interpolates (cubic spline) flux values along pixels in specified direction (according to value in 
                                `get_pix_along`) and returns flux at location of points using the interpolated curve.
        get_surrounding_pix :   `bool`, optional, If True then returns dataframe with Tb values in surrounding `num_pix` pixels
        get_pix_along       :   `str`, `r` | `v`, optional, whether to get pixels along fix r or fix v
        num_pix             :   `int` or `None`, optional, default: None, number of pixels around the pixel on which curve points fall for which Tb values are to be returned
        return_coords       :   `bool`, If true, returning dataframe will have coordinates (r and v vals)
        cf_kwargs           :    dict, optional,
                                 dictionary of other arguments needed to evaluet the `cost_function()`

        -------
        Returns
        -------

        Tb_cube             :   `pandas.DataFrame`,

                                With columns "Tb_on_pt_rs" and "Tb_on_pt_bs" if `get_surrounding_pix` is `False` or 
                                with columns "Tb_sur_pt_rs" and "Tb_sur_pt_bs", if `get_surrounding_pix` is `True`

                                - column1: numpy.ndarray
                                - column2: numpy.ndarray

                                if return_coords = True,

                                - column3 ("r_rs"): numpy.ndarray
                                - column4 ("r_bs"): numpy.ndarray
                                - column5 ("v_rs"): numpy.ndarray
                                - column6 ("v_bs"): numpy.ndarray

        """

        # Get radial coordinates of point on given curve in units of AU
        # if function r = f(v) of the curve is supplied
        if mode.lower() == 'func':
        
            if curve_function is not None:
                # using a customized function to find r_au corresponding to observed velocity channels
                r_au_rs = curve_function(self.v_rot_redshifted, **cf_kwargs)
                r_au_bs = -1.0 * curve_function(self.v_rot_blueshifted, **cf_kwargs)

            else:
                # use 2D keplerian velocity profile
                r_au_rs = (self.G_grav*self.M_star*self.M_sun/(self.v_rot_redshifted* 1.0e3/ np.sin(np.deg2rad(self.inclination)))** 2) / self.auTOm
                r_au_bs = -1.0 * (self.G_grav* self.M_star* self.M_sun/ (self.v_rot_blueshifted* 1.0e3/ np.sin(np.deg2rad(self.inclination)))** 2) / self.auTOm
        
        # if coordinates of curve is supplied
        elif mode.lower() == 'vals':

            r_au_rs, self.v_rot_redshifted  = curve_function[0], curve_function[1]
            r_au_bs, self.v_rot_blueshifted = curve_function[2], curve_function[3]
        
        else:
            print("Please specify correct mode.")
            return 0
        
        # Convert radial coordinates to arcsec
        # (for points in redshifted side) radial distance from star in arcsec
        self.r_as_rs = r_au_rs / self.distance_pc
        # (for points in blueshifted side) radial distance from star in arcsec
        self.r_as_bs = r_au_bs / self.distance_pc
        

        #print("max x axis",self.x_axis.max())
        #print("min x axis", self.x_axis.min())
        # dropping r larger then rmax of pv image and corresponding Vs as well
        self.v_rot_redshifted, self.r_as_rs = (
            self.v_rot_redshifted[self.r_as_rs <= self.x_axis.max()],
            self.r_as_rs[self.r_as_rs <= self.x_axis.max()],
        )
        self.v_rot_blueshifted, self.r_as_bs = (
            self.v_rot_blueshifted[self.r_as_bs >= self.x_axis.min()],
            self.r_as_bs[self.r_as_bs >= self.x_axis.min()],
        )

        # Find x indices of pixels in which the curve points fall
        self.r_as_rs_idx = self._get_pixel_ridx_on_curve(
            x_data=self.x_axis, x_curve=self.r_as_rs
        )
        self.r_as_bs_idx = self._get_pixel_ridx_on_curve(
            x_data=self.x_axis, x_curve=self.r_as_bs
        )

        self.vidx_rs = self._get_pixel_vidx_on_curve(
            v_obs=self.v_obs, v_rot=self.v_rot_redshifted, v_sys=self.v_sys, v_tol=0.005
        )
        self.vidx_bs = self._get_pixel_vidx_on_curve(
            v_obs=self.v_obs,
            v_rot=self.v_rot_blueshifted,
            v_sys=self.v_sys,
            v_tol=0.005,
        )

        # self.pv.data has a redundant z axis of length one. 
        # Since we are considering that image have only one stokes axis. Let's get rid of that
        self.pv_data = np.squeeze(self.pv.data) # Select stokes I

        if get_surrounding_pix:

            # NOTE: not yet tested properly 
            if num_pix is None or not isinstance(num_pix, int):
                raise TypeError(
                    "`num_pix` has to be an integer to get the Tb from surrounding pixels"
                )

            tb_sur_pt_rs = []
            tb_sur_pt_bs = []

            # Gather Tb values along blueshifted side
            for idx, (pt_v_idx, pt_r_idx) in enumerate(zip(self.vidx_bs, self.r_as_bs_idx)):

                # else:
                tb_surroundings = []
                tb_sur_x_points = []

                for i in range(-num_pix, num_pix):

                    if get_pix_along == 'r':
                        if pt_r_idx + i < 0 or pt_r_idx + i >= len(self.x_axis):     # takes care of cases when our point is on edge of image
                            continue
                        else:
                            tb_surroundings.append(float(self.pv_data[pt_v_idx, pt_r_idx + i]))
                            tb_sur_x_points.append(float(self.x_axis[pt_r_idx + i]))
                    elif get_pix_along == 'v':
                        if pt_v_idx + i < 0 or pt_v_idx + i >= len(self.v_obs):     # takes care of cases when our point is on edge of image
                            continue
                        else:
                            tb_surroundings.append(float(self.pv_data[pt_v_idx+i, pt_r_idx]))
                            tb_sur_x_points.append(float(self.v_obs[pt_v_idx+i]-self.v_sys))
                    
                    else:
                        raise ValueError("Only 'r' or 'v' are correct values for `get_pix_along` argument.")
                    
                if return_avg:

                    # interpolate Tb values from surrounding pixel values
                    if return_interp_mean:

                        local_flux_fun = CubicSpline(tb_sur_x_points, tb_surroundings)
                        
                        if get_pix_along == 'r':
                            tb_sur_pt_bs.append(local_flux_fun(self.r_as_bs[idx]).item())
                        else:
                            tb_sur_pt_bs.append(local_flux_fun(self.v_rot_blueshifted[idx]).item())
                    
                    # just return the mean from pixel values
                    else:
                        tb_sur_pt_bs.append(np.mean(tb_surroundings))
                else:
                    tb_sur_pt_bs.append(tb_surroundings)
            
            # Gather Tb values along redshifted side
            for idx, (pt_v_idx, pt_r_idx) in enumerate(zip(self.vidx_rs, self.r_as_rs_idx)):

                tb_surroundings = []
                tb_sur_x_points = []

                for i in range(-num_pix, num_pix):
                    
                    if get_pix_along == 'r':
                        if pt_r_idx + i < 0 or pt_r_idx + i >= len(self.x_axis):     # takes care of cases when our point is on edge of image
                            continue
                        else:
                            tb_surroundings.append(float(self.pv_data[pt_v_idx, pt_r_idx + i]))
                            tb_sur_x_points.append(float(self.x_axis[pt_r_idx + i]))
                    elif get_pix_along == 'v':
                        if pt_v_idx + i < 0 or pt_v_idx + i >= len(self.v_obs):     # takes care of cases when our point is on edge of image
                            continue
                        else:
                            tb_surroundings.append(float(self.pv_data[pt_v_idx+i, pt_r_idx]))
                            tb_sur_x_points.append(float(self.v_obs[pt_v_idx+i]-self.v_sys))
                    else:
                        raise ValueError("Only 'r' or 'v' are correct values for `get_pix_along` argument.")
                
                if return_avg:
                    
                    if return_interp_mean:   
              
                        local_flux_fun = CubicSpline(tb_sur_x_points, tb_surroundings)               
                        if get_pix_along == 'r':
                            tb_sur_pt_rs.append(local_flux_fun(self.r_as_rs[idx]).item())
                        else:
                            tb_sur_pt_rs.append(local_flux_fun(self.v_rot_redshifted[idx]).item())                  
                    else:
                        tb_sur_pt_rs.append(np.mean(tb_surroundings))
                    
                else:
                    tb_sur_pt_bs.append(tb_surroundings)

            data_cube = {"Tb_sur_pt_rs": tb_sur_pt_rs, "Tb_sur_pt_bs": tb_sur_pt_bs}

        else:       # Getting values on the pixel only

            tb_on_point_rs = []
            tb_on_point_bs = []

            for pt_v_idx, pt_r_idx in zip(self.vidx_rs, self.r_as_rs_idx):
                tb_on_point_rs.append(float(self.pv_data[pt_v_idx, pt_r_idx]))

            for pt_v_idx, pt_r_idx in zip(self.vidx_bs, self.r_as_bs_idx):
                tb_on_point_bs.append(float(self.pv_data[pt_v_idx, pt_r_idx]))

            data_cube = {
                    "Tb_on_point_rs": tb_on_point_rs,
                    "Tb_on_point_bs": tb_on_point_bs
                    }
        
        if return_coords:
            data_cube["r_rs"] = self.r_as_rs
            data_cube["r_bs"] = self.r_as_bs
            data_cube["v_rs"] = self.v_rot_redshifted
            data_cube["v_bs"] = self.v_rot_blueshifted

        return data_cube

    def plot_pv(self, plot_curve: bool = False, base_ctr_lvl = None,
                cmap=None, cbarlim=None, **kwargs):
        """
        Plots PV diagram and overplots curve points if `plot_curve` is True. Returns the fig, axes object.
        """
        # Work in progress

        if base_ctr_lvl is not None:
            self.rms = base_ctr_lvl

        if cmap is None:
            cmap = 'inferno'
        
        if cbarlim is None:
            vmin = 0
            vmax = np.max(self.pv_data)
        
        else:
            vmin = cbarlim[0]
            vmax = cbarlim[1]

        canvas = AstroCanvas((1, 1))
        pv_plot = canvas.pvdiagram(
            self.pv,
            vrel=True,
            color=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            contour=True,
            clip=0.0000000,
            ylim=[-8.5, 6.5],
            clevels=np.array([3, 7, 10, 15, 25, 35, 45]) * self.rms,
            # If true, offset (radial distance from star) will be the x axis
            x_offset=True,
            vsys=self.v_sys,  # systemic velocity
            ccolor='grey',
            # plot vertical center (systemic velocity)
            ln_var=True,
            # plot horizontal center (zero offset)
            ln_hor=True,
            cbaroptions=("right", "3%", "3%"),
            cbarlabel=r"Jy (beam$^{-1}$)",
            colorbar=True,
        )

        if plot_curve:
            pv_plot[0].scatter(self.r_as_rs,self.v_rot_redshifted, c='red')
            pv_plot[0].scatter(self.r_as_bs,self.v_rot_blueshifted,  c ='blue')
        
        
        # mark extracted pixels
            pv_plot[0].scatter(self.x_axis[self.r_as_rs_idx],self.v_obs[self.vidx_rs] - self.v_sys,  marker = 'x', color = 'k')
            pv_plot[0].scatter(self.x_axis[self.r_as_bs_idx],self.v_obs[self.vidx_bs] - self.v_sys,  marker = 'x', color = 'white')

        return pv_plot