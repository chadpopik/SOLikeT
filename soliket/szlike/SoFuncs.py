import sys
import numpy as np
from scipy.interpolate import interp2d,interp1d
homepath = "/global/homes/c/cpopik/"

sys.path.append(homepath+"Packages/SOLikeT/")
from soliket.szlike import cosmo, gnfw, projection_functions
from soliket.szlike.projection_functions import kpc_cgs,C_CGS,ME_CGS,MP_CGS,sr2sqarcmin,XH
from soliket.constants import (MPC2CM,C_M_S,h_Planck,k_Boltzmann,electron_mass_kg,proton_mass_kg,hydrogen_fraction,T_CMB,ST_CGS,MSUN_CGS,G_CGS)


class SOinfo:
    def __init__(self, override=False):
        self.params= {
            "Omega_m": 0.25,
            "Omega_b": 0.044,
            "hh": 0.7,
            "Omega_L": 0.75,
            "rhoc_0": 2.77525e2,
            "C_OVER_HUBBLE": 2997.9,
            "XH": 0.76, #hydrogen fraction
            "v_rms": 1.06e-3, #v_rms/c
            
            "gnfw_P0" : 4.67,
            "gnfw_bt_tsz": 4.01,
            "gnfw_A2h_tsz": 1,
            
            "redshift": 0.55,
            "mass_halo_mean_Msol" : 10**11.2,
            "frequency_GHz": 150.
        }
                
        self.files = {
            "sz_data_file": homepath+"emu4d_match_ACT_profiles.txt",
            "beam_file": homepath+"Packages/SOLikeT/soliket/szlike/beam_f150_daynight.txt",
            "transform_type": "Hankel",
            "beam_response": homepath+"Packages/SOLikeT/soliket/szlike/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt",
            "cov_tsz_file": homepath+"Packages/SOLikeT/soliket/szlike/cov_diskring_tsz_varweight_bootstrap.txt",
            "twohalo_term": homepath+"Packages/SOLikeT/soliket/szlike/twohalo_cmass_average.txt"
        }
        
        if override is not False:
            assert isinstance(override, dict), "Param override must be dict"
            for key in override.keys(): self.params[key] = override[key]
        
        self.redshift = self.params["redshift"]
        self.mass_halo_mean_Msol = self.params["mass_halo_mean_Msol"]
        self.frequency_GHz = self.params["frequency_GHz"]
        
        self.thta_arc, self.tsz_data = np.loadtxt(self.files["sz_data_file"], usecols=(0, 6), unpack=True)
        # self.tsz_data /= 3282.8 * 60.0**2  # units muK*sr
        self.cov_tsz = np.loadtxt(self.files["cov_tsz_file"])  * 3282.8 * 60.0**2 # units muK*arcmin**2
        
        self.theta = [self.params.get(p) for p in ["gnfw_P0", "gnfw_bt_tsz", "gnfw_A2h_tsz"]]
        self.Rs = cosmo.AngDist(self.redshift, self) * np.arctan(np.radians(self.thta_arc / 60.0))

            
    def get_param(self, param):
            return self.params[param]
        
    def Pth_gnfw_input(self, rs):
        return [rs, self.mass_halo_mean_Msol, self.redshift, self.theta, self.files["twohalo_term"]]
    
    def Pth_gnfw1h_input(self, rs):
        return [rs, self.mass_halo_mean_Msol, self.redshift, self.theta[0:2]]

    def Pth_gnfw2h_input(self, rs):
        return [rs, self.theta[2], self.files["twohalo_term"]]
        
    def project_tsz_input(self, ii):
        return [self.thta_arc[ii], self.mass_halo_mean_Msol, self.redshift, self.frequency_GHz, 
                self.files["beam_file"], self.files["transform_type"], self.theta,self.files["beam_response"],self.files["twohalo_term"]]
    
    
    
def projected_tsz(provider):
    pth = np.zeros(len(provider.thta_arc))
    for ii in range(len(provider.thta_arc)):
        pth[ii] = projection_functions.project_tsz(*provider.project_tsz_input(ii),provider)
    return pth



def project_tsz_custom(tht,M,z,nu,beam_txt,transform_type,model_params,beam_response,twohalo_term,provider,rs,Pths):
    disc_fac = np.sqrt(2)
    NNR = 100
    resolution_factor = 3.5
    NNR2 = resolution_factor * NNR
    AngDis = cosmo.AngDist(z, provider)

    baseMap, r, r_max = projection_functions.radius_definition(transform_type)

    r_use = AngDis * np.arctan(np.radians(tht / 60.0))
    r_use2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))
    r_ext = AngDis * np.arctan(r_max)  # total profile
    r_ext2 = r_ext

    rad = np.logspace(-3, 1, 200)  # Mpc
    rad2 = rad

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads
    dtht_use = np.arctan(r_use / AngDis) / NNR
    dtht2_use = np.arctan(r_use2 / AngDis) / NNR

    thta_use = (np.arange(NNR) + 1.0) * dtht_use
    thta2_use = (np.arange(NNR) + 1.0) * dtht2_use

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor
    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad**2 + thta_smooth**2 * AngDis**2)
    rint2 = np.sqrt(rad2**2 + thta2_smooth**2 * AngDis**2)

    Pth_inter = interp1d(rs, Pths,bounds_error=True)

    Pth2D = (2 * np.trapz(Pth_inter(rint), x=rad * kpc_cgs,axis=1,)* 1e3)
    Pth2D2 = (2* np.trapz(Pth_inter(rint2),x=rad2 * kpc_cgs,axis=1,)* 1e3)

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor

    if transform_type == "FFT":
        Pth2D_beam = projection_functions.convolve_FFT(r, thta_smooth, Pth2D, beam_txt, baseMap, thta_use, beam_response)
        Pth2D2_beam = projection_functions.convolve_FFT(r, thta2_smooth, Pth2D2, beam_txt, baseMap, thta2_use, beam_response)
    elif transform_type == "Hankel":
        Pth2D_beam = projection_functions.convolve_Hankel(thta_smooth, Pth2D, beam_txt, thta_use, beam_response)
        Pth2D2_beam = projection_functions.convolve_Hankel(thta2_smooth, Pth2D2, beam_txt, thta2_use, beam_response)

    sig_p = 2.0 * np.pi * dtht_use * np.sum(thta_use * Pth2D_beam)
    sig2_p = 2.0 * np.pi * dtht2_use * np.sum(thta2_use * Pth2D2_beam)
    sig_all_p_beam = ((2 * sig_p - sig2_p)* ST_CGS/ (ME_CGS * C_CGS**2)* ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))* 1e6)

    sig_all_p_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_p_beam



def projected_tsz_custom(provider, xs, Pth):
    pth = np.zeros(len(provider.thta_arc))
    for ii in range(len(provider.thta_arc)):
        pth[ii] = project_tsz_custom(*provider.project_tsz_input(ii), provider, xs, Pth)
    return pth








def project_Pth_custom_mis(tht,M,z,nu,beam_txt,transform_type,model_params,beam_response,twohalo_term,provider,
                           rs,Pths, tauRg, f_mis):
    disc_fac = np.sqrt(2)
    NNR = 100
    resolution_factor = 3.5
    NNR2 = resolution_factor * NNR
    AngDis = cosmo.AngDist(z, provider)

    baseMap, r, r_max = projection_functions.radius_definition(transform_type)

    r_use = AngDis * np.arctan(np.radians(tht / 60.0))
    r_use2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))
    r_ext = AngDis * np.arctan(r_max)  # total profile
    r_ext2 = r_ext

    rad = np.geomspace(1e-4, 5e1, 50)  # Mpc
    rad2 = rad

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads
    dtht_use = np.arctan(r_use / AngDis) / NNR
    dtht2_use = np.arctan(r_use2 / AngDis) / NNR

    thta_use = (np.arange(NNR) + 1.0) * dtht_use
    thta2_use = (np.arange(NNR) + 1.0) * dtht2_use

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor
    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad**2 + thta_smooth**2 * AngDis**2)
    rint2 = np.sqrt(rad2**2 + thta2_smooth**2 * AngDis**2)

    def Pth_inter(rsinp):
        return np.interp(rsinp, rs, Pths, right=0)

    Pth2D = (2 * np.trapz(Pth_inter(rint), x=rad * kpc_cgs,axis=1,)* 1e3)
    Pth2D2 = (2* np.trapz(Pth_inter(rint2),x=rad2 * kpc_cgs,axis=1,)* 1e3)

    def Pth2Dfunc(Rs, Pthfunc):
        rint = np.sqrt(Rs[..., None]**2 + rad**2)
        return 2*np.trapz(Pthfunc(rint), x=rad* kpc_cgs, axis=-1,)* 1e3

    phis = np.linspace(0, 2*np.pi, 50)
    R_mis = np.geomspace(2e-4, 2e1, 100)
    rs_theta, rs_theta2  = thta_smooth[:, 0]*AngDis, thta2_smooth[:, 0]*AngDis
    Rints = np.sqrt(R_mis[None, ..., None]**2+rs_theta[..., None, None]**2 \
                    +2*R_mis[None, ..., None]*rs_theta[..., None, None]*np.cos(phis))
    Rints2 = np.sqrt(R_mis[None, ..., None]**2+rs_theta2[..., None, None]**2 \
                    +2*R_mis[None, ..., None]*rs_theta2[..., None, None]*np.cos(phis))
    Pth2D_R_Rmis = np.trapz(Pth2Dfunc(Rints, Pth_inter), phis, axis=-1)/(2*np.pi)
    Pth2D_R_Rmis2 = np.trapz(Pth2Dfunc(Rints2, Pth_inter), phis, axis=-1)/(2*np.pi)
    
    gamma = lambda r_mis: r_mis/tauRg**2 * np.exp(-r_mis/tauRg)
    Pth2D_mis_R = np.trapz(gamma(R_mis) * Pth2D_R_Rmis, R_mis)
    Pth2D2_mis_R = np.trapz(gamma(R_mis) * Pth2D_R_Rmis2, R_mis)
    
    Pth2D_mis = (1-f_mis)*Pth2D+f_mis*Pth2D_mis_R
    Pth2D2_mis = (1-f_mis)*Pth2D2+f_mis*Pth2D2_mis_R

    return {"Rs":thta_smooth*AngDis, "R2s":thta2_smooth*AngDis, "Pth2D": Pth2D, "Pth2D2": Pth2D2, "Pth2D_mis": Pth2D_mis, "Pth2D2_mis": Pth2D2_mis}



def project_tsz_custom_mis(tht,M,z,nu,beam_txt,transform_type,model_params,beam_response,twohalo_term,provider,
                           Pthsout):
    disc_fac = np.sqrt(2)
    NNR = 100
    resolution_factor = 3.5
    NNR2 = resolution_factor * NNR
    AngDis = cosmo.AngDist(z, provider)

    baseMap, r, r_max = projection_functions.radius_definition(transform_type)

    r_use = AngDis * np.arctan(np.radians(tht / 60.0))
    r_use2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))
    r_ext = AngDis * np.arctan(r_max)  # total profile
    r_ext2 = r_ext

    rad = np.geomspace(1e-4, 5e1, 50)  # Mpc
    rad2 = rad

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads
    dtht_use = np.arctan(r_use / AngDis) / NNR
    dtht2_use = np.arctan(r_use2 / AngDis) / NNR

    thta_use = (np.arange(NNR) + 1.0) * dtht_use
    thta2_use = (np.arange(NNR) + 1.0) * dtht2_use

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor
    
    if transform_type == "FFT":
        Pth2D_beam = projection_functions.convolve_FFT(r, thta_smooth, Pthsout['Pth2D_mis'], beam_txt, baseMap, thta_use, beam_response)
        Pth2D2_beam = projection_functions.convolve_FFT(r, thta2_smooth, Pthsout['Pth2D2_mis'], beam_txt, baseMap, thta2_use, beam_response)
    elif transform_type == "Hankel":
        Pth2D_beam = projection_functions.convolve_Hankel(thta_smooth, Pthsout['Pth2D_mis'], beam_txt, thta_use, beam_response)
        Pth2D2_beam = projection_functions.convolve_Hankel(thta2_smooth, Pthsout['Pth2D2_mis'], beam_txt, thta2_use, beam_response)

    sig_p = 2.0 * np.pi * dtht_use * np.sum(thta_use * Pth2D_beam)
    sig2_p = 2.0 * np.pi * dtht2_use * np.sum(thta2_use * Pth2D2_beam)
    sig_all_p_beam = ((2 * sig_p - sig2_p)* ST_CGS/ (ME_CGS * C_CGS**2)* ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))* 1e6)

    sig_all_p_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_p_beam



def projected_tsz_custom_mis(provider, xs, Pth, tauRg, f_mis):
    Pthsout = project_Pth_custom_mis(*provider.project_tsz_input(0), provider, xs, Pth, tauRg, f_mis)
    pth = np.zeros(len(provider.thta_arc))
    for ii in range(len(provider.thta_arc)):
        pth[ii] = project_tsz_custom_mis(*provider.project_tsz_input(ii), provider, Pthsout)
    return pth
