"""
Likelihood for SZ model
"""

import sys
sys.path.append('/global/homes/c/cpopik/')
from Basics import *

from ..gaussian import GaussianData, GaussianLikelihood
from .projection_functions import project_ksz, project_tsz, project_obb

from typing import Optional, Sequence

sys.path.append('/global/homes/c/cpopik/Git/Capybara/')
from Profiles import Pthoverldel, weight, twohalo
from Project import project_tsz_Hankel
import SHMRs


class SZLikelihood2(GaussianLikelihood):
    def initialize(self):
        self.thetas, self.y, self.cov = self._get_data()
        self.data = GaussianData("SZModel", self.thetas, self.y, self.cov)

        print('Using szlike2')
        # This is just setting fixed cosmology parameters as attributes to access them easily
        cospar = {k: v["value"] for k, v in self.params.items() if isinstance(v, dict) and "value" in v}
        guesspar = {k: v["ref"] for k, v in self.params.items() if isinstance(v, dict) and "ref" in v}
        guesspar = cospar | guesspar
        self.h = self.params["hh"]['value']
        self.z = self.params["redshift"]['value']
        self.M = self.params["mass_halo_mean_Msol"]['value']
        self.Om = self.params["Omega_m"]['value']
        self.Ob = self.params["Omega_b"]['value']
        self.OL = self.params["Omega_L"]['value']
        self.XH = self.params["XH"]['value']

        # Everything here should be a function of the halo model, which can be customized to switch the halo model to be used
        
        # Just using astropy to get simple fucntions like rhocz and angdist
        cosmology = astropy.cosmology.LambdaCDM(H0=self.h*100, Tcmb0=2.726, Om0=self.Om, Ode0=self.OL, Ob0=self.Ob)
        self.rhoczfunc = lambda z: cosmology.critical_density(z).to(u.Msun/u.Mpc**3).value
        # print(f"Critical density at z=0.45 {self.rhoczfunc(0.45)} Msun/Mpc^3")
        self.angdistfunc = lambda z: cosmology.angular_diameter_distance(z).value
        # print(f"Angular distance z=0.45 {self.angdistfunc(0.45)} Mpc")

        # print(f"Pth/pdel at M200c=1e13, z=0.45, guess params {Pthoverldel(np.logspace(-1,0), 1e13, 0.45, **guesspar)} unitless")

        self.r200cfunc = lambda m200c, z: (m200c/(4/3*np.pi*200*self.rhoczfunc(z)))**(1/3)
        # print(f"r200c at z=0.45, M200c=1e13 {self.r200cfunc(1e13, 0.45)}")

        self.p200cfunc = lambda m200c, z: self.Ob/self.Om * 2.0*(self.XH+1.0)/(5.0*self.XH+3.0) * c.G.to(u.Mpc**3/u.Msun/u.s**2).value*m200c*200*self.rhoczfunc(z)/(2*self.r200cfunc(m200c, z))
        # print(f"p200c at z=0.45, M200c=1e13 {self.p200cfunc(1e13, 0.45)*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)}")

        rsdata, nada, pth2hdata = np.loadtxt(self.twohalo_term, unpack=True)
        # print(f"pth_2h {pth2hdata}")
        # print(f"pth_1h {Pthoverldel(rsdata/self.r200cfunc(1e13, 0.45), 1e13, 0.45, **guesspar)*self.p200cfunc(1e13, 0.45)*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)}")


        dndzdata = pd.read_csv(self.redshift_dist_file, sep=" ", skiprows=1, names=pd.read_csv(self.redshift_dist_file, sep=" ").columns[1:])
        self.zs, dndz = dndzdata.zmin.values, dndzdata.bin_1_combined.values

        dndmdata = pd.read_csv(self.mass_dist_file, sep=' ', names=['Mstar', 'n', 'err'])
        self.mstars, dndm = 10**dndmdata.Mstar.values, dndmdata.n.values

        # CUSTOM: Function that converts stellar masses into M200c of halos, this will need to change based on the curvey
        self.M200cfunc = lambda m, z: np.interp(m,
                           SHMRs.Gao2023().SHMR('ELG_Auto')(np.logspace(10, 17.5, 1000)),
                           np.logspace(10, 17.5, 1000)) + 0*z
        self.m200cs = self.m200cs = self.M200cfunc(self.mstars, 0)
        
        

        p200cs = self.p200cfunc(self.m200cs[None, :], self.zs[:, None])
        r200cs = self.r200cfunc(self.m200cs[None, :], self.zs[:, None])
        hmf = dndm[None, :]*dndz[:, None]
        normfac = np.trapz(np.trapz(hmf, self.m200cs), self.zs)


        self.rs=np.logspace(-1, 1, 100)
        pth2h = np.interp(self.rs, rsdata, pth2hdata)
        def pth2hfunc(A2h, **kwargs):
            return A2h*pth2h
            
        self.pth1hfunc = lambda **kwargs: p200cs*Pthoverldel(self.rs[:, None, None], self.m200cs[None, None, :], self.zs[None, :, None], **kwargs)

        # print(f"pth1h {self.pth1hfunc(**guesspar)[:, 50, 50]*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)}")
        # print(f"pth2h {pth2hfunc(**guesspar)}")

        self.pthtotfunc = lambda **kwargs: np.trapz(np.trapz(hmf*self.pth1hfunc(**kwargs), self.m200cs), self.zs)/normfac*(u.Msun/u.Mpc/u.s**2).to(u.g/u.cm/u.s**2)+pth2hfunc(**kwargs)

        print(f"pthtot {self.pthtotfunc(**guesspar)}")

        self.convolve_with_beamfunc = project_tsz_Hankel(thetas=self.thetas, AngDist=self.angdistfunc(self.z), beamfile=self.beam_file, responsefile=self.beam_response)

        print(f"final sig {self.convolve_with_beamfunc(self.rs, self.pthtotfunc(**guesspar))}")

        print(f"data {self.tsz_data}")

        

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)



class TSZLikelihood2(SZLikelihood2):  # this is for GNFW model
    def _get_data(self, **params_values):
        thta_arc, tsz_data = np.loadtxt(self.sz_data_file, usecols=(0, 1), unpack=True)
        cov_tsz = np.loadtxt(self.cov_tsz_file)  # units muK*sr

        self.thta_arc = thta_arc
        self.tsz_data = tsz_data*u.sr.to(u.arcmin**2)  # arcmin^2
        self.cov = cov_tsz*u.sr.to(u.arcmin**2)**2  # arcmin^4
        return self.thta_arc, self.tsz_data, self.cov

    def _get_theory(self, **params_values):
        pth = self.pthtotfunc(**params_values)
        sig = self.convolve_with_beamfunc(self.rs, pth)
        return sig