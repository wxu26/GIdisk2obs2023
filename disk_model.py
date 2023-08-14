from util import *
from interp_table import InterpTable

"""
=======================================================================
Disk model class
=======================================================================
"""
class DiskModel:
    """
    Parametrized disk model for generating radial porfiles of disk
    properties and flux density at given wavelengths.

    Attributes:
      (all in cgs)
      M: total mass
      Mstar: stellar mass
      Mdot: accretion rate
      Rd: disk size
      Q: Toomre Q
      
      R: radius grid (R[0]=0)
      Sigma, T_mid, tau_p_mid, tau_r_mid: radial profile at R[1:]
      MR: M(<R) profile at R[1:]
    """
    def __init__(self, opacity_table, opacity_table_lam, disk_property_table):
        self.M = 1*Msun
        self.Mstar = 0.5*Msun
        self.Mdot = 1e-5*Msun/yr
        self.Rd = 40*au
        self.Q10 = 1.5 # Q at 10au
        self.Qs = 0. # radial slope of Q
        self.amax10 = 0.1 # amax at 10au
        self.amaxs = 0. # radial slope of amax
        self.q = 3.5 # slope of grain size distribution
        self.T0 = 0 # floor temperature
        self.opacity_table = opacity_table # interpolation table
        self.opacity_table_lam = opacity_table_lam # interpolation table
        self.disk_property_table = disk_property_table # interpolation table
        return
    def solve_local_disk_properties(self, Omega, kappa, r):
        Mdot = self.Mdot
        Q = self.Q10 * (r/10/au)**self.Qs
        amax = self.amax10 * (r/10/au)**self.amaxs
        exp_q = np.exp(self.q)
        T_eff = (Mdot*Omega**2/(4*pi*sigma_SB) + self.T0**4)**(1/4)
        Sigma_cs_l = self.disk_property_table.interp_scalar(
            'log_Sigma_cs_l', loglog=True, 
            log_amax=amax, q=exp_q, log_T_eff=T_eff, x=1)
        Sigma_cs_r = self.disk_property_table.interp_scalar(
            'log_Sigma_cs_r', loglog=True, 
            log_amax=amax, q=exp_q, log_T_eff=T_eff, x=1)
        Sigma_cs = kappa/(pi*G*Q)
        if Sigma_cs < Sigma_cs_l:
            tau_p_mid, tau_r_mid, Sigma, T_mid = small_number, small_number, small_number, small_number
        elif Sigma_cs > Sigma_cs_r:
            tau_p_mid, tau_r_mid, Sigma, T_mid = small_number, small_number, small_number, small_number
        else:
            x = np.log(Sigma_cs/Sigma_cs_l) / np.log(Sigma_cs_r/Sigma_cs_l)        
            tau_p_mid = self.disk_property_table.interp_scalar(
                'log_tau_p', loglog=True, 
                log_amax=amax, q=exp_q, log_T_eff=T_eff, x=np.exp(x))
            tau_r_mid = self.disk_property_table.interp_scalar(
                'log_tau_r', loglog=True, 
                log_amax=amax, q=exp_q, log_T_eff=T_eff, x=np.exp(x))
            Sigma = self.disk_property_table.interp_scalar(
                'log_Sigma', loglog=True, 
                log_amax=amax, q=exp_q, log_T_eff=T_eff, x=np.exp(x))
            T_mid = self.disk_property_table.interp_scalar(
                'log_T_mid', loglog=True, 
                log_amax=amax, q=exp_q, log_T_eff=T_eff, x=np.exp(x))
            #in_bound = self.disk_property_table.check_in_bounds(
            #    loglog=True, log_amax=amax, q=exp_q, log_T_eff=T_eff, x=np.exp(x))
            #if not in_bound: print('out of bound at r[au] = ', r/au)
        return tau_p_mid, tau_r_mid, Sigma, T_mid
    def generate_disk_profile(
        self,
        N_R = 50, # R grid resolution
        N_itr=8, plot_itr=False,
        ):
        """
        Generate radial disk profile (Sigma, T_mid, tau_p_mid, tau_r_mid)

        Args:
          M, Mdot, Rd, Q: set to None to use current values
                          stored in self
        """
        # update parameters
        M = self.M
        Mdot = self.Mdot
        Rd = self.Rd
        # set up grid
        Rmin = min(0.05*au, Rd/N_R)
        R = np.concatenate(([0],np.logspace(np.log10(Rmin), np.log10(Rd), N_R)))
        Sigma = np.zeros(N_R+1)
        MR = M * np.ones(N_R)
        tau_p_mid, tau_r_mid, T_mid = np.zeros(N_R), np.zeros(N_R), np.zeros(N_R)
        # iteratively update disk profile
        for n in range(N_itr):
            # mass -> Omega, kappa
            MR = np.cumsum(pi*(R[1:]-R[:-1])*(Sigma[1:]*R[1:]+Sigma[:-1]*R[:-1]))
            MR = M + MR - MR[-1]
            MR = np.maximum(MR, 0.1*M)
            Omega = np.sqrt(G*MR/R[1:]**3)
            kappa = Omega*np.minimum(2,np.sqrt(1+2*pi*R[1:]*R[1:]*Sigma[1:]/MR))
            # update disk profile
            for i in range(N_R):
                tau_p_mid[i], tau_r_mid[i], Sigma[i+1], T_mid[i] = \
                self.solve_local_disk_properties(Omega[i], kappa[i], R[i+1])
            if plot_itr:
                plt.plot(R[1:], Sigma[1:])
        self.R = R # N_R+1
        self.MR = MR
        self.Mstar = MR[0]
        self.Sigma = Sigma[1:]
        self.T_mid = T_mid
        self.tau_r_mid = tau_r_mid
        self.tau_p_mid = tau_p_mid
        return
    def generate_observed_flux_single_wavelength_no_scattering(
        self, cosI, lam_obs, N_tau=50, tau_min=0.01,
        ):
        nu = c_light/lam_obs
        # construct tau grid
        tau_max = self.tau_r_mid*2
        tau_min = np.minimum(tau_max/N_tau,tau_min)
        tau_grid = np.logspace(np.log10(tau_min), np.log10(tau_max/2), N_tau) # first dimension: tau, second dimension: r
        tau_grid = np.concatenate((tau_grid, tau_max - tau_grid[-2::-1], [tau_max]), axis=0)
        dtau_grid = tau_grid*1
        dtau_grid[1:] = dtau_grid[1:]-dtau_grid[:-1]
        tau_p = self.tau_p_mid*2
        tau_r = self.tau_r_mid*2
        # get temperature profile
        T_4_over_T_mid_4 = (tau_grid*(1-tau_grid/tau_r) + 1/np.sqrt(3) + 1/(1.5*tau_p))/\
                           (0.25*tau_r + 1/np.sqrt(3) + 1/(1.5*tau_p))
        T_grid = T_4_over_T_mid_4**(1/4) * self.T_mid
        amax_grid = np.zeros_like(T_grid)
        amax_grid[:] = self.amax10 * (self.R[1:]/10/au)**self.amaxs
        # get tau_obs
        kappa_abs = self.opacity_table_lam.interp(
            'log_kappa_abs', loglog=True,
            log_T=T_grid, log_lam=lam_obs*np.ones_like(T_grid), log_amax=amax_grid, q=np.exp(self.q)*np.ones_like(T_grid))
        kappa_r = self.opacity_table.interp(
            'log_kappa_abs_r', loglog=True,
            log_T=T_grid, log_amax=amax_grid, q=np.exp(self.q)*np.ones_like(T_grid))
        dtau_obs_grid = dtau_grid * kappa_abs/kappa_r / cosI
        tau_obs_grid = np.cumsum(dtau_obs_grid, axis=0)
        # in the first verion of the paper, I included cosI for tau_obs_grid but not dtau_obs_grid.
        B_grid = B(T_grid,nu)
        I_obs = np.sum(B_grid*np.exp(-tau_obs_grid+dtau_obs_grid)*(1-np.exp(-dtau_obs_grid)), axis=0)        
        tau_obs = tau_obs_grid[-1] * cosI # ignore geometric factor
        return I_obs, tau_obs, tau_obs # this matches the output dimension of generate_observed_flux_single_wavelength()
    def generate_observed_flux_single_wavelength(
        self, cosI, lam_obs, dtau=0.2,
        ):
        n = len(self.T_mid)
        I_obs = np.zeros(n)
        tau_obs = np.zeros(n)
        tau_a_obs = np.zeros(n)
        for i in range(n):
            amax = self.amax10 * (self.R[i+1]/10/au)**self.amaxs
            f_kappa_r = lambda T: self.opacity_table.interp(
                'log_kappa_abs_r', loglog=True, log_T=T, log_amax=amax*np.ones_like(T), q=np.exp(self.q)*np.ones_like(T))
            f_kappa_obs = lambda T: self.opacity_table_lam.interp(
                'log_kappa_abs', loglog=True, log_T=T, log_lam=lam_obs*np.ones_like(T), log_amax=amax*np.ones_like(T), q=np.exp(self.q)*np.ones_like(T))
            f_kappa_s_obs = lambda T: self.opacity_table_lam.interp(
                'log_kappa_sca_eff', loglog=True, log_T=T, log_lam=lam_obs*np.ones_like(T), log_amax=amax*np.ones_like(T), q=np.exp(self.q)*np.ones_like(T))
            I_obs[i], tau_obs[i], tau_a_obs[i] = get_I_with_scattering(
                self.T_mid[i], self.tau_r_mid[i], self.tau_p_mid[i], cosI, lam_obs, f_kappa_r, f_kappa_obs, f_kappa_s_obs, dtau=dtau)
        return I_obs, tau_obs, tau_a_obs
    def set_lam_obs_list(self, lam_obs_list):
        """
        Set wavelengths of observation
        """
        self.lam_obs_list = lam_obs_list
        self.N_lam_obs = len(lam_obs_list)
    def generate_observed_flux(self, cosI, scattering=True, **kwargs):
        """
        Generate flux density at observed wavelengths
        """
        self.I_obs = []
        self.tau_obs = []
        self.tau_a_obs = []
        self.scattering = scattering
        for i in range(self.N_lam_obs):
            if scattering:
                I_obs, tau_obs, tau_a_obs = self.generate_observed_flux_single_wavelength(cosI, self.lam_obs_list[i], **kwargs)
            else:
                I_obs, tau_obs, tau_a_obs = self.generate_observed_flux_single_wavelength_no_scattering(cosI, self.lam_obs_list[i], **kwargs)
            self.I_obs.append(I_obs)
            self.tau_obs.append(tau_obs)
            self.tau_a_obs.append(tau_a_obs)
        return


"""
=======================================================================
Compute intensity with scattering
=======================================================================
"""
def generate_ddtau_matrix_for_J(tauf_grid):
    # generate an operator corresponding to d^2/dtau^2, which will be applied on J.
    # boundary conditions:
    # at tau=tau_mid (tauf_grid[-1]): dJ/dtau=0
    # at tau=0: J = 1/sqrt(3) * dJ/dtau
    n = len(tauf_grid)
    D2 = np.zeros((n,n))
    # middle portion
    dl = tauf_grid[1:-1]-tauf_grid[:-2]
    dr = tauf_grid[2:]-tauf_grid[1:-1]
    wl = 2/((dl+dr)*dl)
    wr = 2/((dl+dr)*dr)
    wc = -wl-wr
    i = np.arange(n-2, dtype='int')
    D2[i+1, i  ] = wl
    D2[i+1, i+1] = wc
    D2[i+1, i+2] = wr
    # tau=0 bdry (first row)
    d = tauf_grid[1]-tauf_grid[0]
    # J(-1) = J(1) - 2*d*dJdtau = J(1) - 2*d*sqrt(3)*J(0)
    D2[0,0] = -2/d**2 - 1/d**2 * 2*d*np.sqrt(3)
    D2[0,1] = 1/d**2 + 1/d**2
    # tau=tau_mid bdry (last row)
    d = tauf_grid[-1]-tauf_grid[-2]
    D2[-1,-1] = -2/d**2
    D2[-1,-2] = 2/d**2
    return D2
def solve_J(tauf_grid, B, omega):
    D2 = generate_ddtau_matrix_for_J(tauf_grid)
    LHS = 1/3*D2 - np.diag(1-omega)
    RHS = -(1-omega)*B
    try:
        J = np.linalg.solve(LHS, RHS)
    except:
        J = np.ones_like(RHS)*np.nan
    return J
def generate_tauf_grid(tau_mid, dtau=0.2, n_min=5):
    tauf = [0]
    while tauf[-1]<tau_mid:
        tauf_next = max(tauf[-1]*(1+dtau), tauf[-1]+dtau)
        tauf_next = min(tauf_next, tau_mid)
        tauf.append(tauf_next)
    tauf = np.array(tauf)
    if len(tauf)<n_min:
        tauf = np.linspace(0, tau_mid, n_min)
    return tauf
def get_I_with_scattering(
    T_mid, tau_r_mid, tau_p_mid,
    cosI, lam_obs,
    get_kappa_r,
    f_kappa_obs, f_kappa_s_obs,
    dtau=0.2,
    ):
    tauf_grid = generate_tauf_grid(tau_r_mid, dtau=dtau)
    tau_grid = (tauf_grid[1:]+tauf_grid[:-1])/2
    tau_r = 2*tau_r_mid
    tau_p = 2*tau_p_mid
    T_4_over_T_mid_4 = (tauf_grid*(1-tauf_grid/tau_r) + 1/np.sqrt(3) + 1/(1.5*tau_p))/\
                       (0.25*tau_r + 1/np.sqrt(3) + 1/(1.5*tau_p))
    Tf_grid = T_4_over_T_mid_4**(1/4) * T_mid
    T_4_over_T_mid_4 = (tau_grid*(1-tau_grid/tau_r) + 1/np.sqrt(3) + 1/(1.5*tau_p))/\
                       (0.25*tau_r + 1/np.sqrt(3) + 1/(1.5*tau_p))
    T_grid = T_4_over_T_mid_4**(1/4) * T_mid
    dtau_grid = tauf_grid[1:]-tauf_grid[:-1]
    dtau_obs_grid = dtau_grid/get_kappa_r(T_grid)*(f_kappa_obs(T_grid)+f_kappa_s_obs(T_grid))
    dtau_a_obs_grid = dtau_grid/get_kappa_r(T_grid)*f_kappa_obs(T_grid)
    tauf_obs_grid = np.concatenate(([0],np.cumsum(dtau_obs_grid)))
    # print('tau_obs_mid=',tauf_obs_grid[-1])
    omegaf_grid = f_kappa_s_obs(Tf_grid)/(f_kappa_obs(Tf_grid)+f_kappa_s_obs(Tf_grid)+1e-80) # avoid zero division err
    # solve J
    nu_obs = c_light/lam_obs
    Bf_grid = B(Tf_grid,nu_obs)
    Jf_grid = solve_J(tauf_obs_grid, Bf_grid, omegaf_grid)
    Sf_grid = (1-omegaf_grid)*Bf_grid + omegaf_grid*Jf_grid
    S_grid = (Sf_grid[1:]+Sf_grid[:-1])/2
    # extend to full disk
    tauf_obs_full = np.concatenate((tauf_obs_grid, 2*tauf_obs_grid[-1]-tauf_obs_grid[-2::-1])) / cosI
    S_full = np.concatenate((S_grid, S_grid[::-1]))
    I = np.sum(S_full * np.exp(-tauf_obs_full[:-1])*(1-np.exp(-(tauf_obs_full[1:]-tauf_obs_full[:-1]))))
    # sanity check: below should give the same result as no scattering
    #tauf_a_obs_grid = np.concatenate(([0],np.cumsum(dtau_a_obs_grid)))
    #tauf_a_obs_full = np.concatenate((tauf_a_obs_grid, 2*tauf_a_obs_grid[-1]-tauf_a_obs_grid[-2::-1])) / cosI
    #B_grid = (Bf_grid[1:]+Bf_grid[:-1])/2
    #B_full = np.concatenate((B_grid, B_grid[::-1]))
    #I = np.sum(B_full * np.exp(-tauf_a_obs_full[:-1])*(1-np.exp(-(tauf_a_obs_full[1:]-tauf_a_obs_full[:-1]))))
    tau_obs = np.sum(dtau_obs_grid)*2 # this is face-on optical depth
    tau_a_obs = np.sum(dtau_a_obs_grid)*2 # this is face-on optical depth
    return I, tau_obs, tau_a_obs



"""
=======================================================================
Disk image class
=======================================================================
"""
from astropy.io import fits
from scipy import ndimage
import warnings
class DiskImage:
    """
    Stores the image of a system (at one wavelength).
    Can also be used to generate mock observation for given F(R) and
    compare mock observation with image.

    Attributes:
      img: observed image
      img_model: mock observation image
      au_per_pix: au per pixel
      (see others in __init__)
    """
    def __init__(
        self, fname, ra_deg, dec_deg, distance_pc, rms_Jy, disk_pa,
        img_size_au=400, remove_background=False,
        ):
        
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.distance_pc = distance_pc
        distance = self.distance_pc*pc
        self.rms_Jy = rms_Jy
        self.disk_pa = disk_pa
        self.img_size_au = img_size_au
        fits_data = fits.open(fname)
        hdr = fits_data[0].header
        if len(fits_data[0].shape)==4:
            img = fits_data[0].data[0,0]
        else:
            img = fits_data[0].data[0]
        if ra_deg is None:
            icx_float = img.shape[-1]/2-.5
            self.ra_deg = (icx_float+1-hdr['CRPIX1'])*hdr['CDELT1'] + hdr['CRVAL1']
        else:
            icx_float = (ra_deg-hdr['CRVAL1'])/hdr['CDELT1']+hdr['CRPIX1']-1
        if dec_deg is None:
            icy_float = img.shape[-2]/2-.5
            self.dec_deg = (icy_float+1-hdr['CRPIX2'])*hdr['CDELT2'] + hdr['CRVAL2']
        else:
            icy_float = (dec_deg-hdr['CRVAL2'])/hdr['CDELT2']+hdr['CRPIX2']-1
        icx = int(icx_float)
        icy = int(icy_float)
        signx = int(-np.sign(hdr['CDELT1'])) # x propto minus RA
        signy = int(np.sign(hdr['CDELT2']))
        self.au_per_pix = abs(hdr['CDELT1'])/180*pi*distance/au
        Npix_half = int(np.ceil(self.img_size_au/self.au_per_pix))
        if (icx-Npix_half)<0 or (icx+Npix_half)>=img.shape[-1] or (icy-Npix_half)<0 or (icy+Npix_half)>=img.shape[-2]:
            print('warning: image is too small for given img_size_au')
            # reduce Npix_half
            Npix_half = min(Npix_half, icx)
            Npix_half = min(Npix_half, img.shape[-1]-icx-1)
            Npix_half = min(Npix_half, icy)
            Npix_half = min(Npix_half, img.shape[-2]-icy-1)
            if Npix_half<20:
                raise ValueError('Cannot locate source within image!')
            print('new img size =',Npix_half*self.au_per_pix,'au')
        self.Npix_half = Npix_half
        self.img_size_au = Npix_half*self.au_per_pix
        self.img = img[icy-Npix_half*signy:icy+(Npix_half+1)*signy:signy,
                       icx-Npix_half*signx:icx+(Npix_half+1)*signx:signx]
        # put fitted disk center in center of image
        self.img = ndimage.shift(self.img,
                                 ((icy-icy_float)*signy, (icx-icx_float)*signx),
                                 order=1)
        if remove_background:
            background = np.sum(self.img *(self.img <(3*self.rms_Jy))) / np.sum(self.img <(3*self.rms_Jy))
            self.img = self.img - background
        # area of a gaussian beam: pi/(4*ln(2)) * bmaj*bmin
        # beam area in rad^2
        self.beam_area = pi/(4*np.log(2)) * (hdr['BMAJ']/180*pi)*(hdr['BMIN']/180*pi)
        # beam width in au
        self.beam_maj_au = (hdr['BMAJ']/180*pi)*distance/au
        self.beam_min_au = (hdr['BMIN']/180*pi)*distance/au
        self.beam_pa = hdr['BPA']
        fits_data.close()
        return

    def generate_mock_observation(self, R, I, cosI):
        """
        Generate mock observation for gievn I(R). (here I is the intensity)

        Args:
          R, I: 1d array, I = I(R) at R[1:]
                so len(F) = len(R)-1
          cosI: inclination
        """
        R_au = R/au
        I = np.concatenate(([I[0]], I))
        f_I = scipy.interpolate.interp1d(R_au, I, bounds_error=False, fill_value=0, kind='linear')
        N_half = self.Npix_half
        x1d = np.arange(-N_half,N_half+1)*self.au_per_pix
        y,x = np.meshgrid(x1d,x1d,indexing='ij')
        r = np.sqrt(x**2/cosI**2+y**2)
        I = f_I(r)
        # rotate to align with beam
        I = ndimage.interpolation.rotate(I, -self.disk_pa+self.beam_pa,reshape=False) # ccw rotate
        # blur: 2sqrt(2*log(2)) * sigma = FWHM
        sigmas = np.array([self.beam_maj_au, self.beam_min_au])/self.au_per_pix/(2*np.sqrt(2*np.log(2)))
        I = ndimage.gaussian_filter(I, sigma=sigmas)
        # rotate to align with image
        I = ndimage.interpolation.rotate(I, -self.beam_pa,reshape=False)
        # convert to flux density in Jy/beam
        self.img_model = I*1e23*self.beam_area
        return
    def generate_mask(self, mask_radius, cosI):
        """
        Generate a mask covering radius (in au) between mask_radius.
        
        Args:
          mask_radius: (2,) array
          cosI: disk inclination
        Returns:
          mask: 2d array of mask
        """
        N_half = self.Npix_half
        x1d = np.arange(-N_half,N_half+1)*self.au_per_pix
        y,x = np.meshgrid(x1d,x1d,indexing='ij')
        r = np.sqrt(x**2/cosI**2+y**2)
        mask = (r>=mask_radius[0])*(r<=mask_radius[1])*1.
        mask = ndimage.interpolation.rotate(mask, -self.disk_pa,reshape=False)
        # limit to (0,1)
        mask = np.maximum(mask, 0)
        mask = np.minimum(mask, 1)
        self.mask = mask
        return
    def evaluate_log_likelihood_old(self, sigma_log_model=np.log(2)/2):
        img1 = self.img
        img2 = self.img_model
        sigma = self.rms_Jy # noise
        if hasattr(self, 'sig_obs'):
            sigma = self.sig_obs
        chisq1 = (img1-img2)**2/(2*sigma**2) + np.log(sigma)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chisq2 = np.log(img1/img2)**2 / (2*sigma_log_model**2) + np.log(sigma_log_model) + np.log(img1)
        chisq2 = np.nan_to_num(chisq2, nan=1e6)
        chisq = np.minimum(chisq1, chisq2)

        dlog_likelihood =  - chisq # constant shift is unimportant here

        beam_size_au_sq = self.beam_maj_au * self.beam_min_au * pi/(4*np.log(2))
        pix_size_au_sq = self.au_per_pix**2
        beam_per_pix = pix_size_au_sq/beam_size_au_sq
        
        if hasattr(self, 'effective_beam_size_au_sq'):
            beam_per_pix = pix_size_au_sq/self.effective_beam_size_au_sq
        
        if hasattr(self, 'mask'):
            log_likelihood = np.sum(dlog_likelihood*beam_per_pix*self.mask)
        else:
            log_likelihood = np.sum(dlog_likelihood*beam_per_pix)
        
        self.log_likelihood = log_likelihood
        self.sigma_log_model = sigma_log_model
        return log_likelihood
    def evaluate_log_likelihood(self, sigma_log_model=np.log(2)/2):
        img1 = self.img
        img2 = self.img_model
        sigma = self.rms_Jy # noise
        if hasattr(self, 'sig_obs'):
            sigma = self.sig_obs
        sigma_sq = sigma**2 + (sigma_log_model*img2)**2
        chisq = (img1-img2)**2/(2*sigma_sq) + np.log(sigma_sq)/2
        #chisq1 = (img1-img2)**2/(2*sigma**2) + np.log(sigma)
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        #    chisq2 = np.log(img1/img2)**2 / (2*sigma_log_model**2) + np.log(sigma_log_model) + np.log(img1)
        #chisq2 = np.nan_to_num(chisq2, nan=1e6)
        #chisq = np.minimum(chisq1, chisq2)

        dlog_likelihood =  - chisq # constant shift is unimportant here

        beam_size_au_sq = self.beam_maj_au * self.beam_min_au * pi/(4*np.log(2))
        pix_size_au_sq = self.au_per_pix**2
        beam_per_pix = pix_size_au_sq/beam_size_au_sq
        
        if hasattr(self, 'effective_beam_size_au_sq'):
            beam_per_pix = pix_size_au_sq/self.effective_beam_size_au_sq
        
        if hasattr(self, 'mask'):
            log_likelihood = np.sum(dlog_likelihood*beam_per_pix*self.mask)
        else:
            log_likelihood = np.sum(dlog_likelihood*beam_per_pix)
        
        self.log_likelihood = log_likelihood
        self.sigma_log_model = sigma_log_model
        return log_likelihood
    def evaluate_disk_chi_sq(self, sigma_log_model=np.log(2)/2):
        img1 = self.img
        img2 = self.img_model
        sigma = self.rms_Jy # noise
        if hasattr(self, 'sig_obs'):
            sigma = self.sig_obs
        chisq1 = (img1-img2)**2/(2*sigma**2) + np.log(sigma)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chisq2 = np.log(img1/img2)**2 / (2*sigma_log_model**2) + np.log(sigma_log_model) + np.log(img1)
        chisq2 = np.nan_to_num(chisq2, nan=1e6)
        chisq = np.minimum(chisq1, chisq2) - np.log(np.maximum(img1, sigma))

        is_disk = (img2>sigma)

        beam_size_au_sq = self.beam_maj_au * self.beam_min_au * pi/(4*np.log(2))
        pix_size_au_sq = self.au_per_pix**2
        beam_per_pix = pix_size_au_sq/beam_size_au_sq
        if hasattr(self, 'effective_beam_size_au_sq'):
            beam_per_pix = pix_size_au_sq/self.effective_beam_size_au_sq
        disk_area_in_beam = np.sum(is_disk*beam_per_pix)
        #print(np.sum(1-is_disk)*beam_per_pix)

        mean_chisq = np.sum(chisq*is_disk*beam_per_pix)/np.sum(is_disk*beam_per_pix)
        #print(np.sum(chisq*(1-is_disk))/np.sum(1-is_disk))

        self.mean_chisq = mean_chisq
        self.disk_area_in_beam = disk_area_in_beam
        return mean_chisq, disk_area_in_beam









class DiskFitting:
    """
    Class for fitting multi-wavelength observations of a system.
    
    Attributes:
        source_name
        disk_model: DiskModel object
        disk_image_list: list of DiskImage objects
    """
    def __init__(self, source_name, opacity_table, opacity_table_lam, disk_property_table):
        self.source_name = source_name
        self.disk_model = DiskModel(opacity_table, opacity_table_lam, disk_property_table)
        self.disk_image_list = []
        self.lam_obs_list = []
        return
    def add_observation(self, disk_image, lam_obs):
        self.disk_image_list.append(disk_image)
        self.lam_obs_list.append(lam_obs)
        self.disk_model.set_lam_obs_list(self.lam_obs_list)
    def set_cosI(self, cosI):
        self.cosI = cosI
    def evaluate_log_likelihood(self, weights=None, sigma_log_model=np.log(2)/2):
        self.disk_model.generate_disk_profile()
        self.disk_model.generate_observed_flux(cosI=self.cosI)
        N_obs = len(self.lam_obs_list)
        ll = np.zeros(N_obs)
        for i in range(N_obs):
            self.disk_image_list[i].generate_mock_observation(
                R=self.disk_model.R, I=self.disk_model.I_obs[i], cosI=self.cosI)
            ll[i] = self.disk_image_list[i].evaluate_log_likelihood(sigma_log_model=sigma_log_model)
        if weights is None:
            ll = np.sum(ll)
        else:
            ll = np.sum(ll*weights)
        return ll