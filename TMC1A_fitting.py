from disk_model import *
import dill as pickle

disk_opacity = pickle.load(open('./data/opacity_tables/disk_opacity.pkl','rb'))
disk_opacity_lam = pickle.load(open('./data/opacity_tables/disk_opacity_lam.pkl','rb'))
disk_property = pickle.load(open('./data/opacity_tables/disk_property.pkl','rb'))

d_pc = 140

f6 = './data/tmc1a_obs_fulluv/TMC1A.Band6.robust0.5.fulluv.image.pbcor.imsub.fits'
rms6 = 17e-6
f7 = './data/tmc1a_obs_fulluv/TMC1A.Band7.robust0.5.fulluv.image.pbcor.imsub.fits'
rms7 = 143e-6

I6 = DiskImage(
        fname = f6,
        ra_deg = None,
        dec_deg = None,
        distance_pc = d_pc,
        rms_Jy = rms6, # convert to Jy/beam
        disk_pa = 0,
        img_size_au = 300,
    )
I7 = DiskImage(
        fname = f7,
        ra_deg = None,
        dec_deg = None,
        distance_pc = d_pc,
        rms_Jy = rms7, # convert to Jy/beam
        disk_pa = 0,
        img_size_au = 300,
    )

fK = './data/tmc1a_obs_fulluv/TMC1A.Kband.robust0.5.fulluv.image.pbcor.imsub.fits'
rmsK = 10e-6
fQ = './data/tmc1a_obs_fulluv/TMC1A.Qband.robust0.5.fulluv.image.pbcor.imsub.fits'
rmsQ = 13e-6

IK = DiskImage(
        fname = fK,
        ra_deg = None,
        dec_deg = None,
        distance_pc = d_pc,
        rms_Jy = rmsK, # convert to Jy/beam
        disk_pa = 0,
        img_size_au = 300,
    )
IQ = DiskImage(
        fname = fQ,
        ra_deg = None,
        dec_deg = None,
        distance_pc = d_pc,
        rms_Jy = rmsQ, # convert to Jy/beam
        disk_pa = 0,
        img_size_au = 300,
    )

D = DiskFitting('TMC1A' ,disk_opacity, disk_opacity_lam, disk_property)
D.add_observation(IK, 0.87)
D.add_observation(IQ, 0.68)
D.add_observation(I6, 0.13)
D.add_observation(I7, 0.09)

# define bounds for the log-uniform prior
bounds = {}
bounds['M'] = [.1*Msun, 10*Msun]
bounds['Rd'] = [150*au, 300*au]
bounds['Mdot'] = [1e-7*Msun/yr, 1e-4*Msun/yr]
bounds['amax10'] = [1e-5, 1]
bounds['amax100'] = [1e-5,1]
bounds['Q10'] = [.1,1e3]
bounds['Q100'] = [.1,1e3]
bounds['q'] = [2.5,3.5]

var_names = [n for n in bounds]

def get_MCMC_ic(nwalkers=128, sigma_log_model=False):
    # initial distribution: draw from random uniform distribution
    ndim = len(var_names) + 2 + 1*sigma_log_model
    ic_rel = np.random.rand(nwalkers, ndim)
    ic = np.zeros(ic_rel.shape)
    ic[:,0] = ic_rel[:,0] # cosI
    ic[:,1] = ic_rel[:,1]*180 # pa
    i = 1
    for n in var_names:
        i+=1
        b = bounds[n]
        if n=='q':
            ic[:,i] = ic_rel[:,i] * b[0] + (1-ic_rel[:,i])*b[1]
        else:
            ic[:,i] = ic_rel[:,i] * np.log(b[0]) + (1-ic_rel[:,i])*np.log(b[1])
    if sigma_log_model:
        ic[:,-1] = ic_rel[:,-1]*np.log(10)
    return ic

def get_log_probability_before_prior(theta, D=D, weights=None, sigma_log_model=np.log(2)/2):
    # get sigma_log_model
    if len(theta)>2+len(var_names):
        sigma_log_model = theta[-1]
    # store cosI and pa
    cosI, pa = theta[0], theta[1]
    n_img = len(D.disk_image_list)
    for i in range(n_img):
        D.disk_image_list[i].disk_pa = pa
    D.cosI = cosI
    # update disk parameters
    i = 1
    for n in var_names:
        i+=1
        v = theta[i]
        if n=='q':
            setattr(D.disk_model, n, v)
        else:
            setattr(D.disk_model, n, np.exp(v))
    D.disk_model.amaxs = np.log(D.disk_model.amax100/D.disk_model.amax10)/np.log(10)
    D.disk_model.Qs = np.log(D.disk_model.Q100/D.disk_model.Q10)/np.log(10)
    # compute ll for all images
    ll = D.evaluate_log_likelihood(weights=weights,sigma_log_model=sigma_log_model)
    if np.isnan(ll):
        ll = -np.inf
    # edge penalty
    i0 = np.argmin(D.lam_obs_list) # minimum wavelength, which generrally sees the largest apparant disk size
    normalized_edge_F = D.disk_model.I_obs[i0][-1]*D.disk_image_list[i0].beam_area*1e23/D.disk_image_list[i0].rms_Jy
    ll_edge_penalty = 0
    #if normalized_edge_F<1:
    #    ll_edge_penalty = -np.inf
    return ll+ll_edge_penalty

def get_log_probability(theta, weights=None, Mp_prior=True, sigma_log_model=np.log(2)/2):
    ll = get_log_probability_before_prior(theta, D=D, weights=weights, sigma_log_model=sigma_log_model)
    
    # prior bounds
    cosI, pa = theta[0], theta[1]
    if (cosI<0) or (cosI>1) or (pa<0) or (pa>180):
        return -np.inf
    i = 1
    for n in var_names:
        i+=1
        b = bounds[n]
        if n=='q':
            if theta[i]<b[0] or theta[i]>b[1]:
                return -np.inf
        else:
            if theta[i]<np.log(b[0]) or theta[i]>np.log(b[1]):
                return -np.inf

    # Mp prior
    if Mp_prior:
        M = np.exp(theta[2+var_names.index('M')])/Msun
        cosI = theta[0]
        sini_sq = 1-cosI**2
        Mp = M * sini_sq
        Mp0 = 0.56
        sigmaMp = 0.05
        ll_Mp_prior = -(Mp-Mp0)**2/(2*sigmaMp**2)
        ll += ll_Mp_prior

    return ll