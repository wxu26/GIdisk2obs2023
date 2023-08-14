#from TMC1A_fitting import *
from disk_model import *
from TMC1A_fitting import *
import dill as pickle
import corner
import matplotlib.pyplot as plt

"""
show fit results: print a table, show posterior plot, and return truths and flat samples
"""

def get_range(data, fraction):
    # given data, return the minimum interval that contains given fraction of the data
    data = np.sort(data)
    n = len(data)
    n_range = int(n*fraction)
    n_left = n-n_range
    L = data[:n_left]
    R = data[n_range:]
    i = np.argmin(R-L)
    l = L[i]
    r = R[i]
    return [l, r]

def print_confidence_interval(y,log10=True,label='variable',header=False):
    if header:
        print("\t\tpeak\t\t1sig\t\t1sig\t\t2sig\t\t2sig\t\t1sigwidth")
    p = np.zeros(6)
    p[0] = np.sum(get_range(y,.1))/2 # peak: using central 10%
    p[1], p[2] = get_range(y,.68) # 1 sigma
    p[3], p[4] = get_range(y,.95) # 2 sigma
    if log10:
        p = 10**p
    p[5] = p[2]/p[1] # 1 sigma width
    s = label+"\t{:6.3f}\t\t{:6.3f}\t\t{:6.3f}\t\t{:6.3f}\t\t{:6.3f}\t\t{:.3f}".format(*p)
    print(s)

def print_percentiles(y,log10=True,label='variable',header=False):
    # obsolete
    if log10:
        y = 10**y
    if header:
        print("\t\t5%\t\t16%\t\t50%\t\t84%\t\t95%\t\t84/16")
    percentiles = []
    for p in [5,16,50,84,95]:
        percentiles += [np.percentile(y,p)]
    s = label+"\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}\t\t{:.2f}".format(*percentiles,percentiles[3]/percentiles[1])
    #s = label+"\t{:6.2f}".format(*percentiles[:1])
    print(s)

def show_fit_results(fname, plot_chain=True, plot_corner=True, print_result=True, return_normalized_sample=False):
    sampler = pickle.load(open(fname,'rb'))
    samples = sampler.get_chain()
    ndim = samples.shape[-1]
    sigma_log_model = ndim>2+len(var_names)
    new_a_bound = True
    ia10 = 2+var_names.index('amax10')
    ia100 = 2+var_names.index('amax100')

    discard = samples.shape[0]//2
    n_thin = 2
    flat_samples = sampler.get_chain(discard=discard, thin=n_thin, flat=True)
    #I = (flat_samples[:,ia10]>=np.log(1e-4)) * (flat_samples[:,ia100]>=np.log(1e-4))
    I = (flat_samples[:,ia10]>=np.log(1e-5))
    if sigma_log_model:
        I = I*(flat_samples[:,-1]>=0)*(flat_samples[:,-1]<=1)
    flat_samples = flat_samples[I]
    truths = []
    for i in range(ndim):
        #truths.append(np.median(flat_samples[:,i]))
        truths.append(np.sum(get_range(flat_samples[:,i],.1))/2)

    samples[:,:,2] = samples[:,:,2]*np.log10(np.e) - np.log10(Msun)
    samples[:,:,3] = samples[:,:,3]*np.log10(np.e) - np.log10(au)
    samples[:,:,4] = samples[:,:,4]*np.log10(np.e) - np.log10(Msun/yr)
    samples[:,:,5] = samples[:,:,5]*np.log10(np.e)
    samples[:,:,6] = samples[:,:,6]*np.log10(np.e)
    samples[:,:,7] = samples[:,:,7]*np.log10(np.e)
    samples[:,:,8] = samples[:,:,8]*np.log10(np.e)

    labels = ["cosI","pa","log10 M/Msun","log10 Rd/au","log10 Mdot/(Msun/yr)","log10 a_max_10/cm","log10 a_max_100/cm","log10 Q_10","log10 Q_100","q"]
    labels_tex = [r"cos$I$","pa",r"log$_{10}$ $M/M_\odot$",r"log$_{10}$ $R_{\rm d}/$au",r"log$_{10}$ $\dot M/(M_\odot/{\rm yr})$",r"log$_{10}$ $a_{\rm max}^{10{\rm au}}/$cm",r"log$_{10}$ $a_{\rm max}^{100{\rm au}}/$cm",r"log$_{10}$ $Q^{10{\rm au}}$",r"log$_{10}$ $Q^{100{\rm au}}$",r"$q$"]
    if sigma_log_model:
        labels += ["sigma_rel_model"]
        labels_tex += [r"$\sigma_{\rm rel,model}$"]

    flat_samples_scaled = sampler.get_chain(discard=discard, thin=n_thin, flat=True)
    flat_samples_scaled = flat_samples_scaled[I]
    truths_scaled = []
    for i in range(ndim):
        #truths_scaled.append(np.median(flat_samples_scaled[:,i]))
        truths_scaled.append(np.sum(get_range(flat_samples_scaled[:,i],.1))/2)

    if plot_chain:
        fig, axes = plt.subplots(ndim,figsize=(10, 2*ndim), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            #ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");

    if plot_corner:
        fig = corner.corner(
            flat_samples_scaled, labels=labels_tex, truths=truths_scaled
        )

    if print_result:
        print_confidence_interval(flat_samples_scaled[:,0],log10=False,header=True,label='cosI\t')
        print_confidence_interval(flat_samples_scaled[:,1],log10=False,label='pa\t')
        print_confidence_interval(flat_samples_scaled[:,2],label='M[Msun]\t')
        print_confidence_interval(flat_samples_scaled[:,3],label='Rd[au]\t')
        print_confidence_interval(flat_samples_scaled[:,4]+6,label='Mdot[1e-6Ms/yr]')
        print_confidence_interval(flat_samples_scaled[:,5]+1,label='a_max_10[mm]')
        print_confidence_interval(flat_samples_scaled[:,6]+1,label='a_max_100[mm]')
        print_confidence_interval(flat_samples_scaled[:,7],label='Q_10\t')
        print_confidence_interval(flat_samples_scaled[:,8],label='Q_100\t')
        print_confidence_interval(flat_samples_scaled[:,9],log10=False,label='q\t')
        if sigma_log_model:
            print_confidence_interval(flat_samples_scaled[:,10],log10=False,label='sigma_log_model')
    if return_normalized_sample:
        return truths, flat_samples_scaled
    else:
        return truths, flat_samples

"""
plot images and residuals
"""
# plot the following:
# 1. image (linear scale, T_b)
# 2. residual (linear scale, same colorbar as image)
# 3. residual (absolute, in sigma_obs)
# 4. residual (relative)
def plot_one_panel(I, v, xmax, label='', plot_beam=True, beam_color='w', plot_scale=False, colorbar=True, **kwargs):
    plt.imshow(v, origin='lower',
               extent=(-I.img_size_au,I.img_size_au,-I.img_size_au,I.img_size_au),
               **kwargs)
    plt.xlim(-xmax,xmax)
    plt.ylim(-xmax,xmax)
    plt.gca().set_aspect('equal','box')
    if colorbar:
        plt.colorbar()
    plt.title(label)

    #plt.gca().axis('off')
    plt.xticks([])
    plt.yticks([])
    
    # scale bar
    if plot_scale:
        x0,y0 = -xmax*0.5, -xmax*0.7
        plt.plot([x0-50,x0+50],[y0,y0],beam_color,lw=2)
        plt.text(x0, y0-xmax*0.08, '100 au', color=beam_color, fontsize=10, va='top', ha='center')
    
    # plot beam
    if plot_beam:
        th = np.linspace(0,2*np.pi,100)
        x0 = np.cos(th)*I.beam_min_au/2
        y0 = np.sin(th)*I.beam_maj_au/2
        dth = I.beam_pa/180*np.pi
        x = x0*np.cos(dth) - y0*np.sin(dth)
        y = x0*np.sin(dth) + y0*np.cos(dth)
        xc = xmax*0.7
        yc = -xmax*0.7
        plt.fill(x+xc,y+yc,beam_color)

def take_cut_and_plot(I,img,f,cut_deg,err_mode='obs',**kwargs):
    img = ndimage.interpolation.rotate(img,-cut_deg,reshape=False) # ccw rotate by cut_deg
    i0 = img.shape[1]//2
    x = (np.arange(2*i0+1)-i0) * I.au_per_pix
    y = img[:,i0]
    #if err_mode == 'obs': x = np.abs(x)
    p,=plt.plot(x,y*f,**kwargs)
    if err_mode == 'obs':
        yl = y-I.rms_Jy
        yh = y+I.rms_Jy
        alpha = 0.15
    else:
        yl = y/np.exp(I.sigma_log_model)
        yh = y*np.exp(I.sigma_log_model)
        alpha = 0.3
    plt.fill_between(x, yl*f, yh*f, color=p.get_color(), lw=0, label='_nolegend_', alpha=alpha)
    return x,y

def get_chi(D, i, sigma_log_model=np.log(2)/2):
    img1 = D.disk_image_list[i].img
    img2 = D.disk_image_list[i].img_model
    sigma = D.disk_image_list[i].rms_Jy # noise
    chi1 = (img1-img2)/np.sqrt(2*sigma**2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chi2 = np.log(img1/img2) / np.sqrt(2*sigma_log_model**2)
    chi2 = np.nan_to_num(chi2, nan=1e6)
    chi = chi1
    chi[np.abs(chi2)<np.abs(chi1)] = chi2[np.abs(chi2)<np.abs(chi1)]
    return chi

def compare_img(D,i_img,fig_size_au,color):
    lam = D.disk_model.lam_obs_list[i_img]
    I = D.disk_image_list[i_img]
    Jy_per_beam_to_T_b = 1e-23 * lam**2 / (2*kB*I.beam_area)
    x,y = take_cut_and_plot(I, I.img, Jy_per_beam_to_T_b, -I.disk_pa, err_mode='obs', color='k')
    x2,y2 = take_cut_and_plot(I, I.img_model, Jy_per_beam_to_T_b, -I.disk_pa, err_mode='model', color=color, lw=2, alpha=0.8)
    ymax = max(np.amax(y*(np.abs(x)<fig_size_au)), np.amax(y2*(np.abs(x2)<fig_size_au)))
    #plt.plot([-fig_size_au,fig_size_au], [I.rms_Jy, I.rms_Jy], 'k:', lw=.5)
    plt.legend(['observation','model',r'$\sigma_{\rm obs}$'], loc=2, frameon=False, handlelength=1)
    plt.xlim(-fig_size_au,fig_size_au)
    #plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    #plt.xlabel('Offset along major axis [au]')
    #plt.yscale('log')
    #plt.ylim(I.rms_Jy/2,ymax*3)
    #plt.ylabel(ylabel_text)
    #set_ticks()
    
from matplotlib.colors import LogNorm
#from turbo_cmap import *

def get_T_b(I_Jy_beam, lam, I):
    Jy_per_beam_to_T_b = 1e-23 * lam**2 / (2*kB*I.beam_area)
    T_b = I_Jy_beam*Jy_per_beam_to_T_b
    return T_b

def plot_image_and_residual(D, i_img, ax):
    # convert image to T_b
    # T_b = I * lam^2 / (2k_B*Omega_beam)
    lam = D.disk_model.lam_obs_list[i_img]
    I = D.disk_image_list[i_img]
    #Jy_per_beam_to_T_b = 1e-23 * lam**2 / (2*kB*I.beam_area)
    sigma_obs = I.rms_Jy
    # plot image
    #T_b = I.img * Jy_per_beam_to_T_b
    #T_b_model = I.img_model * Jy_per_beam_to_T_b
    T_b = get_T_b(I.img, lam, I)
    T_b_model = get_T_b(I.img_model, lam, I)
    xmax = 250
    plt.sca(ax[0])
    T_b_max = np.amax(T_b)
    T_b_min = get_T_b(I.rms_Jy, lam, I)*3
    T_b_max = 50
    plot_one_panel(I, np.maximum(T_b,T_b_min), xmax, vmin=T_b_min, vmax=T_b_max, cmap='turbo',
                   norm=LogNorm(vmin=T_b_min, vmax=T_b_max), label='T_b: observation',
                   plot_scale=True)
    # residual T_b
    plt.sca(ax[1])
    plot_one_panel(I, np.maximum(T_b_model,T_b_min), xmax, vmin=T_b_min, vmax=T_b_max, cmap='turbo',
                   norm=LogNorm(vmin=T_b_min, vmax=T_b_max), label='T_b: model')
    
    plt.sca(ax[2])
    #plot_one_panel(I, np.maximum(T_b-T_b_model,T_b_min), xmax, vmin=T_b_min, vmax=T_b_max, cmap='turbo',
    #               norm=LogNorm(vmin=T_b_min, vmax=T_b_max), label='absolute difference')
    plot_one_panel(I, T_b-T_b_model, xmax, vmin=T_b_min, vmax=T_b_max, cmap='Blues',
               norm=LogNorm(vmin=T_b_min, vmax=T_b_max), label='absolute difference', colorbar=False, beam_color='k')
    cb = plt.colorbar(pad=-0.075)
    plot_one_panel(I, T_b_model-T_b, xmax, vmin=T_b_min, vmax=T_b_max, cmap='Reds',
               norm=LogNorm(vmin=T_b_min, vmax=T_b_max), label='absolute difference', colorbar=False, beam_color='k')
    cb = plt.colorbar(pad=0.01,ticks=[])
    cb.ax.tick_params(which='both',length=0)
    
    # residual T_b in sigma_obs
    #plt.sca(ax[2])
    #plot_one_panel(I, (I.img-I.img_model)/sigma_obs, xmax, vmin=-5, vmax=5, cmap='PiYG', label='residual T_b/sigma_obs')
    
    # residual T_b in I_model
    plt.sca(ax[3])
    err_relative = (I.img-I.img_model)/I.img_model
    err_relative[I.img_model<sigma_obs] = np.nan
    plot_one_panel(I, err_relative, xmax, vmin=-2, vmax=2, cmap='RdBu', label='relative difference',
                   beam_color='k')
    
    #plot_one_panel(I, np.abs(T_b_model-T_b), xmax, vmin=T_b_min, vmax=T_b_max, cmap='turbo',
    #               norm=LogNorm(vmin=T_b_min, vmax=T_b_max), label='abs T_b err')
    
    plt.sca(ax[4])
    chi = get_chi(D,i_img)
    plot_one_panel(I, chi, xmax, vmin=-3, vmax=3, cmap='PiYG', label=r'$\chi$', beam_color='k')
    
    
    plt.sca(ax[5])
    compare_img(D,i_img,xmax,"tab:blue")
    plt.yscale('log')
    T_b_min = get_T_b(I.rms_Jy, lam, I)/2
    T_b_max = 100
    plt.ylim(T_b_min, T_b_max)
    plt.ylabel('T_b')
    plt.xlabel('offset along major axis [au]')


def plot_image_and_residual_all_wavelengths(D, theta_scaled, print_parameters=False):
    
    if len(theta_scaled)>6:
        cosI, pa, log10_Mstar, log10_Rd, log10_Mdot, log10_a_max, log10_Q = theta_scaled
    else:
        log10_Q = np.log10(Q0)
        cosI, pa, log10_Mstar, log10_Rd, log10_Mdot, log10_a_max = theta_scaled
    Mstar = 10**(log10_Mstar)*Msun
    Rd = 10**(log10_Rd)*au
    Mdot = 10**(log10_Mdot)*Msun/yr
    a_max = 10**log10_a_max
    Q = 10**log10_Q
    
    ll = get_log_probability_before_prior(cosI, pa, Mstar, Rd, Mdot, a_max, Q, D)
    
    if print_parameters:
        print('Mstar =',D.disk_model.Mstar/Msun)
        print('Mtot =',D.disk_model.M/Msun)
        
    fig, ax = plt.subplots(4,6,figsize=(20,12))
    labels = ['band K','band Q','band 6','band 7']
    for i in range(4):
        plot_image_and_residual(D, i, ax[i])
        plt.sca(ax[i,0])
        plt.text(-200,200,labels[i], color='w', va='top', fontsize=14)
    plt.tight_layout()
    
    return D

"""
plot SED
"""
def get_integrated_flux(I,model=True,only_disk=True):
    if model:
        img = I.img_model
    else:
        img = I.img
    img_above_threshold = I.img_model>I.rms_Jy
    beam_size_au_sq = I.beam_maj_au * I.beam_min_au * pi/(4*np.log(2))
    pix_size_au_sq = I.au_per_pix**2
    beam_per_pix = pix_size_au_sq/beam_size_au_sq
    if only_disk:
        img = img*img_above_threshold
    F = np.sum(img*beam_per_pix)
    return F
def get_SED_and_plot(D,model=True,only_disk=True,**kwargs):
    lam = D.lam_obs_list
    n_obs = len(lam)
    F = np.zeros(n_obs)
    for i in range(n_obs):
        F[i] = get_integrated_flux(D.disk_image_list[i],model=model,only_disk=only_disk)
    if 'marker' not in kwargs:
        kwargs['marker'] = 'd'
    plt.plot(lam, F, **kwargs)
def plot_SED_at_theta(D, theta_scaled, **kwargs):
    if len(theta_scaled)>6:
        cosI, pa, log10_Mstar, log10_Rd, log10_Mdot, log10_a_max, log10_Q = theta_scaled
    else:
        log10_Q = np.log10(Q0)
        cosI, pa, log10_Mstar, log10_Rd, log10_Mdot, log10_a_max = theta_scaled
    Mstar = 10**(log10_Mstar)*Msun
    Rd = 10**(log10_Rd)*au
    Mdot = 10**(log10_Mdot)*Msun/yr
    a_max = 10**log10_a_max
    Q = 10**log10_Q
    
    ll = get_log_probability_before_prior(cosI, pa, Mstar, Rd, Mdot, a_max, Q, D)
    # plot
    get_SED_and_plot(D, **kwargs)

"""
rotate & stretch into face-on and make beam circular
"""
s_to_w = 2*np.sqrt(2*np.log(2)) # sigma to FWHM
w_to_s = 1/s_to_w
def beam(beam_maj, beam_min, beam_pa, plot=False, xc=0, yc=0):
    s_maj, s_min = beam_maj*w_to_s, beam_min*w_to_s
    thpa = beam_pa/180*np.pi # pa=0 -> maj in y
    sxx = s_maj**2*np.sin(thpa)**2 + s_min**2*np.cos(thpa)**2
    syy = s_maj**2*np.cos(thpa)**2 + s_min**2*np.sin(thpa)**2
    sxy = (-s_maj**2+s_min**2)*np.sin(thpa)*np.cos(thpa)
    
    # plot beam
    if plot:
        th = np.linspace(0,2*np.pi,100)
        x0 = np.cos(th)*beam_min/2
        y0 = np.sin(th)*beam_maj/2
        x = x0*np.cos(thpa) - y0*np.sin(thpa)
        y = x0*np.sin(thpa) + y0*np.cos(thpa)
        plt.fill(x+xc,y+yc,'w')
        
    return sxx, syy, sxy

def f(x,p):
    s_maj, s_min, thpa = x
    sxx0, syy0, sxy0 = p
    sxx = s_maj**2*np.sin(thpa)**2 + s_min**2*np.cos(thpa)**2
    syy = s_maj**2*np.cos(thpa)**2 + s_min**2*np.sin(thpa)**2
    sxy = (-s_maj**2+s_min**2)*np.sin(thpa)*np.cos(thpa)
    return sxx-sxx0, syy-syy0, sxy-sxy0

def solve_beam(sxx, syy, sxy):
    res = scipy.optimize.fsolve(f,
                                x0=[(sxx)**(1/2),(syy)**(1/2),np.sign(sxy)],
                                args=([sxx, syy, sxy]),
                                full_output=True)
    s_maj, s_min, thpa = res[0]
    print(res[-1],res[1]['fvec'])
    if s_maj<0 and s_min<0:
        s_maj=-s_maj
        s_min=-s_min
    elif s_maj<0:
        s_maj=-s_maj
        #thpa=-thpa
    elif s_min<0:
        s_min=-s_min
        #thpa=-thpa
    beam_maj = s_to_w*s_maj
    beam_min = s_to_w*s_min
    beam_pa = thpa/np.pi*180
    return s_maj, s_min, beam_maj, beam_min, beam_pa

def solve_1d_beam(sxx,syy,sxy):
    s_maj = np.sqrt(sxx+syy)
    s_min = 0
    thpa = np.arctan(-np.sign(sxy)*np.sqrt(sxx)/np.sqrt(syy))
    beam_maj = s_to_w*s_maj
    beam_min = s_to_w*s_min
    beam_pa = thpa/np.pi*180
    return s_maj, s_min, beam_maj, beam_min, beam_pa

# sxy*sxy = sxx*syy = sxy0*sxy0
# sxx0+sxx = syy0+syy
# x = sxx
# (sxx0-syy0+x)*x = sxy0*sxy0

def add_beam_to_circularize(sxx0, syy0, sxy0):
    a = 1
    b = sxx0-syy0
    c = -sxy0**2
    sxx = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    syy = sxy0**2/sxx
    sxy = -sxy0
    s = np.sqrt(sxx+sxx0)
    #print(sxx+sxx0, syy+syy0)
    return sxx, syy, sxy, s

def get_img(I, img, pa, cosI, plot=False, vmin=None, vmax=None, extent=None, circularize_beam=True, beam_au=None):
    
    beam_maj = I.beam_maj_au*1
    beam_min = I.beam_min_au*1
    beam_pa = I.beam_pa*1
    eps = 1.e-40
    
    if plot:
        plt.figure()
        plt.imshow(np.maximum(eps,img),origin='lower',cmap='turbo',norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
        beam(beam_maj, beam_min, beam_pa, plot=True, xc=150, yc=-150)
        plt.title('original')
        plt.colorbar()

    # rotate
    img = scipy.ndimage.rotate(img, pa, reshape=False)
    if plot:
        plt.figure()
        plt.imshow(np.maximum(eps,img),origin='lower',cmap='turbo',norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
        plt.title('rotate')
        plt.colorbar()
    beam_pa = beam_pa - pa
    sxx, syy, sxy = beam(beam_maj, beam_min, beam_pa, plot=plot, xc=150, yc=-150)
    #print(sxx,syy,sxy,beam_pa)
    
    # stretch
    img = scipy.ndimage.affine_transform(img, [1,cosI], offset=[0,len(img)/2*(1-cosI)])
    if plot:
        plt.figure()
        plt.imshow(np.maximum(eps,img),origin='lower',cmap='turbo',norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
        plt.title('stretch')
        plt.colorbar()
    sxx = sxx/cosI**2
    sxy = sxy/cosI
    if circularize_beam:
        s_maj, s_min, beam_maj, beam_min, beam_pa = solve_beam(sxx, syy, sxy)
        sxx, syy, sxy = beam(beam_maj, beam_min, beam_pa, plot=plot, xc=150, yc=-150)

    # circularize beam
    if circularize_beam:
        sxx, syy, sxy, s = add_beam_to_circularize(sxx, syy, sxy)
        s_maj, s_min, beam_maj, beam_min, beam_pa = solve_1d_beam(sxx, syy, sxy)
        # convolve with added beam
        img = scipy.ndimage.rotate(img, beam_pa, reshape=False)
        sigmas = np.array([s_maj, s_min])/I.au_per_pix
        img = scipy.ndimage.gaussian_filter(img, sigma=sigmas)
        img = scipy.ndimage.rotate(img, -beam_pa, reshape=False)
        if plot:
            plt.figure()
            plt.imshow(np.maximum(eps,img),origin='lower',cmap='turbo',norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
            plt.title('increase beam size')
            plt.colorbar()
        w = s*s_to_w
        _ = beam(w, w, 0, plot=plot, xc=150, yc=-150)
        print('beam size [au] =',w)
        #print(s_maj, s_min)
    
    if (beam_au is not None) and beam_au>w:
        dw = np.sqrt(beam_au**2-w**2)
        ds = dw*w_to_s
        img = scipy.ndimage.gaussian_filter(img, sigma=ds//I.au_per_pix)
        print('increase to beam size [au] =',beam_au)
    return img

def map_to_polar(I, img,
                 rmax = 250, Nr = 100, Nphi = 64):
    x = np.arange(img.shape[0])
    x = (x-(len(x)-1)/2)*I.au_per_pix
    img = np.nan_to_num(img)
    fc = np.vectorize(scipy.interpolate.interp2d(x,x,img))
    #def remap_to_polar():
    rf_grid = np.linspace(0,rmax,Nr)
    rc_grid = (rf_grid[1:]+rf_grid[:-1])/2
    phif_grid = np.linspace(0,2*pi,Nphi)
    phic_grid = (phif_grid[1:] + phif_grid[:-1])/2
    r, phi = np.meshgrid(rc_grid, phic_grid)
    x, y = r*np.cos(phi), r*np.sin(phi)
    c = fc(x,y)
    return c, rf_grid, rc_grid, phif_grid, phic_grid


from scipy.special import ellipe
import scipy.interpolate

def generate_uncertainty_and_effective_beam_size(I, cosI, update_beam_size=True, plot=False):
    pa = I.disk_pa
    xmax = I.img_size_au
    img_face_on = get_img(I, I.img, pa, cosI, circularize_beam=True, plot=plot, vmin=1e-5, extent=(-xmax,xmax,-xmax,xmax))
    img_polar, rf, rc, phif, phic = map_to_polar(I,img_face_on,Nphi=128,rmax=I.img_size_au)
    img_phi_std = np.std(img_polar, axis=0)
    N_half = I.Npix_half
    x1d = np.arange(-N_half,N_half+1)*I.au_per_pix
    y,x = np.meshgrid(x1d,x1d,indexing='ij')
    r = np.sqrt(x**2/cosI**2+y**2)
    rc2 = rc.copy()
    rc2[0] = 0
    f_sig = scipy.interpolate.interp1d(rc2, img_phi_std, bounds_error=False, fill_value=I.rms_Jy, kind='linear')
    sig = f_sig(r)
    # rotate to align with beam
    sig = ndimage.interpolation.rotate(sig, -pa,reshape=False) # ccw rotate
    sig = np.maximum(sig, I.rms_Jy)
    beam_size_au_sq = I.beam_maj_au * I.beam_min_au * np.pi/(4*np.log(2))
    e_sq = 1-cosI**2
    r_rot = ndimage.interpolation.rotate(r, -pa,reshape=False,cval=np.inf) # ccw rotate
    # ellipse circumference * sqrt beam size
    beam_size_eff = 4*r_rot*ellipe(e_sq) * np.sqrt(beam_size_au_sq)
    beam_size_eff = np.maximum(beam_size_au_sq,beam_size_eff)
    
    I.sig_obs = sig
    if update_beam_size: I.effective_beam_size_au_sq = beam_size_eff
    return

def generate_uncertainty_and_effective_beam_size_v2(I, cosI):
    pa = I.disk_pa
    xmax = I.img_size_au
    img_face_on = get_img(I, I.img, pa, cosI, circularize_beam=False, plot=False)
    img_polar, rf, rc, phif, phic = map_to_polar(I,img_face_on,Nphi=128,rmax=I.img_size_au)
    img_polar_m2 = np.real(np.mean(img_polar*np.exp(1j*phic*2)[:,None],axis=0) * np.exp(-1j*phic*2)[:,None]) * 2
    img_phi_std = np.std(img_polar-img_polar_m2, axis=0)
    N_half = I.Npix_half
    x1d = np.arange(-N_half,N_half+1)*I.au_per_pix
    y,x = np.meshgrid(x1d,x1d,indexing='ij')
    r = np.sqrt(x**2/cosI**2+y**2)
    rc2 = rc.copy()
    rc2[0] = 0
    f_sig = scipy.interpolate.interp1d(rc2, img_phi_std, bounds_error=False, fill_value=I.rms_Jy, kind='linear')
    sig = f_sig(r)
    # rotate to align with beam
    sig = ndimage.interpolation.rotate(sig, -pa,reshape=False) # ccw rotate
    sig = np.maximum(sig, I.rms_Jy)
    
    I.sig_obs = sig
    return

"""
assign parameters
"""
def assign_parameters(D, theta, **kwargs):
    ll = get_log_probability_before_prior(theta, D=D, **kwargs)
    return D