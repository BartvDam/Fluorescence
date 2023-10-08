#%% libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import factorial


# %%

def convoluted_exponential_decay(decay_parameters: list,
                                 t: np.ndarray,
                                 irf: np.ndarray,
                                 t_shift: int):

    amplitude = np.abs(decay_parameters[0])
    decay_rate = np.abs(decay_parameters[1])
    # t_shift = int(decay_parameters[2])
    t_shift = t_shift
    bgr = np.abs(decay_parameters[2])

    size = len(t)

    exp_decay = amplitude * np.exp(-decay_rate * (t - np.min(t)))
    conv_exp_decay = np.convolve(exp_decay,irf)

    if t_shift >= 0:
        conv_exp_decay_shifted = conv_exp_decay[t_shift : size + t_shift]
    else:
        conv_exp_decay_shifted = np.hstack((conv_exp_decay[t_shift:],conv_exp_decay[0 : size + t_shift]))

    conv_exp_decay_shifted = conv_exp_decay_shifted + bgr

    return conv_exp_decay_shifted

def merit_function(t: np.ndarray,
                    decay: np.ndarray,
                    fit: np.ndarray,
                    method: str):
        
    if method == 'MLE':
        M = np.sum(-1 * decay * np.log(fit) + fit) # + np.log(np.fact)
    elif method == 'LS':
        M = np.sum((decay - fit)**2)

    return M

def fit_exponential_decay(t: np.ndarray,
                            decay: np.ndarray,
                            irf: np.ndarray,
                            initial_guess: tuple,
                            t_shift_range: tuple,
                            fitting_method: str = 'MLE'):
    
    for i,t_shift in enumerate(np.arange(t_shift_range[0],t_shift_range[1]+1)): 

        merit_func = lambda decay_parameters : merit_function(
                    t,
                    decay,
                    convoluted_exponential_decay(decay_parameters,t,irf,t_shift),fitting_method)
        
        _fit = optimize.minimize(merit_func,
                                 initial_guess,
                                 options={'gtol': 1e-9}
                                 #bounds = ((0, None),(0,None),(0,None)),
                                 #method = 'L-BFGS-B'
                                 )
                                #  method = 'trust-ncg')
                                #  method = 'BFGS')
        
        if (i == 0):
            fit = _fit
            best_fit_timeshift = t_shift
        elif (_fit.fun < fit.fun):
            fit = _fit
            best_fit_timeshift = t_shift
        else:
            None
        print(fit.fun)
        
    print(fit.message)
    best_fit_parameters = fit.x

    I_fit = convoluted_exponential_decay(best_fit_parameters,t,irf,best_fit_timeshift)
    I_res = decay - I_fit

    n = len(decay) # number of data points
    p = len(initial_guess) # number of fitting parameters

    # if fitting_method == 'MLE':
    #     chi2 = 2*np.sum(I_fit - decay * np.log(I_fit)) #+ np.log(factorial(decay))) # equation 7 Bajzer 1991
    # elif fitting_method == 'LS':
    #     chi2 = np.sum((I_res**2)/I_fit) # equation 9 Bajzer 1991. alternatively you could also divide by I_data but you have to deal with the Infs
    # weighted_chi2 = chi2 / (n-p)
    # print(weighted_chi2)

    return fit,I_fit

def phasor_approach(t: np.ndarray,
                    decay: np.ndarray,
                    ref_decay: np.ndarray,
                    ref_lifetime: float,
                    laser_freq: float = []):
     # units: nanoseconds. e.g. ref_life_time = 5e-9. 0e-9 for irf
     # script adapted from Stefl et al. Anal Biochem. 2011 March 1; 410(1): 62–69. doi:10.1016/j.ab.2010.11.010.
     # see also Ranjit et al. https://doi.org/10.1038/s41596-018-0026-5

     t = t*1e-9

     dt = t[1] - t[0] # time step
     bgr = np.mean(decay[-10:-1]) # background intensity calculated from tail
     ref_bgr = np.mean(ref_decay[-10:-1])

     decay = decay - bgr # subtract background intensity
     #decay[decay<0] = 0  # used in https://zenodo.org/record/159557#.XpXhUJmxU2w
     ref_decay = ref_decay - ref_bgr

     i_max = np.where(ref_decay == np.max(ref_decay))[0][0] # determine t=0 from max of reference decay
     print(t[i_max])
     t -= t[i_max]

     plt.figure()
     plt.plot(t,decay)
     plt.plot(t,ref_decay)

     # angular frequency
     if laser_freq:
         freq = 2 * np.pi * laser_freq
     else:
         t_tot = np.max(t) - np.min(t)
         freq = 2 * np.pi / t_tot
    
    # cosine and sine terms
     cos_term = lambda decay_data : np.sum(decay_data * np.cos(freq * t) * dt) / np.sum(decay_data * dt)
     sin_term = lambda decay_data : np.sum(decay_data * np.sin(freq * t) * dt) / np.sum(decay_data * dt)

    # calculate phasors
     u = cos_term(decay)
     v = sin_term(decay)

     ref_u = cos_term(ref_decay)
     ref_v = sin_term(ref_decay)

    # correct negative values
     u = (0 if u<0 else u)
     v = (0 if v<0 else v)

    # correction factor from reference decay
     ref_m = (1+(freq * ref_lifetime)**2)**(-1/2)
     ref_ph = np.arctan(freq * ref_lifetime)

     cor_m = np.sqrt(ref_u**2 + ref_v**2) / ref_m
     cor_ph = -1 * np.arctan2(ref_v,ref_u) + ref_ph

    # correct phasor (Stefl et al. 2011)
     correct_u = lambda uu,vv : (uu * np.cos(cor_ph) - vv * np.sin(cor_ph)) / cor_m
     correct_v = lambda uu,vv : (uu * np.sin(cor_ph) + vv * np.cos(cor_ph)) / cor_m

     ref_u_cor = correct_u(ref_u,ref_v)
     ref_v_cor = correct_v(ref_u,ref_v)

     u_cor = correct_u(u,v)
     v_cor = correct_v(u,v)

     return u, v, u_cor,v_cor, ref_u_cor,ref_v_cor,cor_m,cor_ph,freq

 
 
def phasor_plot(
        u_list: np.ndarray,
        v_list: np.ndarray,
        freq: float,
        ref_decay_rates: np.ndarray, 
        labels = np.ndarray
        ):
    
    ref_lifetimes = 1/ref_decay_rates
    ref_u = 1 / (1+(freq * ref_lifetimes)**2)
    ref_v = (freq * ref_lifetimes) / (1+(freq * ref_lifetimes)**2)
    print(ref_u)
    print(ref_v)

    unity_circle = plt.Circle((0.5, 0), 0.5, color='k',fill=False)

    fig = plt.figure()
    plt.clf()
    plt.xlim(0,1.1)
    plt.ylim(0,0.6)
    plt.xlabel('u')
    plt.ylabel('v')
    fig.gca().add_patch(unity_circle)
    fig.gca().set_box_aspect(0.6/1.1)

    for i, txt in enumerate(labels):
        fig.gca().annotate(str(txt), (ref_u[i], ref_v[i]))

    plt.plot(ref_u,ref_v,'xk')
    plt.plot(u_list,v_list,'or')






#%% generate decays

t = np.linspace(0,500,200)
irf = t*0
irf[15:17] = 1
irf = irf / np.sum(irf)
decay_parameters = [500,0.02,20]
t_shift = 0

exps = []
exp_0 = convoluted_exponential_decay(decay_parameters,t,irf,t_shift)
for i in np.arange(40):
    exp = np.random.poisson(exp_0)
    exps.append(exp)
#%%
initial_guess = [100,0.06,20]
t_shift_range = [t_shift-2,t_shift+2]

# initial_fit = convoluted_exponential_decay(initial_guess,t,irf,t_shift)

# M = merit_function(t,exp,initial_fit,'MLE')
fits = []
fitted_decay_rates = []

for i,exp in enumerate(exps):
    print(i)
    fit,final_fit = fit_exponential_decay(t,exp,irf,initial_guess,t_shift_range,'MLE')
    fits.append(fit)
    fitted_decay_rates.append(fit.x[0])

fitted_decay_rates=np.array(fitted_decay_rates)

#%%
print(fit.x)
# print(np.diagonal(fit.hess_inv.todense()))
print(np.diagonal(fit.hess_inv))

#%%

fig = plt.figure()
fig.clf()
ax = fig.subplots(2,1)

ax[0].plot(irf*decay_parameters[0]+decay_parameters[2],color='gray',label = 'irf')
ax[0].plot(exp,color='k',label = 'data')
# plt.plot(initial_fit,color='g',label='intial fit')
ax[0].plot(final_fit,color='r',label = 'converged fit')
ax[0].set_yscale('log')
ax[0].legend()

ax[1].plot(t,final_fit-exp)
fig.show()
#%%
plt.figure()
plt.scatter


#%%
phasor = phasor_approach(t,exp,irf,0,[])

print(1/phasor[-1]*phasor[3]/phasor[2]*1e9)
print(1/decay_parameters[1])
#%%
ref_decay_rates = np.array([0.01,0.02,0.05,0.1,0.2,1])
phasor_plot([phasor[2]],[phasor[3]],phasor[-1],ref_decay_rates*1e9,labels=(1/ref_decay_rates).astype(int))

# %% to do
# find solution for time shift fit. currently it doesnt fit it. overrule and use value in search range around IRF value
# ensure that inf values for bgr = 0 do not fail fit

# select the proper optimizer function. e.g. trust-constr with 2 point or 3 points. something that outputs the hessian
# find proper way to find hessian and estimate errors in fitted parameters
# check if with contraint, the time shift can be fitted too. 



# %%
