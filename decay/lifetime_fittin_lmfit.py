#%% libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import factorial

from lmfit import Minimizer, Parameters, create_params, report_fit

# %%

def convoluted_exponential_decay(decay_parameters: Parameters,
                                 t: np.ndarray,
                                 irf: np.ndarray):
    
    # unpack fitting parameters
    decay_parameters_values = decay_parameters.valuesdict()
    amplitude = decay_parameters_values['amplitude']
    decay_rate = decay_parameters_values['decay_rate']
    background_intensity = decay_parameters_values['background_intensity']
    time_shift = decay_parameters_values['time_shift']

    # amplitude = np.abs(decay_parameters[0])
    # decay_rate = np.abs(decay_parameters[1])
    # t_shift = int(decay_parameters[2])
    # bgr = np.abs(decay_parameters[3])

    size = len(t)

    exp_decay = amplitude * np.exp(-decay_rate * (t - np.min(t)))
    conv_exp_decay = np.convolve(exp_decay,irf)

    if time_shift >= 0:
        conv_exp_decay_shifted = conv_exp_decay[time_shift : size + time_shift]
    else:
        conv_exp_decay_shifted = np.hstack((conv_exp_decay[time_shift:],conv_exp_decay[0 : size + time_shift]))

    conv_exp_decay_shifted = conv_exp_decay_shifted + background_intensity

    return conv_exp_decay_shifted


def log_likelihood_poisson(t: np.ndarray,
                    data: np.ndarray,
                    fit: np.ndarray,
                    ):

    M = np.sum(-1 * data * np.log(fit) + fit) # + np.log(np.fact)

    return M

def objective_function(fit_parameters: Parameters,
                   t: np.array,
                   decay: np.array,
                   irf: np.array,
                   ):

    # # unpack data dictionary
    # decay = data['decay']
    # irf = data['irf']
    # t = data['t']

    # fitting model
    fit = convoluted_exponential_decay(fit_parameters,t,irf)

    # likelihood of model describing data
    log_likelihood = log_likelihood_poisson(t,decay,fit)

    return log_likelihood

#%%
# def fit_exponential_decay(t: np.ndarray,
#                             decay: np.ndarray,
#                             irf: np.ndarray,
#                             initial_guess: tuple,
#                             t_shift_range: tuple,
#                             fitting_method: str = 'MLE'):
    
#     for i,t_shift in enumerate(np.arange(t_shift_range[0],t_shift_range[1]+1)): 

#         merit_func = lambda decay_parameters : merit_function(
#                     t,
#                     decay,
#                     convoluted_exponential_decay(decay_parameters,t,irf,t_shift),fitting_method)
        
#         _fit = optimize.minimize(merit_func,
#                                  initial_guess,
#                                  options={'gtol': 1e-9}
#                                  #bounds = ((0, None),(0,None),(0,None)),
#                                  #method = 'L-BFGS-B'
#                                  )
#                                 #  method = 'trust-ncg')
#                                 #  method = 'BFGS')
        
#         if (i == 0):
#             fit = _fit
#             best_fit_timeshift = t_shift
#         elif (_fit.fun < fit.fun):
#             fit = _fit
#             best_fit_timeshift = t_shift
#         else:
#             None
#         print(fit.fun)
        
#     print(fit.message)
#     best_fit_parameters = fit.x

#     I_fit = convoluted_exponential_decay(best_fit_parameters,t,irf,best_fit_timeshift)
#     I_res = decay - I_fit

#     n = len(decay) # number of data points
#     p = len(initial_guess) # number of fitting parameters

#     # if fitting_method == 'MLE':
#     #     chi2 = 2*np.sum(I_fit - decay * np.log(I_fit)) #+ np.log(factorial(decay))) # equation 7 Bajzer 1991
#     # elif fitting_method == 'LS':
#     #     chi2 = np.sum((I_res**2)/I_fit) # equation 9 Bajzer 1991. alternatively you could also divide by I_data but you have to deal with the Infs
#     # weighted_chi2 = chi2 / (n-p)
#     # print(weighted_chi2)

#     return fit,I_fit





#%% generate decays
# parameters

decay_parameters = Parameters()
decay_parameters.add('amplitude', value = 500, min = 0)
decay_parameters.add('decay_rate', value = 0.02, min = 0)
decay_parameters.add('background_intensity', value = 10, min = 0)
decay_parameters.add('time_shift', value = 3, min = 0, max = 6)

#

#%%



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
