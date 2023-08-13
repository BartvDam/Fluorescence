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

    amplitude = decay_parameters[0]
    decay_rate = decay_parameters[1]
    # t_shift = int(decay_parameters[2])
    t_shift = t_shift
    bgr = decay_parameters[2]

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
                                 options={'gtol': 1e-9})
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
         






#%%
t = np.linspace(0,500,200)
irf = t*0
irf[15:17] = 1
irf = irf / np.sum(irf)
decay_parameters = [500,0.02,20]
t_shift = 2
exp = convoluted_exponential_decay(decay_parameters,t,irf,t_shift)
exp = np.random.poisson(exp)
#%%
initial_guess = [100,0.6,20]
t_shift_range = [2,7]
initial_fit = convoluted_exponential_decay(initial_guess,t,irf,t_shift)

M = merit_function(t,exp,initial_fit,'MLE')
print(fit)

fit,final_fit = fit_exponential_decay(t,exp,irf,initial_guess,t_shift_range,'MLE')
print(fit.x)
print(np.diagonal(fit.hess_inv))

plt.figure()
plt.clf()
plt.plot(irf+decay_parameters[2],color='gray',label = 'irf')
plt.plot(exp,color='k',label = 'data')
plt.plot(initial_fit,color='g',label='intial fit')
plt.plot(final_fit,color='r',label = 'converged fit')
plt.yscale('log')
plt.legend()
plt.show()



# %% to do
# find solution for time shift fit. currently it doesnt fit it. overrule and use value in search range around IRF value
# ensure that inf values for bgr = 0 do not fail fit