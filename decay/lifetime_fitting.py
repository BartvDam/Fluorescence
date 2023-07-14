#%% libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# %%

def convoluted_exponential_decay(decay_parameters: list,
                                 t: np.ndarray,
                                 irf: np.ndarray):

    amplitude = decay_parameters[0]
    decay_rate = decay_parameters[1]
    t_shift = int(decay_parameters[2])
    bgr = decay_parameters[3]

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
                            fitting_method: str = 'MLE'):
    
    merit_func = lambda decay_parameters : merit_function(
                t,
                decay,
                convoluted_exponential_decay(decay_parameters,t,irf),fitting_method)
    
    fit = optimize.minimize(merit_func,
                    initial_guess,
                    method = 'BFGS')

    print(fit.message)
    best_fit_parameters = fit.x

    I_fit = convoluted_exponential_decay(best_fit_parameters,t,irf)

    return fit,I_fit
         






#%%
t = np.linspace(0,500,100)
irf = t*0
irf[15:16] = 1
irf = irf / np.sum(irf)
decay_parameters = [500,0.02,5,20]
exp = convoluted_exponential_decay(decay_parameters,t,irf)
exp = np.random.poisson(exp)

initial_guess = [100,0.15,4,15]
initial_fit = convoluted_exponential_decay(initial_guess,t,irf)

M = merit_function(t,exp,initial_fit,'MLE')
print(M)

fit,final_fit = fit_exponential_decay(t,exp,irf,initial_guess,'MLE')
print(fit.x)
print(np.diagonal(fit.hess_inv))

plt.figure()
plt.clf()
plt.plot(irf+decay_parameters[3],label = 'irf')
plt.plot(exp,label = 'data')
plt.plot(initial_fit,label='intial fit')
plt.plot(final_fit,label = 'converged fit')
plt.legend()
plt.show()



# %% to do
# find solution for time shift fit. currently it doesnt fit it. overrule and use value in search range around IRF value
# ensure that inf values for bgr = 0 do not fail fit