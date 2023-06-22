#%% test2

print('test')
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

    conv_exp_decay_shifted = irf * 0

    if t_shift > 0:
        conv_exp_decay_shifted = conv_exp_decay[t_shift : size + t_shift]
    else:
        conv_exp_decay_shifted = np.hstack((conv_exp_decay[t_shift:],conv_exp_decay[0 : size + t_shift]))

    conv_exp_decay_shifted = conv_exp_decay_shifted + bgr

    def merit_function(t: np.ndarray,
                       decay: np.ndarray,
                       fit: np.ndarray,
                       method: str):
            
            if method == 'MLE':
                M = np.sum(decay * np.log(fit) + fit) # + np.log(np.fact)
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
                           method = 'Newton-CG')
         
         return fit
         






#%%
t = np.linspace(0,100,100)
irf = t*0
irf[15:20] = 0.2
exp = convoluted_exponential_decay([1,0.1,2],t,irf,2)
# fit = fit_exponential_decay(t,)

print(len(exp))

plt.figure()
plt.plot(irf)
plt.plot(exp)
plt.show()
# %%
exp
# %%
