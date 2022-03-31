import numpy as np
import pandas as pd
from PostSolver import hjb_post_damage_post_tech
from PreSolver import hjb_post_damage_pre_tech
import pickle
##################
xi_a = 1000.
xi_p = 1000.
xi_b = 1000.
xi_g = 1000.
arrival = 20
y_bar = 2.
y_bar_lower = 1.5
n_model = 20
##################

I_g_first   = 1./arrival
I_g_second  = 1./arrival
# xi_g_first  = ξ_p
# xi_g_second = ξ_p
# xi_b = ξ_p

# Model parameters
delta   = 0.010
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
# Technology
theta        = 2 # 3
lambda_bar   = 0.1206
vartheta_bar = 0.0453

gamma_1 = 1.7675/10000
gamma_2 = .0022*2

gamma_3_lower = 0.
gamma_3_upper = 1./3

# Compute gamma_3 for the n models
def Γ(y, y_bar, gamma_1, gamma_2, gamma_3):
    return gamma_1 * y + gamma_2 / 2 * y ** 2 + gamma_3 / 2 * (y > y_bar) * (y - y_bar) ** 2

prop_damage_lower = np.exp(-Γ(2.5, 2., gamma_1, gamma_2, gamma_3_upper))
prop_damage_upper = np.exp(-Γ(2.5, 2., gamma_1, gamma_2, gamma_3_lower))
gamma_3 = (-np.log(np.linspace(prop_damage_lower, prop_damage_upper, n_model)) - gamma_1 * 2.5 - gamma_2 / 2 * 2.5**2) / .5**2 * 2
gamma_3.sort()
gamma_3[0] = 0
πd_o = np.ones(n_model)/n_model

theta_ell = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o = np.ones_like(theta_ell)/len(theta_ell)
sigma_y = 1.2 * np.mean(theta_ell)

# Grid setting
k_step = .1
k_grid = np.arange(4., 8.5 + k_step, k_step)
nk = len(k_grid)
y_step = .1
y_grid_long = np.arange(0., 3. +y_step, y_step)
y_grid_short = np.arange(0., 2.5+y_step, y_step)
ny = len(y_grid_long)
# n_bar = find_nearest_value(y_grid_long, τ) + 1

logI_step = 0.1
logI_min  = -2.
logI_max  = -0.
logI_grid = np.arange(logI_min, logI_max + logI_step, logI_step)
nlogI = len(logI_grid)

zeta    = 0.0
psi_0   = 10.
psi_1   = 1
sigma_g = 0.0
# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
vartheta_bar_second = 0.

gamma_3_i = 0.

# After second jump
model_args = (delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3_i, y_bar, theta, lambda_bar_first, vartheta_bar_first)


model_res = hjb_post_damage_post_tech(
        k_grid, y_grid_long, model_args, v0=None, 
        epsilon=1., fraction=.5,tol=1e-8, max_iter=2000, print_iteration=True)

with open("./res_data/post_jump", "wb") as f:
    pickle.dump(model_res, f)

with open("./res_data/post_jump", "rb") as f:
    model_res = pickle.load(f)

v_post = model_res["v"]

V_post = np.zeros((nk, ny, nlogI))
for i in range(nlogI):
    V_post[:,:,i] = v_post

model_args_pre = ( delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_g, v_post, gamma_1, gamma_2, gamma_3_i, y_bar, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar_first + 1e-8, vartheta_bar_first + 1e-8)

res_pre = hjb_post_damage_pre_tech(
        k_grid, y_grid_long, logI_grid, model_args_pre, v0=V_post, 
        ε=0.1, fraction = 0.1, tol=1e-15
        )
with open("./res_data/pre_jump", "wb") as f:
    pickle.dump(res_pre, f, protocol=pickle.HIGHEST_PROTOCOL)
