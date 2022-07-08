import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
from src.supportfunctions import finiteDiff_3D
import os

from scipy import optimize


delta = 0.01
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
sigma_k = 0.0095
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
psi_1 = 1/2
sigma_g = 0.016
gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2
gamma_3_list = np.linspace(0., 1./3., 10)
y_bar = 2.
y_bar_lower = 1.5


# Tech
theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.

K_min = 4.00
K_max = 9.00
hK    = 0.20
K     = np.arange(K_min, K_max + hK, hK)
nK    = len(K)
Y_min = 0.
Y_max = 5.
hY    = 0.20 # make sure it is float instead of int
Y     = np.arange(Y_min, Y_max + hY, hY)
nY    = len(Y)
L_min = - 5.
L_max = - 0.
hL    = 0.2
L     = np.arange(L_min, L_max,  hL)
nL    = len(L)

id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = 0.
Y_max_short = 3.
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)
# print("bY_short={:d}".format(nY_short))
(K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_short, L, indexing="ij")

stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), L_mat.reshape(-1, 1, order='F')])



initial=(np.log(85/0.115), 1.1, np.log(1/80))

LogK_0, Y_0, LogL_0 = initial
T0=0
T=80
dt=1/3

psi_0 = 0.0005

a = kappa/delta
b = -(1+alpha*kappa)/delta
c = alpha/delta-1

i = (-b - np.sqrt(b**2-4*a*c))/(2*a)
x = 0.004 * alpha * np.exp(LogK_0)

def mu_K(i_x):
    return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2

def mu_L(Xt, state):
    return -zeta + psi_0 * (Xt * (np.exp(state[0] - state[2]) ) )**psi_1 - 0.5 * sigma_g**2

years  = np.arange(T0, T0 + T + dt, dt)
pers   = len(years)

hist      = np.zeros([pers, 3])
i_hist    = np.zeros([pers])
x_hist    = np.zeros([pers])

mu_K_hist = np.zeros([pers])
mu_L_hist = np.zeros([pers])

for tm in range(pers):
    if tm == 0:
        # initial points
        hist[0,:] = [LogK_0, Y_0, LogL_0] # logL
        i_hist[0] = i
        x_hist[0] = x
        mu_K_hist[0] = mu_K(i_hist[0])
        mu_L_hist[0] = mu_L(x_hist[0], hist[0,:])

    else:

        i_hist[tm] = i
        x_hist[tm] = 0.004 * alpha * np.exp(hist[tm-1,0])

        

        mu_K_hist[tm] = mu_K(i_hist[tm])
        mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])

        hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
        hist[tm,1] = hist[tm-1,1] 
        hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ

true_tech_intensity = np.exp(hist[:, 2]) 
true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]

temp = np.abs(true_tech_prob[-1]-0.995)
# temp

# true_tech_prob[-1]-0.995

true_tech_prob
pers