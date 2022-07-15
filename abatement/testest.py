# %%

import numpy as np

# %%

import pandas as pd
import sys
print(sys.path)


import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse

K_min = 4.00
K_max = 9.00
hK    = 0.20
K     = np.arange(K_min, K_max + hK, hK)
nK    = len(K)

y_bar =2

Y_min = 0.
Y_max = 4.
hY    = 0.10 # make sure it is float instead of int
Y     = np.arange(Y_min, Y_max + hY, hY)
nY    = len(Y)

id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = 0.
Y_max_short = 3.
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)

L_min = - 5.5
L_max = - 0.
hL    = 0.20
L     = np.arange(L_min, L_max,  hL)
nL    = len(L)


xi_a=1000.
xi_g=1000.

psi_0=0.008
psi_1=0.8

name="midwaynew"
# data

gridpoints = (K, Y_short, L)




with open("/home/bincheng/TwoCapital_Bin/abatement/data_2tech/midwaynew_Psi0_Psi1_xi_a_1000.0_xi_g_1000.0_psi_0_0.01_psi_1_0.8_model_tech2_pre_damage", "rb") as f:
    tech1 = pickle.load(f)

x = tech1["x_star"][ 13:16, :, 6]


x_func = RegularGridInterpolator(gridpoints, tech1["x_star"])

print(x_func([np.log(85/0.115),1.1,np.log(1/80)]))
# pdf
PDF_Dir = "/home/bincheng/TwoCapital_Bin/abatement/pdf_2tech/"
File_Dir = "result_inspection" 

pdf_pages = PdfPages(PDF_Dir+File_Dir+'.pdf')


fig1, axs1 = plt.subplots(1, 1, sharex=False, figsize=(8, 5))  


axs1.plot(x.T)


pdf_pages.savefig(fig1)
plt.close()

pdf_pages.close()
