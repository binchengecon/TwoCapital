
from isort import file
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import array, loadtxt
import numpy as np

import sympy
import csv
import pandas as pd



#########################################
gamma_3_list = np.linspace(0., 1./3., 10)
eta_list     = np.array([0.1,0.01,0.001])

(gamma_3_list,eta_list) = np.meshgrid(gamma_3_list,eta_list,indexing='ij')

gamma_3_list = gamma_3_list.ravel(order='F')
eta_list = eta_list.ravel(order='F')

param_list = list(zip(gamma_3_list,eta_list))
#########################################



for num in range(len(param_list)):
    gamma = param_list[num][0]
    eta=param_list[num][1]

    file_name = "./data/PostJump/"+"gamma-" + str(gamma)+"eta-"+str(eta)
    file = open(file_name+'.csv','r')
    reader = csv.reader(file,delimiter=',')
    file_header= next(reader)
    file_varnum = len(file_header)

    data = np.array(list(reader)).astype(float)


    figwidth = 10
    fig, axs = plt.subplots(int(file_varnum/2), 2, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  


    for varnum in np.array(range(file_varnum)):
        axs[varnum//2,varnum%2].plot(data[:,varnum])
        axs[varnum//2,varnum%2].set_ylabel('%s' %file_header[varnum],fontsize = 16)
        axs[varnum//2,varnum%2].set_title('%s' %file_header[varnum])
        axs[varnum//2,varnum%2].grid(linestyle=':')
        axs[varnum//2,varnum%2].legend()
    
    
    plt.savefig(f"../tech4D/data/PostJump/gamma_{str(gamma)[:5]}_eta_{eta}.pdf")




