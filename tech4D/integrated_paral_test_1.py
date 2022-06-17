#########################################
# Optimization of post jump HJB
#########################################


#########################################
# Library Loading
#########################################

import os
import sys
sys.path.append('../src')
import csv
from supportfunctions import *
sys.stdout.flush()

from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
import argparse
import numpy as np
import joblib
from joblib import Parallel, delayed
import pickle

Attemp = 6

# Attemp = 1: add more output of variable like multi1,2 to diagnose where the problem is.  change starting multi* as 1.
# Attemp = 2: delete multi2[<1e-8]=1e-8 to see the explosion. This is for presentation to Lars.
# Attemp = 3: Maybe epsilon=0.1 worked, I need to check.
# Attemp = 4: delete multi2[<1e-8] and see the effect, previous results didnt get stored.
# Attemp = 5: freeze id=0 and see what is the effect.
# Attemp = 6: add pickle to save ultimate convergent result


path_name = "./tech4D/data/PostJump/"

# test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])
epsilon_list = np.array([0.1])





############# step up of optimization
# FC_Err = 1
# epoch = 0
# tol = 1e-7
# epsilon  = 0.005
# fraction = 0.005
max_iter = 1

# id_star = np.zeros_like(K_mat)
# ig_star = np.zeros_like(K_mat)

# #########################################
# # Result Storage Setup
# #########################################
(gamma_3_list,eta_list,epsilon_list) = np.meshgrid(gamma_3_list,eta_list,epsilon_list,indexing='ij')

gamma_3_list = gamma_3_list.ravel(order='F')
eta_list = eta_list.ravel(order='F')
epsilon_list = epsilon_list.ravel(order='F')

param_list = zip(gamma_3_list,eta_list,epsilon_list)

#########################################
epoch_list = list(range(1,max_iter+1,1))

def model(gamma_3, eta, epsilon ):
    A=1
    # filename = filename
    my_shelf = {}
    for key in dir():
        print(dir())
        print( globals()[key])
        # if isinstance(globals()[key], (int,float, float, str, bool, np.ndarray,list)):
        #     try:
        #         my_shelf[key] = globals()[key]
        #     except TypeError:
        #         #
        #         # __builtins__, my_shelf, and imported modules can not be shelved.
        #         #
        #         print('ERROR shelving: {0}'.format(key))
        # else:
        #     pass


    file = open(path_name+"test", 'wb')
    pickle.dump(my_shelf, file)
    file.close()




number_of_cpu = joblib.cpu_count()
delayed_funcs = [delayed(model)(gamma_3, eta, epsilon) for gamma_3, eta, epsilon in param_list]
parallel_pool = Parallel(n_jobs=number_of_cpu,require = 'sharedmem')
res = parallel_pool(delayed_funcs)
