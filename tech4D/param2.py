from library import *


tol = 1e-7
epsilon  = 0.1
fraction = 0.1
max_iter = 20
path_name = "./tech4D/data/PostJump/"
test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])


