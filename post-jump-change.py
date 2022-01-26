# Optimization of poat jump HJB
#Required packages
import os
import sys
sys.path.append('./src')
from supportfunctions import *
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime

reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'petsc'

write_test = False
if write_test:
    f = open("test-log.txt", 'a')


current_time = datetime.now()
filename =  "res" + '_' + "{:d}-{:d}-{:d}".format(current_time.day, current_time.hour, current_time.minute)

if write_test:
    f.write("Script starts: {:d}/{:d}-{:d}:{:d}\n".format(current_time.month, current_time.day, current_time.hour, current_time.minute))
    f.write("Linear solver: " + linearsolver+ "\n" )


start_time = time.time()
# Parameters as defined in the paper
delta = 0.01
A_d = 0.12
A_g = 0.2

alpha_d = -0.02
alpha_g = -0.02
sigma_d = 0.016
sigma_g = 0.016

varsigma = 1.2 * 1.86 / 1000
phi_d = 7
phi_g = 7
########## arrival rate
varphi = 0.1
sigma_lam = 0.016
########## Scaling factor
eta = 0.17


###### damage
gamma_1 = 0.00017675
gamma_2 = 2. * 0.0022

y_bar = 2
beta_f = 1.86 / 1000

# Grids Specification

# Coarse Grids
# R_min = 0
# R_max = 9
# F_min = 0
# F_max = 4000
# K_min = 0
# K_max = 18
# nR = 4
# nF = 4
# nK = 4
# R = np.linspace(R_min, R_max, nR)
# F = np.linspace(F_min, F_max, nF)
# K = np.linspace(K_min, K_max, nK)

# hR = R[1] - R[0]
# hF = F[1] - F[0]
# hK = K[1] - K[0]

# Dense Grids

#R_min = 0
#R_max = 9

Y_min = 0.
Y_max = 4.
# range of capital
Kd_min = 0.
Kd_max = 9.
Kg_min = 0.1
Kg_max = 0.99
################### arrival rate######################
lam_min = 0
lam_max = 0.1
hlam = 0.01

# hR = 0.05
hY  = 0.1 # make sure it is float instead of int
hKd = 0.4
hKg = 0.1

# R = np.arange(R_min, R_max + hR, hR)
# nR = len(R)
Y = np.arange(Y_min, Y_max + hY, hY)
nY = len(Y)
Kd = np.arange(Kd_min, Kd_max + hKd, hKd)
nKd = len(Kd)
Kg = np.arange(Kg_min, Kg_max + hKg, hKg)
nKg = len(Kg)
lam = np.arange(lam_min, lam_max + hlam, hlam)
nlam = len(lam)


if write_test:
    f.write("Grid dimension: [{}, {}, {}]\n".format(nKd, nKg, nY))
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(Kd_mat, Kg_mat, Y_mat) = np.meshgrid(Kd, Kg, Y, indexing = 'ij')
stateSpace = np.hstack([Kd_mat.reshape(-1,1,order = 'F'), Kg_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1, 1, order='F')])

# For PETSc
Kd_mat_1d =Kd_mat.ravel(order='F')
Kg_mat_1d = Kg_mat.ravel(order='F')
Y_mat_1d = Y_mat.ravel(order='F')
lowerLims = np.array([Kd_min, Kg_min, Y_min], dtype=np.float64)
upperLims = np.array([Kd_max, Kg_max, Y_max], dtype=np.float64)


v0 = Kd_mat - beta_f * Y_mat
# import pickle
# data = pickle.load(open("data/res_13-1-37", "rb"))
# v0 = data["v0"]
############# step up of optimization
FC_Err = 1
epoch = 0
tol = 1e-8
epsilon = 0.01
fraction = 0.01

max_iter = 40000
# file_iter = open("iter_c_compile.txt", "w")
while FC_Err > tol and epoch < max_iter:
    print("-----------------------------------")
    print("---------Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    ######## first order
    dKd = finiteDiff(v0,0,1,hKd)
    # dKd[dKd < 1e-8] = 1e-8
    dKg = finiteDiff(v0,1,1,hKg)
    # dKg[dKg < 1e-8] = 1e-8
    dY = finiteDiff(v0,2,1,hY)
    ######## second order
    ddKd = finiteDiff(v0,0,2,hKd)
    ddKg = finiteDiff(v0,1,2,hKg)
    ddY = finiteDiff(v0,2,2,hY)

    # if epoch > 2000:
        # epsilon = 0.01
    # elif epoch > 1000:
        # epsilon = 0.05
    # else:
        # pass

    # update control
    if epoch == 0:
        # i_d = np.zeros(Kd_mat.shape)
        # i_g = np.zeros(Kg_mat.shape)
        consumption_0 = A_d * (1 - Kg_mat) + A_g * Kg_mat
        consumption = consumption_0
        mc = delta / consumption
        i_d = 1 -  mc / (dKd - Kg_mat *  dKg)
        i_d /= phi_d
        i_d[i_d < 0] = 0
        # i_d[i_d > A_d] = A_d
        i_g = 1 - mc / (dKd + (1 - Kg_mat) * dKg)
        i_g /= phi_g
        # i_g[i_g < 0] = 0
        # i_g[i_g > A_g] = A_g
    else:

        # consumption_new = (A_d - id_star) * Kd_mat + (A_g - ig_star) * Kg_mat
        # consumption_new[consumption_new <= 1e-8] = 1e-8
        # consumption_new[consumption_new > consumption_0] = consumption_0[consumption_new > consumption_0]
        # mc = delta / consumption_new
        nums = 0
        converge = False
        while not converge:

            id_new = 1 / phi_d * (1 - mc / (dKd - Kg_mat * dKg )) * fraction + i_d * (1 - fraction)
            id_new[id_new < 0] = 0
            id_new[id_new > A_d] = A_d- 1e-8
            # id_new[id_new > 1/phi_d] = 1 / phi_d
            ig_new = 1 / phi_g * (1 - mc / (dKd  + (1 - Kg_mat) * dKg)) * fraction + i_g * (1 - fraction)
            ig_new[ig_new < 0] = 0
            # ig_new[ig_new > A_g] = A_g
            ig_new[ig_new > 1 / phi_g] = 1 / phi_g

            mc_new = fraction * delta / ((A_d -id_new) * (1 - Kg_mat) + (A_g - ig_new) * Kg_mat) + mc * (1 - fraction)
            i_d = id_new
            i_g = ig_new
            diff = np.max(np.abs(mc - mc_new) / fraction)
            if diff  < 1e-5 or nums > 10000:
                converge = True
                mc = mc_new
            else:
                mc = mc_new
                pass
            nums += 1

        print(diff)
    print(np.min(i_d), np.min(i_g))
    # i_d = np.zeros(Kd_mat.shape)
    # i_g = np.zeros(Kg_mat.shape)
    # i_d[i_d < 0] = 0
    # i_g[i_g < 0] = 0
    consumption = (A_d -i_d) * (1 - Kg_mat) + (A_g - i_g) * Kg_mat
    consumption[consumption < 1e-8] = 1e-8
    # i_d[i_d >= A_d] = A_d - 1e-8
    # i_g[i_g >= A_g] = A_g - 1e-8
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # See remark 2.1.3 for more details
    start_time2 = time.time()
    if epoch == 0:
        dVec = np.array([hKd, hKg, hY])
        increVec = np.array([1, nKd, nKd * nKg],dtype=np.int32)
        # These are constant
        A = - delta * np.ones(Kd_mat.shape)
        C_dd = 0.5 * ( sigma_d * (1 - Kg_mat) + sigma_g * Kg_mat )**2
        C_gg = np.zeros(Kd_mat.shape)
        C_yy = 0.5 * (eta * varsigma * A_d * np.exp(Kd_mat) * (1 - Kg_mat))** 2
        if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
            petsc_mat = PETSc.Mat().create()
            petsc_mat.setType('aij')
            petsc_mat.setSizes([nKd*nKg*nY, nKd*nKg*nY])
            petsc_mat.setPreallocationNNZ(13)
            petsc_mat.setUp()
            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.setType('bcgs')
            ksp.getPC().setType('ilu')
            ksp.setFromOptions()

            A_1d = A.ravel(order = 'F')
            C_dd_1d = C_dd.ravel(order = 'F')
            C_gg_1d = C_gg.ravel(order = 'F')
            C_yy_1d = C_yy.ravel(order = 'F')

            if linearsolver == 'petsc4py':
                I_LB_d = (stateSpace[:,0] == Kd_min)
                I_UB_d = (stateSpace[:,0] == Kd_max)
                I_LB_g = (stateSpace[:,1] == Kg_min)
                I_UB_g = (stateSpace[:,1] == Kg_max)
                I_LB_y = (stateSpace[:,2] == Y_min)
                I_UB_y = (stateSpace[:,2] == Y_max)
                diag_0_base = A_1d[:]
                diag_0_base += (I_LB_d * C_dd_1d[:] + I_UB_d * C_dd_1d[:] - 2 * (1 - I_LB_d - I_UB_d) * C_dd_1d[:]) / dVec[0] ** 2
                diag_0_base += (I_LB_g * C_gg_1d[:] + I_UB_g * C_gg_1d[:] - 2 * (1 - I_LB_g - I_UB_g) * C_gg_1d[:]) / dVec[1] ** 2
                diag_0_base += (I_LB_K * C_kk_1d[:] + I_UB_K * C_kk_1d[:] - 2 * (1 - I_LB_K - I_UB_K) * C_kk_1d[:]) / dVec[2] ** 2
                diag_d_base = - 2 * I_LB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                diag_dm_base = - 2 * I_UB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                diag_g_base = - 2 * I_LB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                diag_gm_base = - 2 * I_UB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                diag_y_base = - 2 * I_LB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                diag_ym_base = - 2 * I_UB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                diag_dd = I_LB_d * C_dd_1d[:] / dVec[0] ** 2
                diag_ddm = I_UB_d * C_dd_1d[:] / dVec[0] ** 2
                diag_gg = I_LB_g * C_gg_1d[:] / dVec[1] ** 2
                diag_ggm = I_UB_g * C_gg_1d[:] / dVec[1] ** 2
                diag_yy = I_LB_y * C_yy_1d[:] / dVec[2] ** 2
                diag_yym = I_UB_y * C_yy_1d[:] / dVec[2] ** 2


    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details

    B_d = (alpha_d + i_d - 0.5* phi_d * i_d**2) * (1 - Kg_mat) +  (alpha_g + i_g - 0.5 * phi_g * i_g**2) * Kg_mat - C_dd
    B_g = ((alpha_g + i_g - 0.5 * phi_g * i_g**2) -  (alpha_d + i_d - 0.5* phi_d * i_d**2)) * Kg_mat * (1 - Kg_mat)
    B_y = beta_f * eta * A_d * np.exp(Kd_mat) * (1 - Kg_mat)

    D = delta * np.log(consumption) + delta * Kd_mat  - (gamma_1 + gamma_2 * Y_mat)* beta_f * eta * A_d * np.exp(Kd_mat) * (1 - Kg_mat)  - 0.5 * gamma_2 * (varsigma * eta * A_d * np.exp(Kd_mat) * (1 - Kg_mat) )**2


    if linearsolver == 'petsc4py':
        bpoint1 = time.time()
        # ==== original impl ====
        B_d_1d = B_d.ravel(order = 'F')
        B_g_1d = B_g.ravel(order = 'F')
        B_y_1d = B_y.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        # profiling
        # bpoint2 = time.time()
        # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        diag_0 = diag_0_base - 1 / epsilon + I_LB_R * B_r_1d[:] / -dVec[0] + I_UB_R * B_r_1d[:] / dVec[0] - (1 - I_LB_R - I_UB_R) * np.abs(B_r_1d[:]) / dVec[0] + I_LB_F * B_f_1d[:] / -dVec[1] + I_UB_F * B_f_1d[:] / dVec[1] - (1 - I_LB_F - I_UB_F) * np.abs(B_f_1d[:]) / dVec[1] + I_LB_K * B_k_1d[:] / -dVec[2] + I_UB_K * B_k_1d[:] / dVec[2] - (1 - I_LB_K - I_UB_K) * np.abs(B_k_1d[:]) / dVec[2]
        diag_R = I_LB_R * B_r_1d[:] / dVec[0] + (1 - I_LB_R - I_UB_R) * B_r_1d.clip(min=0.0) / dVec[0] + diag_R_base
        diag_Rm = I_UB_R * B_r_1d[:] / -dVec[0] - (1 - I_LB_R - I_UB_R) * B_r_1d.clip(max=0.0) / dVec[0] + diag_Rm_base
        diag_F = I_LB_F * B_f_1d[:] / dVec[1] + (1 - I_LB_F - I_UB_F) * B_f_1d.clip(min=0.0) / dVec[1] + diag_F_base
        diag_Fm = I_UB_F * B_f_1d[:] / -dVec[1] - (1 - I_LB_F - I_UB_F) * B_f_1d.clip(max=0.0) / dVec[1] + diag_Fm_base
        diag_K = I_LB_K * B_k_1d[:] / dVec[2] + (1 - I_LB_K - I_UB_K) * B_k_1d.clip(min=0.0) / dVec[2] + diag_K_base
        diag_Km = I_UB_K * B_k_1d[:] / -dVec[2] - (1 - I_LB_K - I_UB_K) * B_k_1d.clip(max=0.0) / dVec[2] + diag_Km_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_R, diag_Rm, diag_RR, diag_RRm, diag_F, diag_Fm, diag_FF, diag_FFm, diag_K, diag_Km, diag_KK, diag_KKm]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        b = -v0_1d/epsilon - D_1d
        # A_sp = spdiags(data, diags, len(diag_0), len(diag_0))
        # A_sp = csr_matrix(A_sp.T)
        # b = -v0/ε - D
        # profiling
        # bpoint4 = time.time()
        # print("create matrix and rhs: {:.3f}s".format(bpoint4 - bpoint3))
        petsc_mat = PETSc.Mat().createAIJ(size=A_sp.shape, csr=(A_sp.indptr, A_sp.indices, A_sp.data))
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint5 = time.time()
        # print("assemble: {:.3f}s".format(bpoint5 - bpoint4))

        # dump to files
        #x.set(0)
        #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_A.dat', 'w')
        #petsc_mat.view(viewer)
        #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        #petsc_rhs.view(viewer)

        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_d * dKd + B_g * dKg + B_y * dY + C_dd * ddKd + C_gg * ddKg + C_yy * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            # profling
            # bpoint7 = time.time()
            # print("compute error: {:.3f}s".format(bpoint7 - bpoint6))
        # if linearsolver == 'both':
            # compare
            # csr_mat = csr_mat*(-ε)
            # b = b*(-ε)
            # A_diff =  np.max(np.abs(out_eigen[3] - csr_mat))
            #
            # print("Coefficient matrix difference: {:.3f}".format(A_diff))
            # b_diff = np.max(np.abs(out_eigen[4] - np.squeeze(b)))
            # print("rhs difference: {:.3f}".format(b_diff))

    if linearsolver == 'petsc' or linearsolver == 'both':
        bpoint1 = time.time()
        B_d_1d = B_d.ravel(order = 'F')
        B_g_1d = B_g.ravel(order = 'F')
        B_y_1d = B_y.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(Kd_mat_1d, Kg_mat_1d, Y_mat_1d, A_1d, B_d_1d, B_g_1d, B_y_1d, C_dd_1d, C_gg_1d, C_yy_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # profiling
        # bpoint2 = time.time()
        # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        b = v0_1d + D_1d*epsilon
        # petsc4py setting
        # petsc_mat.scale(-1./ε)
        # b = -v0_1d/ε - D_1d
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint3 = time.time()
        # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(Kd_mat.shape,order = "F")
        end_ksp = time.time()
        # profiling
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        num_iter = ksp.getIterationNumber()
        # file_iter.write("%s \n" % num_iter)
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_d * dKd + B_g * dKg + B_y * dY + C_dd * ddKd + C_gg * ddKg + C_yy * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
    # step 9: keep iterating until convergence
    id_star = i_d
    ig_star = i_g
    v0 = out_comp
    epoch += 1
if reporterror:
    print("===============================================")
    print("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch -1, PDE_Err, FC_Err))
print("--- Total running time: %s seconds ---" % (time.time() - start_time))

if write_test:
    f.write("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}\n" .format(epoch -1, PDE_Err, FC_Err))
    f.write("--- Total running time: %s seconds ---\n" % (time.time() - start_time))

# exit()

import pickle
# filename = filename
my_shelf = {}
for key in dir():
    if isinstance(globals()[key], (int,float, float, str, bool, np.ndarray,list)):
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    else:
        pass


file = open("data/" + filename, 'wb')
pickle.dump(my_shelf, file)
file.close()
