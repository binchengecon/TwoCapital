"""
file for post HJB with k and y
"""
import os
import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
from numba import njit
from supportfunctions import finiteDiff
import SolveLinSys

def false_transient_one_iteration_cpp(stateSpace, A, B1, B2, C1, C2, D, v0, ε):
    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), B2.reshape(-1, 1, order='F')])
    C = np.hstack([C1.reshape(-1, 1, order='F'), C2.reshape(-1, 1, order='F')])
    D = D.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0.reshape(-1, 1, order='F'), ε, -10)
    return out[2].reshape(v0.shape, order = "F")

def _hjb_iteration(
        v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar, delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction):

    dvdk  = finiteDiff(v0, 0, 1, dk)
    dvdkk = finiteDiff(v0, 0, 2, dk)
    dvdy  = finiteDiff(v0, 1, 1, dy)
    dvdyy = finiteDiff(v0, 1, 2, dy)

    temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat))) ** theta
    mc = 1. / temp

    i_new = - (mc / dvdk - 1) / kappa

    G = dvdy  - 1. / delta * d_Delta
    F = dvdyy - 1. / delta * dd_Delta

    temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
    a = temp / (alpha * lambda_bar * np.exp(k_mat)) ** 2
    b = - 2 * temp / (alpha * lambda_bar * np.exp(k_mat))\
        + (F - G**2/xi_b) * sigma_y ** 2
    c = temp + G * np.sum(pi_c * theta_ell, axis=0)

    # Method 1 : Solve second order equation
    if vartheta_bar != 0 and theta == 3:
        temp = b ** 2 - 4 * a * c
        temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2
    elif vartheta_bar != 0 and theta == 2:
        temp =  mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
        a = - mc * temp / (alpha * lambda_bar * np.exp(k_mat)) + F * sigma_y**2
        b = mc * temp + G * np.sum(pi_c * theta_ell, axis=0)
        e_new = - b / a
    else:
        e_new = c / (-b)

#     # Method 2 : Fix a and solve
#     e_new = (a * e**2 + c) / (-b)

    e_new = e_new * (e_new > 0) + 1e-8 * (e_new <= 0)
    
    i = i_new * fraction + i * (1-fraction)
    e = e_new * fraction + e * (1-fraction)

    log_pi_c_ratio = - G * e * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = pi_c / np.sum(pi_c, axis=0)
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

    A = np.ones_like(y_mat) * (- delta)
    B_k = mu_k + i - kappa / 2. * i ** 2 - sigma_k ** 2 / 2.
    B_y = np.sum(pi_c * theta_ell, axis=0) * e
    C_kk = sigma_k ** 2 / 2 * np.ones_like(y_mat)
    C_yy = .5 * sigma_y **2 * e**2

    D = np.log(1. / mc)\
        + k_mat - 1./ delta * (d_Delta * np.sum(pi_c * theta_ell, axis=0) * e + .5 * dd_Delta * sigma_y ** 2 * e ** 2)\
        + xi_a * entropy - C_yy * G**2 / xi_b

    h = - G * e * sigma_y / xi_b

    return pi_c, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h


def hjb_post_damage_post_tech(k_grid, y_grid, model_args=(), v0=None, epsilon=1., fraction=.1,
                              tol=1e-8, max_iter=10_000, print_iteration=True):

    delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3, y_bar, \
    theta, lambda_bar, vartheta_bar = model_args
    
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    
    a_i = kappa * (1. / delta)
    b_i = - (1. + alpha * kappa) * (1. / delta)
    c_i = alpha * (1. / delta) - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = 1. / delta * k_mat -  y_mat ** 2

    d_Delta  = gamma_1 + gamma_2 * y_mat + gamma_3 * (y_mat > y_bar) * (y_mat - y_bar)
    dd_Delta = gamma_2 + gamma_3 * (y_mat > y_bar)

    pi_c_o = np.array([temp * np.ones_like(y_mat) for temp in pi_c_o])
    theta_ell = np.array([temp * np.ones_like(y_mat) for temp in theta_ell])
    pi_c = pi_c_o.copy()

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        pi_c, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h = \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar,
                           delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction)

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, epsilon)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/epsilon))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {'v': v,
            "k": k_grid,
            'y': y_grid,
           'e': e,
           'i': i,
           'pi_c': pi_c,
           'h': h}

    return res


def hjb_post_damage_pre_tech(k_grid, y_grid, model_args=(), v0=None, ϵ=1., fraction=.1,
                              tol=1e-8, max_iter=10_000, print_iteration=True):

    δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g, I_g, v_g, γ_1, γ_2, γ_3, τ, theta, lambda_bar, vartheta_bar = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    
    a_i = κ * (1. / δ)
    b_i = - (1. + α * κ) * (1. / δ)
    c_i = α * (1. / δ) - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)
    
    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = 1. / δ * k_mat -  y_mat ** 2

    d_Λ = γ_1 + γ_2 * y_mat + γ_3 * (y_mat > τ) * (y_mat - τ)
    dd_Λ = γ_2 + γ_3 * (y_mat > τ)

    πc_o = np.array([temp * np.ones_like(y_mat) for temp in πc_o])
    θ = np.array([temp * np.ones_like(y_mat) for temp in θ])
    πc = πc_o.copy()

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        πc, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h = \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Λ, dd_Λ, theta, lambda_bar, vartheta_bar,
                           δ, α, κ, μ_k, σ_k, πc_o, πc, θ, σ_y, ξ_a, ξ_b, i, e, fraction)
        
#         # Method 1:
#         D -= ξ_g * I_g * (np.exp(- v_g / ξ_g) - np.exp(- v0 / ξ_g)) / (np.exp(- v0 / ξ_g))
        
        # Method 2:
        g_tech = np.exp(1. / ξ_g * (v0 - v_g))
        A -= I_g * g_tech
        D += I_g * g_tech * v_g + ξ_g * I_g * (1 - g_tech + g_tech * np.log(g_tech))

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, ε)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    g_tech = np.exp(1. / ξ_g * (v - v_g))

    res = {'v': v,
           'e': e,
           'i': i,
           'g_tech': g_tech,
           'πc': πc,
           'h': h,
           'k': k_grid,
           'y': y_grid
           }

    return res

