# import packages
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autograd import grad
from autograd import numpy as anp

def GP_conditional(m, k, params, X, y, return_data_contribution = True):
    """
    m: mean function
    k: kernel function
    noise: scalar
    X: input data
    y: observations
    return: predictive mean and covariance function and negative log marginal likelihood
    """
    N = X.shape[0]
    K = k(X, X, params)
    L = np.linalg.cholesky(K + params["noise"]**2 * np.eye(N))
    A = lambda Xtest: np.linalg.solve(L, k(X, Xtest, params))   # k_star = kernel(X, Xtest), calculating v := l\k_star
    b = np.linalg.solve(L, y - m(X).flatten()) # vector 5 x 1 np.linalg.solve(L, y)
    mu_f = lambda Xtest: m(Xtest) + np.dot(A(Xtest).T, b)    # \alpha = np.linalg.solve(L, y) 
    k_f = lambda Xtest1, Xtest2, params=params: k(Xtest1, Xtest2, params) - np.dot(A(Xtest1).T, A(Xtest2))
    log_Z_new = (-0.5 * np.dot(b.T,b) - np.sum(np.log(np.diag(L))) - N / 2 * np.log(2 * np.pi))
    if return_data_contribution: 
        return mu_f, k_f, log_Z_new, lambda Xtest1, Xtest2: np.dot(A(Xtest1).T, A(Xtest2)), lambda Xtest: np.dot(A(Xtest).T, b)
    return mu_f, k_f, -log_Z_new



def draw_samples(m, k, n):
    """
    m: mean vector
    k: covariance matrix
    return: 
    """
    L = np.linalg.cholesky(k)
    z = np.random.normal(loc=0,scale=1,size=(len(m), n)) # before: len(x)
    y = np.dot(L, z) + m[:,None]
    return y


def GP_conditional_iterative(m, k, params, X, y, log_Z_old=0):
    """
    m: mean function
    k: kernel function
    params: dictionary containing parameters, including noise
    X: input data
    y: observations
    log_Z_old: previous log marginal likelihood
    N_old: previous number of data points
    return: predictive mean and covariance function, updated log marginal likelihood, updated N
    """
    N = X.shape[0]
    K = k(X, X, params)
    L = np.linalg.cholesky(K + params["noise"]**2 * np.eye(N))
    A = lambda Xtest: np.linalg.solve(L, k(X, Xtest, params))
    b = np.linalg.solve(L, y - m(X).flatten())
    mu_f = lambda Xtest: m(Xtest) + np.dot(A(Xtest).T, b)
    k_f = lambda Xtest1, Xtest2, params=params: k(Xtest1, Xtest2, params) - np.dot(A(Xtest1).T, A(Xtest2))
    
    log_Z_new = log_Z_old - 0.5 * np.dot(b.T, b) - np.sum(np.log(np.diag(L))) - N / 2 * np.log(2 * np.pi)
    
    return mu_f, k_f, log_Z_new

    
def GP_conditional_manually(m, k, X, y, params, return_gradients=True):
    N = X.shape[0]
    K, derivatives_theta = k(X, X, params, True)
    L = np.linalg.cholesky(K + params['noise']**2 * np.eye(X.shape[0]))
    A = lambda Xtest: np.linalg.solve(L, k(X, Xtest, params))
    b = np.linalg.solve(L, y - m(X).flatten())
    mu_f = lambda Xtest: m(Xtest) + np.dot(A(Xtest).T, b)
    k_f = lambda Xtest1, Xtest2: k(Xtest1, Xtest2, params) - np.dot(A(Xtest1).T, A(Xtest2))

    # Compute the log marginal likelihood
    log_Z = (-0.5 * np.dot(b.T, b) - np.sum(np.log(np.diag(L))) - N / 2 * np.log(2 * np.pi))

    # Compute the gradients manually
    if return_gradients:
        inv_L = np.linalg.solve(L, np.eye(L.shape[0]))
        K_inv = inv_L.T @ inv_L
        alpha = np.linalg.solve(L.T, b)
        alpha_outer = np.outer(alpha, alpha)
        grads = np.array([-(0.5 * np.trace((alpha_outer - K_inv) @ dk_theta)) for dk_theta in derivatives_theta])
        return mu_f, k_f, -log_Z, grads
    return mu_f, k_f, -log_Z


# Define the negative log marginal likelihood function
def neg_log_marginal_likelihood(flat_params, param_keys, X, y, m, k, return_components = False):
    params = {key: flat_params[i] for i, key in enumerate(param_keys)}
    N = X.shape[0]
    K = k(X, X, params)
    L = anp.linalg.cholesky(K + params['noise']**2 * anp.eye(N))
    b = anp.linalg.solve(L, y - m(X).flatten())
    neg_log_likelihood = -0.5 * anp.dot(b.T, b) - anp.sum(anp.log(anp.diag(L))) - N / 2 * anp.log(2 * anp.pi)
    if return_components:
        data_fit = -0.5 * anp.dot(b.T, b)
        complexity = -anp.sum(anp.log(anp.diag(L)))
        constant_term = -N / 2 * anp.log(2 * anp.pi)
        neg_log_likelihood = data_fit + complexity + constant_term
        return -neg_log_likelihood, -data_fit, -complexity, -constant_term
    return -neg_log_likelihood

    
# Define the gradient of the negative log marginal likelihood using autograd
neg_log_marginal_likelihood_grad = grad(neg_log_marginal_likelihood)

# Define the Gaussian Process inference function
def GP_conditional_optimised(m, k, X, y, params):
    param_keys = list(params.keys())
    flat_params = np.array([params[key] for key in param_keys])
    
    # Optimize the parameters using minimize and the Jacobian
    opt_result = minimize(
        neg_log_marginal_likelihood, 
        flat_params, 
        args=(param_keys, X, y, m, k), 
        method='L-BFGS-B', 
        jac=neg_log_marginal_likelihood_grad
    )
    optimized_params = {param_keys[i]: opt_result.x[i] for i in range(len(param_keys))}
    
    # Update params with optimized values
    params.update(optimized_params)
    
    # Recompute based on optimized params
    N = X.shape[0]
    K = k(X, X, params)
    L = np.linalg.cholesky(K + params['noise']**2 * np.eye(N))
    A = lambda Xtest: np.linalg.solve(L, k(X, Xtest, params))
    b = np.linalg.solve(L, y - m(X).flatten())
    mu_f = lambda Xtest: m(Xtest) + np.dot(A(Xtest).T, b)
    k_f = lambda Xtest1, Xtest2: k(Xtest1, Xtest2, params) - np.dot(A(Xtest1).T, A(Xtest2))

    # Compute the log marginal likelihood
    log_Z = opt_result.fun
    return mu_f, k_f, log_Z
    

def GP_marginal(m, k, X): # marginal operation 
    """
    m: mean function 
    k: kernel function 
    X: observations matrix (N x D) 
    return mean vector and covariance matrix 
    """    
    # Compute the mean vector using a vectorized approach
    m_vector = m(X)
    
    # Compute the covariance matrix using broadcasting and vectorized operations
    K_matrix = k(X, X)
    return m_vector, K_matrix