# import packages
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
from scipy.optimize import minimize
from autograd import grad
from autograd import numpy as anp
import base64
from io import BytesIO


        
# Define the kernel function with derivatives if noise is added
def rbf_kernel(X1, X2, params):
    d = cdist(X1, X2, 'sqeuclidean')
    K = (params['varSigma']**2) * anp.exp(-d / (2 * params['lengthscale']**2))
    return K


def lin_kernel(x1, x2, params):
    K = params["varSigma"]**2*x1.dot(x2.T)# var * x1 * x2 (5,1) * (1,5) -> (5,5) 
    return K


def white_kernel(x1, x2, params):
    if np.array_equal(x1, x2):
        K = params["varSigma"]**2 * anp.eye(x1.shape[0])
    else:
        K = anp.zeros((x1.shape[0], x2.shape[0]))
    return K


def periodic_kernel(x1, x2, params):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return params["varSigma"]**2*anp.exp(-(2*anp.sin((anp.pi/params["period"])*d)**2)/params["lengthscale"]**2)


# Kernel dictionary
kernel_dict = {
    'RBF': rbf_kernel,
    'Linear': lin_kernel,
    'White': white_kernel,
    'Periodic': periodic_kernel
}
