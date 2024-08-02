import numpy as np
np.random.seed(42)

def data_1d(N, noise):
    x = np.random.randn(N) 
    x = np.sort(x)
    y = f(x)
    y = y + noise * np.random.randn(N)  # Ensure the noise has the same shape as y
    x = x.reshape(-1, 1)
    return x, y

def data_1d_evenly_spaced(N, noise, x_range= 3):
    x = np.linspace(-x_range,x_range,N)
    x = np.sort(x)
    y = f(x)
    y = y + noise * np.random.randn(N)  # Ensure the noise has the same shape as y
    x = x.reshape(-1, 1)
    return x, y

def f(x):
    return np.sin(3 * x).flatten()