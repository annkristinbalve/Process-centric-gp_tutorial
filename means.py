import numpy as np

def zero_mean(X):
    return np.zeros(X.shape[0])

def sine_mean(x):
    return np.sin(x).flatten()

def lin_mean(x):
    return x.flatten()

def step_mean(x):
    return np.array(list(map(lambda a: 1 if a > 0 else 0, x))).flatten()



mean_dict = {
    'Zero': zero_mean,
    'Sine': sine_mean,
    'Linear': lin_mean,
    'Step': step_mean
}
