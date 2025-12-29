import numpy as np
import matplotlib.pyplot as plt

g = 3.72
theta_0 = 213.19#212.30#230#210
deg2rad = np.pi / 180
T = 88750#24 * 3600
omega = 2*np.pi / T

def calc_critical_alpha(gamma):
    N = np.sqrt(gamma * g / theta_0)
    
    alpha = np.arcsin(omega / N) / deg2rad
    return alpha

    #alpha = np.linspace(0, 90, 1000)
    #N_alpha = N * np.sin(alpha*deg2rad)

if __name__=="__main__":

    gamma = 3.42e-3
    alpha = calc_critical_alpha(gamma)
    print("gamma [K/m] = {}".format(gamma))
    print("Critical slope angle (degree) is {:3f}".format(alpha))
