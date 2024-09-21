import numpy as np
import matplotlib.pyplot as plt

def slopewind(theta0, g, K, gamma, alpha_deg, theta_s):
    lamda = g / theta0
    deg2rad = np.pi / 180
    alpha_rad = alpha_deg * deg2rad
    
    D =  (lamda * gamma * np.sin(alpha_rad)**2 / 4 / K**2 )**(-0.25)
    print('D = '+str(D) + 'm')
    z = np.linspace(0, 2000, 1000)
    zp = z / D
    u = theta_s * (lamda / gamma)**0.5 * np.exp(-zp) * np.sin(zp) #theta_s = theta0としている
    theta = theta_s * np.exp(-zp) * np.cos(zp)
    
    return D, z, u, theta
