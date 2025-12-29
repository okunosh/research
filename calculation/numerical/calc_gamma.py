import numpy as np
from calc_critical_alpha import calc_critical_alpha

Rd = 192 #m^2 s^-2 K^-1
cp = 844 #J K^-1 kg^-1
KAPPA = Rd/cp
g = 3.72
p_bottom = 6 #hPa

def theta(T,p):
    return T*(p_bottom/p)**KAPPA

def calc_dtheta_dz(theta_bottom, theta_top, z_top):
    delta_z = z_top #m

    return (theta_top - theta_bottom) / delta_z

def calc_z(T, p):
    return Rd*T/g * np.log(p_bottom/p)
p_top = 1 #hPa

#set numbers from the figure---------------
T_top = 180
T_bottom = 190
#-------------------------------------

theta_bottom = theta(T_bottom, p_bottom)
theta_top = theta(T_top, p_top)

T = (T_bottom + T_top)/2
z_top = 20000#calc_z(T, p_top) 

gamma = calc_dtheta_dz(theta_bottom, theta_top, z_top)
alpha = calc_critical_alpha(gamma)

print("z_top: {:.3f} [m]".format(z_top))
print("theta_top [K]: {:.3f}".format(theta_top))
print("theta_bottom [K]: {:.3f}".format(theta_bottom))
print("gamma [K/m]: {:e}".format(gamma))
print("Critical slope angle (degree) is {:3f}".format(alpha))
