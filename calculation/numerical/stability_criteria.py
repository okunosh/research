import numpy as np

#CONSTANT
DEG2RAD = np.pi/180
omega = 7.08e-5
g = 3.72

#condition
alpha_deg = 0.42
gamma = 4.5e-1
theta_0 = 210
Kmin, Kmax = 3, 100
dn = 1

def calc_stable_dt(dn, K):
    return 0.5 * dn / K

def calc_stable_dts(dn, Kmin, Kmax):
    dt_min = calc_stable_dt(dn, Kmax)
    dt_max = calc_stable_dt(dn, Kmin)
    return dt_min, dt_max

#the number of GP SPATIAL direction
def calc_the_number_of_GP(dn, K, omega_plus):
    n = np.pi / dn * np.sqrt(8*K/omega_plus)
    return n

def calc_N_alpha(gamma, Theta, alpha_deg):
    alpha = alpha_deg * DEG2RAD
    N = np.sqrt(gamma*g/theta_0)
    N_alpha = N * np.sin(alpha)
    return N_alpha

def calc_omega_plus(gamma, theta_0, alpha):
    N_alpha = calc_N_alpha(gamma, theta_0, alpha)
    return omega + N_alpha

def calc_l_plus(omega_plus, K):
    return np.sqrt(2* K / omega_plus)

#the number of grid points (SPATIAL direction) in each K
def calc_the_number_of_GPs(dn, alpha, gamma, theta_0, Kmin, Kmax):
    N_alpha = calc_N_alpha(gamma, theta_0, alpha)
    omega_plus = calc_omega_plus(gamma, theta_0, alpha)
    print("omega_plus is {}".format(omega_plus))

    n_min = calc_the_number_of_GP(dn, Kmin, omega_plus)
    n_max = calc_the_number_of_GP(dn, Kmax, omega_plus)
    return n_min, n_max

if __name__ == "__main__":
    dt_min, dt_max = calc_stable_dts(dn, Kmin, Kmax)
    n_min, n_max = calc_the_number_of_GPs(dn, alpha_deg, gamma, theta_0, Kmin, Kmax)
    omega_plus = calc_omega_plus(gamma, theta_0, alpha_deg)
    l_plus_min = calc_l_plus(omega_plus, Kmin)
    l_plus_max = calc_l_plus(omega_plus, Kmax)

    print("-----")
    print("If dn={}[m], for STABLE CALCLATION".format(dn))
    print("-----")
    print("(When K={}, dt should be less than {}[s])".format(Kmin, dt_max))
    print("When K={}, dt should be less than {}[s]".format(Kmax, dt_min))
    print("-----")
    print("When K={}, the number of GPs should be more than {}".format(Kmin, n_min))
    print("When K={}, the number of GPs should be more than {}".format(Kmax, n_max))
    print('-----')
    print("The top layer is 2*pi*l_plus")
    print("When K={}, altitude of top layer will be {}[m]".format(Kmin, 2*np.pi*l_plus_min))
    print("When K={}, altitude of top layer will be {}[m]".format(Kmax, 2*np.pi*l_plus_max))

    
          
