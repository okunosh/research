import numpy as np
import xarray as xr
#from ql_plot import filter_files_by_time
import os

g = 3.72 #
p00 = 5 #surface pressure on Mars [hPa]
cp = 1
R = 1
KAPPA = cp / R

def T_func(theta, p):
    return theta * (p00 / p)**-KAPPA

def ro_func(T, p):
    return p / R / T

def p_func(z, T):
    return p00 * np.exp(-g / R / T * z)


def calc_RiNum(directory, u_max, t_at_umax, z_at_umax, theta_bar_surf_at_umax, theta_bar_z_at_umax):
    #load data
    """
    filtered_files = filter_files_by_time()
    files_num = len(filtered_files) #time
    first_file_path = os.path.join(directory, filtered_files[0])
    ds0 = xr.open_dataset(first_file_path)
    """
    theta0 = 230#210 #ds.theta0.values
    gamma = 2.41e-3#4.02e-3  

    #surface at umax
    theta0_at_umax = theta0 + theta_bar_surf_at_umax
    
    #z at umax
    thetaz_at_umax = theta0 + gamma * z_at_umax + theta_bar_z_at_umax
    
    theta_mean = (thetaz_at_umax + theta0_at_umax) / 2

    dudz = u_max / z_at_umax
    
    Ri = g / theta_mean * (thetaz_at_umax - theta0_at_umax) / z_at_umax / dudz**2

    print("----------")
    print("g [m/s^2]: ", g)
    print("theta_surf at umax [K]: ", theta0_at_umax)
    print("theta_z at umax [K]:", thetaz_at_umax)
    print("theta mean between two altitudes [K]: ", theta_mean)
    print("z at umax [m]: ", z_at_umax)
    print("umax [m/s]: ", u_max)
    print("Richardson Number: ", Ri)
    print("----------")
    
    return Ri


def calc_RiNum_ro(directory, u_max_min, theta_max_min, theta_bar_surf_at_umax, theta_bar_z_at_umax):
    #load data
    """
    filtered_files = filter_files_by_time()
    files_num = len(filtered_files) #time
    first_file_path = os.path.join(directory, filtered_files[0])
    ds0 = xr.open_dataset(first_file_path)
    """
    theta0 = 230#210 #ds.theta0.values
    gamma = 2.41e-3#4.02e-3
    #env
    #T_env = T_func(theta0, p00)
    #ro_env = ro_func(T_env, p00)

    u_max = u_max_min["max_value"]
    t_at_umax = u_max_min["t_at_max"]
    z_at_umax = u_max_min["altitude_at_max"]

    #surface at umax
    theta0_at_umax = theta0 + theta_bar_surf_at_umax
    #T_surf_at_umax = T_func(theta0_at_umax, p00)
    #ro_surf_at_umax = ro_func(T_surf_at_umax, p00)

    #z at umax
    thetaz_at_umax = theta0 + gamma * z_at_umax + theta_bar_z_at_umax
    #T_z_at_umax = T_func(thetaz_at_umax, p_at_umax)###
    #p_at_umax = p_func(z_at_umax, T_z_at_umax)###
    #ro_at_umax = ro_func(T_z_at_umax, p_at_umax)

    #at surface
    #T0 = T_func(theta0+theta_max, p00)
    #ro0 = ro_func(T0, p00)

    theta_mean = (thetaz_at_umax + theta0_at_umax) / 2

    dudz = u_max / z_at_umax
    
    #Ri = - g / ro_env * (ro_at_umax - ro_surf_at_umax) / z_at_umax / dudz**2

    Ri = g / theta_mean * (thetaz_at_umax - theta0_at_umax) / z_at_umax / dudz**2

    print("----------")
    print("g [m/s^2]: ", g)
    print("theta_surf at umax [K]: ", theta0_at_umax)
    print("theta_z at umax [K]:", thetaz_at_umax)
    print("theta mean between two altitudes [K]: ", theta_mean)
    print("z at umax [m]: ", z_at_umax)
    print("umax [m/s]: ", u_max)
    print("Richardson Number: ", Ri)
    print("----------")
    
    return Ri

        

if __name__ == "__main__":

    """
    directory =
    u_max_min =
    theta_max_min =
    theta_bar_surf_at_umax =
    theta_bar_z_at_umax =
    
    Rinum = calc_RiNum(directory, u_max_min, theta_max_min, theta_bar_surf_at_umax, theta_bar_z_at_umax)
    """
