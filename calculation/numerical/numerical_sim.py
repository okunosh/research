import numpy as np
import os
import sys
import re
import xarray as xr
from datetime import datetime
import warnings
from simulation_def import Simulation
from condition_manager import write_conditions_to_file, handle_error, setup_signal_handler, log_error


def show_params(params):
    for key, value in params.items():
        print(f"{key}: {value}")

def theta0_from_forcing(path):
    ds = xr.open_dataset(path)
    theta0 = ds.attrs["theta0_surface_mean"]
    return float(theta0)

#warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

    
if __name__ == "__main__":
    # Parameters
    num = 260 #260*3#10
    alpha_deg = 2.25
    Theta = 100.
    K = None #to decide l_+ (height)
    #K_file ="K/K_261.nc" #"K/K_781_100.nc"   #"K/AllDay_3.nc"#"K/K_261_25.nc"
    #surf_temp = "surface_forcing/test_surface_forcing.nc"

    K_file = "K_parametric/lat+0.0_Ls90.0/210.5_40.0_0.0_NZ261.nc"
    surf_temp = "surface_forcing_parametric/lat+0.0_Ls90.0/210.5_40.0_0.0.nc"

    #"surface_forcing/NormalSurfaceForcing.nc"#"surface_forcing/test_surface_forcing.nc"

    dt = 1.e-1

    theta0 = theta0_from_forcing(surf_temp)
    Ls = re.search(r"(Ls\d+)", surf_temp).group(1)
    gamma_Mars_dic ={"Ls90": 4.020e-03,
                     "Ls180": 2.410e-03}

    
    """
    #Earth
    params = {
        "num": num,
        "alpha_deg": alpha_deg,
        "Theta": Theta,
        "K": K,
        "K_file":K_file,
        "g": 9.81,
        "omega": 2*np.pi/86400,# 7.28e-5,  2*pi /86400
        "theta_0": 288.,
        "surface_temp": surf_temp,
        "gamma": 3.e-3,
        "dt": dt,
        "output_path": output_path
    }
    """
    #Mars
    
    params = {
        "num": num,
        "alpha_deg": alpha_deg,
        "Theta": Theta,
        "K": K,
        "K_file":K_file,
        "g": 3.72,
        "omega":  2*np.pi /88750,
        "theta_0": theta0,
        "surface_temp": surf_temp,
        "gamma": gamma_Mars_dic[Ls],#2.410e-03,#4.020e-03,#2.410e-03,
        "dt": dt,
    }

    
    #path
    output_path = f"output/results_{Ls}_gamma_{params['gamma']}_theta0_{params['theta_0']}"

    params["output_path"] =  output_path
    
    show_params(params)
    #calculate
    sim = Simulation(params["g"],
                     params["alpha_deg"],
                     params["Theta"],
                     params["theta_0"],#no use
                     params["gamma"],
                     params["omega"],
                     params["K"],
                     params["num"],
                     params["dt"],
                     params["K_file"],
                     params["surface_temp"])

    write_conditions_to_file('../calculation_conditions.csv', params, None)
    setup_signal_handler(params)
    sim.run_simulation(params["output_path"], params['dt'])
