import numpy as np
import os
import sys
import xarray as xr
from datetime import datetime
import warnings
from simulation_def import Simulation
from condition_manager import write_conditions_to_file, handle_error, setup_signal_handler, log_error


def show_params(params):
    for key, value in params.items():
        print(f"{key}: {value}")

#warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

    
if __name__ == "__main__":
    # Parameters
    num = 260#10
    alpha_deg = 1.00 #0.48/2
    Theta = 100.
    K = None #to decide l_+ (height)
    K_file = "K/K_261.nc"
    surf_temp = "surface_forcing/test_surface_forcing.nc"

    dt = 1.e-1
    #path
    output_path = "output/results_v4_test" #"output/results"
    
    #Earth
    """
    params = {
        "num": num,
        "alpha_deg": alpha_deg,
        "Theta": Theta,
        "K": K,
        "K_file":K_file,
        "g": 9.81,
        "omega": 7.28e-5,  # 2*pi /86400
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
        "theta_0": 210.,
        "surface_temp": surf_temp,
        "gamma": 1.e-4,
        "dt": dt,
        "output_path": output_path
    }

    #show_params(params)
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

"""
    try:
        sim.run_simulation(params["output_path"], params['dt'])
        write_conditions_to_file('../calculation_conditions.csv', params, 1)

    except Exception as e:
        #print("inturrupted!")
        log_error(str(e))
        write_conditions_to_file("../calculation_conditions.csv", params, 0)
"""
