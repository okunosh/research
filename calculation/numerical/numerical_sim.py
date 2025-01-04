import numpy as np
import os
import sys
import xarray as xr
from datetime import datetime
from simulation_def import Simulation

def show_params(params):
    for key, value in params.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    # Parameters
    num = 260
    alpha_deg = 0.46
    Theta = 5.
    K = 100.

    dt = 1.e-1
    #path
    output_path ="output/results"

    
    #Earth
    """
    params = {
        "num": num,
        "alpha_deg": alpha_deg,
        "Theta": Theta,
        "K": K,
        "g": 9.81,
        "omega": 7.28e-5,  # 2*pi /86400
        "theta_0": 288.,
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
        "g": 3.72,
        "omega": 7.08e-5,  # 2*pi /88750
        "theta_0": 210.,
        "gamma": 4.5e-3,
        "dt": dt,
        "output_path": output_path
    }
    
    #show_params(params)
    #calculate
    sim = Simulation(params["g"],
                     params["alpha_deg"],
                     params["Theta"],
                     params["theta_0"],
                     params["gamma"],
                     params["omega"],
                     params["K"],
                     params["num"],
                     params["dt"])

    sim.run_simulation(params["output_path"], params['dt'])
