import numpy as np
import os
import sys
import xarray as xr
from datetime import datetime
from simulation_def import Simulation

if __name__ == "__main__":
    #Parameters
    num = 700
    alpha_deg = 30.
    Theta = 5.
    K = 3.
    
    #Earth parameters----------
    #planet = "Earth"
    g = 9.81
    omega = 7.28e-5 #2*pi /86400
    theta_0 = 288.
    gamma = 3.e-3
    
    #Mars parameters-----------
    """
    planet = "Mars"
    #g = 3.72 #Mars
    #omega = 7.08e-5 #2*pi /88775
    #theta_0 = 210
    #gamma = 4.5e-3
    """

    #time parameters------------
    dt = 1.e-3

    #path
    output_path ="output/results"

    #calculate
    sim = Simulation(g, alpha_deg, Theta, theta_0, gamma, omega, K, num, dt)
    #input('stop')
    sim.run_simulation(output_path)