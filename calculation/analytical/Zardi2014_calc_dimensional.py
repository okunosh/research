import numpy as np
import os
import sys
import xarray as xr
from importlib import reload
from datetime import datetime
import Zardi2014_def
reload(Zardi2014_def)
from Zardi2014_def import WaveResolutions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_to_netcdf import DatasetToNetcdf
from ql_plot import process_netcdf_directory

class AnalyticalDatasetToNetcdf(DatasetToNetcdf):
    @staticmethod
    def make_dataset(wave):
        data = {
            "u_bar": wave.resolutions()["u_bar"],
            "theta_bar": wave.resolutions()["theta_bar"],
            "K": wave.K,
            "alpha": wave.alpha_deg,
            "theta_0": wave.theta_0,
            "Theta": wave.Theta,
            "gamma": wave.gamma,
            "N": wave.N,
            "N_alpha": wave.N_alpha,
            "omega": wave.omega,
            "omega_plus": wave.omega_plus,
            "omega_minus": wave.omega_minus,
            "l_plus": wave.l_plus,
            "l_minus": wave.l_minus,
            "altitude": wave.resolutions()["altitude"],
            "time": np.array([wave.resolutions()["t"]]),
            "planet": wave.planet,
            "psi": wave.psi,
            "regime": wave.resolutions()["regime"]
        }
        ds = DatasetToNetcdf.make_dataset(data)
        ds = ds.assign(psi=(["time"], np.array([wave.psi]), {"description": "initial phase"}))
        ds = ds.assign_attrs(title="analytical solution", flow_regime=wave.resolutions()["regime"])
        #change attributes order
        existing_attrs = ds.attrs
        ordered_keys = ["title", "flow_regime", "planet", "history", "reference"]
        ordered_attrs = {key: existing_attrs[key] for key in ordered_keys if key in existing_attrs}
        ds.attrs = ordered_attrs
        return ds

def calculation(g,  alpha_deg, Theta, theta_0, gamma, omega, K, omega_t_value, num, psi=0):
    wave = WaveResolutions(g, alpha_deg, Theta, theta_0, gamma, omega, K, omega_t_value, num, psi)
    return wave



if __name__ == "__main__":
        #Parameters
        num = 260
        alpha_deg = 0.41
        Theta = 5.
        K = 3.
        
        #Earth parameters----------
        g = 9.81
        omega = 7.28e-5 #2*pi /86400
        theta_0 = 288.
        gamma = 3.e-3
        
        #Mars parameters-----------
        """
        #g = 3.72 #Mars
        #omega = 7.08e-5 #2*pi /88775
        #theta_0 = 210
        #gamma = 4.5e-3
        """

        #time parameters------------
        #omega_t_values = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 2*np.pi]

        day = 4
        day2sec = 24 * 3600
        t_fin = day * day2sec
        omega_t_values = omega * np.arange(0, t_fin+1, 3600)
        psi = 0.

        #path
        output_path ="output/results"

        #calculate & save to netcdf
        for j in range(len(omega_t_values)):
            wave = calculation(g, alpha_deg, Theta, theta_0, gamma, omega, K, omega_t_values[j], num, psi)
            ds = AnalyticalDatasetToNetcdf.make_dataset(wave)
            data = AnalyticalDatasetToNetcdf(ds)
            if j==0:
                new_dir_path = data.make_new_dir_path(output_path)
                print(new_dir_path)
                data.make_new_dir(new_dir_path)
            data.save_to_netcdf(new_dir_path)
        #add Quick Look Plot and save png file
        process_netcdf_directory(new_dir_path)
