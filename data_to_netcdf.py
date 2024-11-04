import xarray as xr
import numpy as np
import os
from datetime import datetime

class DatasetToNetcdf:
    def __init__(self, ds):
        self.ds = ds
        self.now = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.num = self.get_num()
        self.K = self.format_value(ds.K.values[0])
        self.alpha = self.format_value(ds.alpha.values[0])
        self.gamma = self.format_gamma(ds.gamma.values[0])
        self.time = self.padded_time(ds.time.values[0])
        self.theta_0 = self.format_value(ds.theta_0.values[0])
        self.Theta = self.format_value(ds.Theta.values[0])
        self.t = ds.time.values[0]
        self.flow_regime = ds.flow_regime
        self.planet = ds.planet

    def make_dataset(data):
        data_vars = {
            "u_bar": (["altitude", "time"], data["u_bar"], {"unit": "m/s", "description": "along slope velocity"}),
            "theta_bar": (["altitude", "time"], data["theta_bar"], {"unit": "kelvin", "description": "mean potential temperature anomaly"}),
            "K": (["time"], np.array([data["K"]]), {"unit": "m^2 s^-1", "description": "diffusion coefficient"}),
            "alpha": (["time"], np.array([data["alpha"]]), {"unit": "degree", "description": "slope angle"}),
            "theta_0": (["time"], np.array([data["theta_0"]]), {"description": ""}),
            "Theta": (["time"], np.array([data["Theta"]]), {"description": ""}),
            "gamma": (["time"], np.array([data["gamma"]]), {"description": "vertical gradient of potential temperature"}),
            "N": (["time"], np.array([data["N"]]), {"description": "Bulant-Visala"}),
            "N_alpha": (["time"], np.array([data["N_alpha"]]), {"description": "Bulant-Visala along slope"}),
            "omega": (["time"], np.array([data["omega"]]), {"description": "angular frequency"}),
            "omega_plus": (["time"], np.array([data["omega_plus"]]), {"description": ""}),
            "omega_minus": (["time"], np.array([data["omega_minus"]]), {"description": ""}),
            "l_plus": (["time"], np.array([data["l_plus"]]), {"description": ""}),
            "l_minus": (["time"], np.array([data["l_minus"]]), {"description": ""}),
        }
        coords = {
            "altitude": ("altitude", data["altitude"], {"unit": "meters"}),
            "time": ("time", data["time"], {"unit": "seconds"})
        }

        attrs = {
            "institution": "KSU/sci",
            "planet": data["planet"],
            "history": f"Created on {datetime.now().isoformat()}",
            "reference": "Zardi et al. (2014), DOI:10.1002/qj.2485"
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return ds

    def get_num(self):
        return str(self.ds.altitude.shape[0])

    def format_value(self, value):
        return str(value).replace('.', '-')

    def format_gamma(self, gamma_value):
        gamma = f"{gamma_value:.2e}"
        gamma_base, gamma_exp = gamma.split("e")
        gamma_show = gamma_base.replace('.', '-')
        return gamma_show

    def padded_time(self, time_value):
        return f"{int(time_value):05}"

    def make_new_dir_path(self, output_path):
        new_dir_path = f"{output_path}/{self.planet}/{self.now}_{self.flow_regime}_num{self.num}_alp{self.alpha}_gamma{self.gamma}"
        return new_dir_path

    def make_new_dir(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok=True)
        
    def save_to_netcdf(self, new_dir_path): #
        kind = "A" #analytical change to E when numerical
        output_file = f"{new_dir_path}/{kind}_{self.now}_{self.flow_regime}_t{self.time}.nc"
        self.ds.to_netcdf(output_file)
            
                    
