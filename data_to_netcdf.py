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
        self.flow_regime = ds.flow_regime
        self.planet = ds.planet
        self.time = self.padded_time(ds.time.values[0])
        self.theta_0 = self.format_value(ds.theta_0.values[0])
        self.Theta = self.format_value(ds.Theta.values[0])
        self.t = ds.time.values[0]
        #self.new_dir_path = self.make_new_dir_path(output_path)

    @staticmethod
    def make_Dataset(wave):
        ds = xr.Dataset(
            data_vars=dict(
            u_bar=(["altitude", "time"], wave.resolutions()["u_bar"], {"unit":"m/s", "description":"along slope velocity"}),
            theta_bar = (["altitude", "time"], wave.resolutions()["theta_bar"], {"unit": "kelvin", "description":"mean potential temperature anomaly"}),
            K = (["time"], np.array([wave.K]), {"unit": "m^2 s^-1", "description":"diffusion coefficient"}),
            alpha = (["time"], np.array([wave.alpha_deg]), {"unit": "degree", "description":"slope angle"}),
            theta_0 = (["time"], np.array([wave.theta_0]), {"description":""}),
            Theta = (["time"], np.array([wave.Theta]), {"description":""}),
            gamma = (["time"], np.array([wave.gamma]), {"description":"vertical gradient of potential temperature"}), 
            N = (["time"], np.array([wave.N]), {"description": "Bulant-Visala"}),
            N_alpha = (["time"], np.array([wave.N_alpha]),{"description":"Bulant-Visala along slope"}),
            omega = (["time"], np.array([wave.omega]), {"description":"angular frequency"}),
            omega_plus = (["time"], np.array([wave.omega_plus]), {"description":""}),
            omega_minus = (["time"], np.array([wave.omega_minus]), {"description":""}),
            l_plus = (["time"], np.array([wave.l_plus]), {"description":""}),
            l_minus = (["time"], np.array([wave.l_minus]), {"description":""}),
            #psi = (["time"], np.array([wave.psi]), {"description":"initial phase"})                                   
        ),
        coords = dict(
            altitude = ("altitude", wave.resolutions()["altitude"], {"unit":"meters"}),
            time = ("time", np.array([wave.resolutions()["t"]]), {"unit":"seconds"})
        ),
        attrs = dict(
            title="analytical solution",
            institution="KSU/sci",
            planet=wave.planet,
            flow_regime=wave.resolutions()["regime"],
            history=f"Created on {datetime.now().isoformat()}",
            reference="Zardi et al. (2014), DOI:10.1002/qj.2485"
        ),
    )
        return ds

    def get_num(self):
        return str(self.ds.altitude.shape[0])

    def format_value(self, value):
        return str(value).replace('.', '-')

    def format_gamma(self, gamma_value):
        gamma = f"{gamma_value:.2e}"
        gamma_base, gamma_exp = gamma.split("e")
        return gamma_base.replace(',', '-')

    def padded_time(self, time_value):
        return f"{int(time_value):05}"

    def make_new_dir_path(self, output_path):
        new_dir_path = f"{output_path}/{self.planet}/{self.now}_{self.flow_regime}_num{self.num}_alp{self.alpha}_gamma{self.gamma}"
        return new_dir_path

    def make_new_dir(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok=True)
        
    def save_to_netcdf(self, new_dir_path):
        kind = "A" #analytical change to E when numerical
        output_file = f"{new_dir_path}/{kind}_{self.now}_t{self.time}_{self.flow_regime}_num{self.num}_K{self.K}_{self.alpha}deg_{self.gamma}.nc"
        self.ds.to_netcdf(output_file)
            
                    
