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
        #omega_t_values = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 2*np.pi]
        omega_t_values = omega * np.arange(0, 3600*24+1, 3600)
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
            print(ds)
            #input("stop")
            data.save_to_netcdf(new_dir_path)
            
# #Plot
# t_vals = np.arange(0, 2.1, 0.25) * np.pi / wave.omega
# t_vals = np.arange(0, 3600*24+1, 10800)/wave.omega 
# t_labels = [f"{val:.3g}" for val in t_vals]

# #plot
# fig, ax = plt.subplots(2,1, figsize=(15,12))
# for i, result in enumerate (results):
#     params = result['params']
#     label = f"alpha={params['alpha_deg']}, Theta={params['Theta']}"

#     ax[0].vlines(result['const_u']*i, -0.1, 2*np.pi*result['l_plus'], color='black', linestyle='--', linewidth=1)
#     ax[0].plot(result['const_u']*i +result['u_bar'], result['n'], color='darkblue')
#     ax[0].text(result['const_u']*i, 1.9*np.pi*result['l_plus'], 't={}'.format(t_labels[i]))

#     ax[1].vlines(params['Theta']*i, -0.5, 2*np.pi*result['l_plus'], color='black', linestyle='--', linewidth=1)
#     ax[1].plot(params['Theta']*i +result['theta_bar'],result['n'], color='red')
#     ax[1].text(params['Theta']*i, 1.9*np.pi*result['l_plus'], 't={}'.format(t_labels[i]))


# yticks_label = np.array(['0', '$\pi l_-$/2', '$\pi l_-$', '$3 \pi l_- / 2$', '$2\pi l_-$'] )
# yticks = [0, np.pi*result['l_minus'] / 2,  np.pi*result['l_minus'], 3/2 * np.pi*result['l_minus'], 2* np.pi*result['l_minus']]

# xticks_label = np.array(['-$\Theta N $/2$\gamma$', '0', '$\Theta N $/2$\gamma$'])
# xticks = [-params['Theta']/2 * result['N']/params['gamma'], 0, params['Theta']/2 * result['N']/params['gamma']]

# twinx_label = [r'-$\Theta$', '0', r'$\Theta$']
# twinx_ticks = [-params['Theta'], 0, params['Theta']]
             
# twiny_label = ['0', '$\pi l_+$/2', '$\pi l_+$', '$3 \pi l_+ / 2$', '$2\pi l_+$']
# twiny_ticks = [0, np.pi*result['l_plus'] / 2,  np.pi*result['l_plus'], 3/2 * np.pi*result['l_plus'], 2* np.pi*result['l_plus']]

# ax[0].set_title('u [m/s]')
# ax[0].set_xticks([0, result['const_u']])

# ax[1].set_title(r'$\theta$ [K]')
# ax[1].set_xticks([0, params['Theta']])

# for k in range(2):
#     #ax[k].set_yticks(yticks, yticks_label)
#     ax[k].set_ylabel('n [m]')
#     #ax[k].twinx().set_yticks(twiny_ticks, twiny_label)
#     #ax[k].twinx().minorticks_on()
#     #ax[k].minorticks_on()

# plt.show()
