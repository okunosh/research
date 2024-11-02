import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from importlib import reload
from datetime import datetime
import Zardi2014_def
reload(Zardi2014_def)
from Zardi2014_def import WaveResolutions


def calculation(g, alpha_deg, Theta, theta_0, gamma, omega, K, omega_t_value, num, psi=0):
    wave = WaveResolutions(g, alpha_deg, Theta, theta_0, gamma, omega, K, omega_t_value, num, psi)
    return wave

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
            psi = (["time"], np.array([wave.psi]), {"description":"initial phase"})                                   
            ),
        coords = dict(
            altitude = ("altitude", wave.resolutions()["altitude"], {"unit":"meters"}),
            time = ("time", np.array([wave.resolutions()["t"]]), {"unit":"seconds"})
            ),
        attrs = dict(title="analytical solution",
            institution="KSU/sci",
            planet=planet,
            flow_regime=wave.resolutions()["regime"],
            history=f"Created on {datetime.now().isoformat()}",
            reference="Zardi et al. (2014), DOI:10.1002/qj.2485"
            ),
      )
    return ds

def padded_time(time):
    str_time = str(time).replace('.', '-')
    if len(str_time) <= 6:
        padded_time = f"{time:07}"
        return padded_time
    else:
        return str_time

def make_new_dir(ds, output_path):
    time = datetime.now().strftime('%Y%m%dT%H%M%S')
    num = str(ds.altitude.shape[0])
    K = str(ds.K.values[0]).replace('.', '-') 
    alpha = str(ds.alpha.values[0]).replace('.', '-')
    gamma = f"{ds.gamma.values[0]:.2e}"
    gamma_base, gamma_exp = gamma.split("e")
    gamma = gamma_base.replace('.', '-') 
    new_dir_path = f"output_path/{time}_{ds.flow_regime}}_num{num}_K{K}_alp{alpha}_gamma{gamma}"
    os.makedirs(new_dir_path, exist_ok=True)
    
def save_to_netcdf(ds, output_path): #output/results/
    kind = "A"#"analytical"
    now = datetime.now().strftime('%Y%m%dT%H%M%S')
    time = padded_time(ds.time.values[0])
    num = str(ds.altitude.shape[0])
    K = str(ds.K.values[0]).replace('.', '-') 
    alpha = str(ds.alpha.values[0]).replace('.', '-')
    gamma = f"{ds.gamma.values[0]:.2e}"
    gamma_base, gamma_exp = gamma.split("e")
    gamma = gamma_base.replace('.', '-') 
    theta_0 = str(ds.theta_0.values[0]).replace('.', '-')
    Theta = str(ds.Theta.values[0]).replace('.', '-')
    
    output_file = f"{output_path}/{ds.planet}/{kind}_{now}_t{time}_{ds.flow_regime}_num{num}_K{K}_{alpha}deg_{gamma}.nc"

    ds.to_netcdf(output_file)
        

if __name__ == "__main__":
        #Parameters
        num = 701
        alpha_deg = 30.
        Theta = 5.
        K = 3.
        
        #Earth parameters----------
        planet = "Earth"
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
        
        print(planet)
        #time parameters------------
        #omega_t_values = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi, 2*np.pi]
        omega_t_values = omega * np.arange(0, 3600*24+1, 3600)
        psi = 0.
        
        for j in range(len(omega_t_values)):
            wave = calculation(g, alpha_deg, Theta, theta_0, gamma, omega, K, omega_t_values[j], num) 
            ds = make_Dataset(wave)
            save_to_netcdf(ds, "output/results")

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
