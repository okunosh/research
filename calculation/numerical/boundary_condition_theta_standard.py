import numpy as np
import xarray as xr
from datetime import datetime 
import matplotlib.pyplot as plt

def standard(Theta, omega, t):
    return Theta * np.sin(omega * t)

def surface_theta_0(Theta, omega,t):
    hour2sec = 3600
    T = (t/hour2sec)%24
    if 6 < T <= 13:
        return Theta + Theta * np.cos(( 2*np.pi *(T-13)/14))
    elif 13 < T:
        return Theta + Theta * np.cos(2*np.pi*(T-13)/22)
    else:
        return 0
"""
#see Martinez et al. (2017) Fig8
def GroundTempModel():
    T = [184,
         183,
         181,
         178.5,
         177,
         174,
         170,
         192,
         219,
         234,
         248,
         259,
         260,
         261,
         258.5,
         252,
         240,
         225,
         210,
         198,
         193,
         190,
         189,
         186]
    
    return T

def AnomalyFromBasic(T):
    t = np.arange(0, 24)
    num_coeff = 22

    T_mean = np.mean(T)
    T_obs = T-T_mean
    coeff = np.polyfit(t,T_obs,num_coeff)
    
    T_fitted = np.poly1d(coeff)(t)

    return t, coeff, T_fitted

def interp_fittedT(t, coeff, T_fitted):
    newt = np.arange(0, 24*3600, 0.1)/3600
    new_T_fitted = np.interp(newt, t, T_fitted)

    return newt, new_T_fitted

def comparePlot():
    T = GroundTempModel()
    t, coeff, T_fitted = AnomalyFromBasic(T)
    newt, new_T_fitted = interp_fittedT(t, coeff, T_fitted)

    T_mean = np.mean(T)
    T_anm = T - T_mean

    T_diff = T_anm - T_fitted

    fig, ax = plt.subplots(1,2, figsize=(12,10))
    ax[0].plot(t, T_anm, label='visual estimation')
    ax[0].plot(t, T_fitted, label='polynomal fitting: '+str(len(coeff)-1))
    ax[0].set_xlabel('hour')
    ax[0].set_ylabel('temperature anomaly [K]')
    ax[0].set_title(r'$\theta(t,0)$')
    ax[0].legend()

    ax[1].plot(t, T_diff)
    ax[1].set_xlabel('hour')
    ax[1].set_ylabel('temperature [K]')
    ax[1].set_title('difference between visual estimation and polynomal fitting')

    for row in range(ax.shape[0]):
        ax[row].grid()
    plt.show()
"""

def makeDataset(t, new_T_fitted):
    tar = np.arange(0, 24*3600, 0.1)
    data_vars = {
        "theta_0": ("time",
                    new_T_fitted,
                    {"unit":"kelvin",
                     "description":"surface temperature oscillation"
                     })
    }
    coords = {
        "time": ("time",
                 tar,
                 {"unit": "seconds"
                  }),
    }
    attrs = {
        "history":f"Created on {datetime.now().isoformat()}",
        #"reference": "Martinez et al. (20117), DOI: 10.1007/s11214-017-0360-x"
    }

    ds = xr.Dataset(data_vars = data_vars, coords=coords, attrs=attrs)
    return ds

def saveToNetcdf(ds, output, save=False):
    if save == True:
        ds.to_netcdf(output)
    else:
        print(ds)

def make_boundary(output, save):
    T = GroundTempModel()
    t, coeff, T_fitted = AnomalyFromBasic(T)
    newt, new_T_fitted = interp_fittedT(t, coeff,  T_fitted)
    ds = makeDataset(newt, new_T_fitted)
    saveToNetcdf(ds, output, save)
    return ds

def make_boundary2(output, save):
    t = np.arange(0, 24*3600, 0.1)
    omega = 2*np.pi /86400
    #surf_temp = 40 * np.cos(2*np.pi *(t-1) / 24)*(-1)
    surf_temp = 5 * np.sin(omega * t)
    ds = makeDataset(t, surf_temp)
    saveToNetcdf(ds, output, save)
    return ds
    
if __name__ == "__main__":

    output = "surface_forcing/NormalSurfaceForcing.nc"
    save = True
    ds = make_boundary2(output, save)

    ds.theta_0.plot()
    plt.show()

    """
    output = "TestGroundTheta0.nc"
    ds = make_boundary(output, False)
    

    t = ds.time.values /3600
    y = ds.theta_0.values

    plt.plot(t, y)
    plt.grid()
    plt.ylabel("temperature anomaly [K]")
    plt.xlabel("hour")
    plt.show()
    """
    """
    Theta = 20
    omega = 7.e-5

    day = 4
    hour = 24
    hour2sec = 3600
    t = np.arange(0, day * hour * hour2sec +0.1, 0.1)

    #-----boundary condition-----
    y = np.zeros(len(t))*np.nan
    for i in range(len(y)):
        y[i] = surface_theta_0(Theta, omega,t[i])
    #----------------------------

    #plot
    hour_ticks = np.linspace(t[0], t[-1], num=int(day*hour/12) +1)  
    tick_labels = [str(int(tick // 3600)) for tick in hour_ticks]

    plt.plot(t, y)
    plt.grid()
    plt.xticks(hour_ticks, tick_labels)
    plt.xlabel('hour')
    plt.ylabel(r'$\theta$')
    plt.title(r'$\theta$(0,t)')
    plt.show()
    """

    
