import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime


def K_condition(time_res, space_res, tarr):
    K_val = 3
    len_tarr = len(tarr)
    K_arr = np.ones((len_tarr, space_res, 1)) * K_val

    return K_arr

def K_condition2(t_arr, space_res):
    dawn = 6*36000
    dusk = 18*36000
    arr = np.ones_like(t_arr)*3
    arr[dawn:dusk] = 30
    K_arr = np.tile(arr, (space_res, 1))
    return K_arr

def makeDataset(space_res, time_res, tarr, narr, Karr):
    tar = np.arange(0, 24*3600, time_res)
    data_vars = {
        "K": (["altitude", "time"],
                    Karr,
                    {"unit":"m^2/s",
                     "description":"spacial and time distribution of K"
                     })
    }
    coords = {
        "altitude":("altitude",
                    narr,
                    {"unit":"meters"
                     }),
        "time": ("time",
                 tarr,
                 {"unit": "seconds"
                  })
    }
    attrs = {
        "history":f"Created on {datetime.now().isoformat()}"
        """
        "reference":"
        """
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return ds

def saveToNetcdf(ds, path, confirm=False):
    if confirm==True:
        ds.to_netcdf(path)
    else:
        print('If you want to save, you need -> confirm==True')
        K = ds.K.values
        t = ds.time.values
        n = ds.altitude.values

        T, N = np.meshgrid(t,n)
        plt.pcolormesh(T, N, K, cmap='bwr')
        pp = plt.colorbar(orientation='vertical')
        
        #plt.xticks(t_arr/3600)
        plt.xlabel('t [hour]')
        #plt.ylabel('altitude [m]')
        plt.show()
        
if __name__ == '__main__':
    DAY_SEC= 24*3600
    time_res = 0.1
    space_res = 260
    path = "K/Day30_Night3.nc"

    n_arr = np.arange(0, space_res)    
    t_arr = np.arange(0, DAY_SEC, time_res)
    len_tarr = len(t_arr)
    
    #K_arr = K_condition(time_res, space_res, t_arr)
    K_arr = K_condition2(t_arr, space_res)
    K = K_arr.reshape(space_res, len_tarr)
    
    ds = makeDataset(space_res, time_res, t_arr, n_arr, K)
    saveToNetcdf(ds, path, confirm=False)

    
