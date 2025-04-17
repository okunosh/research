import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

#parameters-----------
DAY_SEC= 24*3600
time_res = 0.1
space_res = 11#260+1
#----------------------

#generate diurnal K distribution. write NetCDF.
def K_const(val, time_res, space_res, tarr):
    K_val = val
    len_tarr = len(tarr)
    K_arr = np.ones((len_tarr, space_res, 1)) * K_val

    return K_arr

def K_condition2(t_arr, space_res):#6-18: K=3, 18- : K=100 
    dawn = 6*36000
    dusk = 18*36000
    arr = np.ones_like(t_arr)*3
    arr[dawn:dusk] = 100

    K_arr = np.tile(arr, (space_res, 1))
    return K_arr

def K_condition3(space_res):
    ds = xr.open_dataset('TestGroundTheta0.nc')
    vals = ds.theta_0.values
    vals[vals<3] = 3 #Not 0 at least 3
    Ks = np.tile(vals, (space_res, 1))
    return Ks

def K_func4(t):
        if 6*3600 <=t<= 18*3600:
            #return 3 + 97*np.cos( 2*np.pi * (t-12*3600)/24/3600)
            return 3 + 47*np.cos( 2*np.pi * (t-12*3600)/24/3600)
        else:
            return 3
        
def K_condition4(t_arr, space_res):
    Ks = np.ones_like(t_arr)*np.nan
    for i, n in enumerate(t_arr):
        Ks[i] = K_func4(n)
    K_arr = np.tile(Ks, (space_res, 1))
    return K_arr

def K_standard(t_arr, space_res, amplitude):#sin distributuion at all altitude
    Ksts = amplitude * np.cos(2*np.pi * ((t_arr - 1*3600) / DAY_SEC)) * (-1)
    #Ksts = amplitude * np.sin(2*np.pi * (t_arr / DAY_SEC)) #sample
    K_arr = np.tile(Ksts, (space_res, 1))
    return K_arr

#write to netcdf------------------------------------------------------
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
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return ds

#plot distribution----------------------------------------------------
def plotK(ds):
     K = ds.K.values
     t = ds.time.values
     n = ds.altitude.values
     T, N = np.meshgrid(t,n)
     plt.pcolormesh(T, N, K, cmap='bwr')
     pp = plt.colorbar(orientation='vertical')
       
     #plt.xticks(t_arr)
     plt.xlabel('t')
     plt.ylabel('n')
     plt.show()
    
def saveToNetcdf(ds, path, confirm=False):
    if confirm==True:
        ds.to_netcdf(path)
    else:
        print('If you want to save, you need -> confirm==True')
        plotK(ds)
       
        
if __name__ == '__main__':
    save_name = "K/testdata/test_num10_2" #output name
    nc_path = save_name+".nc"
    png_path = save_name+".png"

    try:
        n_arr = np.arange(0, space_res)    
        t_arr = np.arange(0, DAY_SEC, time_res)
        len_tarr = len(t_arr)
        #K_arr = K_const(100, time_res, space_res, t_arr)
        K_arr = K_condition2(t_arr, space_res)
        #K_arr = K_condition3(space_res)
        #K_arr = K_condition4(t_arr, space_res)
        #K_arr = K_standard(t_arr, space_res, 40)
        
        K = K_arr.reshape(space_res, len_tarr)
        
        ds = makeDataset(space_res, time_res, t_arr, n_arr, K)
        saveToNetcdf(ds, nc_path, confirm=True)      

        ds.K.plot()
        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig(png_path, dpi=300)
    except Exception as e:
        print(e)
        
