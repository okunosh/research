import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

#parameters-----------
DAY_SEC= 24*3600
time_res = 0.1
space_res = 260+1
#----------------------

#generate diurnal K distribution. write NetCDF.
def load_data(file): #surface forcing file
    ds = xr.open_dataset(file)
    t = ds.time.values
    theta = ds.theta_0.values

    return t, theta

def make_K_1dim(theta_fit, K_min, K_max):
    # 拡散係数 K(t)：温位偏差に同期（値域 3〜100）
    K_mean = (K_max + K_min) / 2      # = 51.5
    K_amp = (K_max - K_min) / 2       # = 48.5
    K = K_mean + K_amp * (theta_fit / np.max(np.abs(theta_fit)))  # 正規化
    return K

def make_K_2dim(t, space_num, K): #num is the number of grid points
    #Ks = np.ones_like(t)*np.nan
    K_arr = np.tile(K, (space_num, 1))
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
        plot(ds)

def make_K_SF_fig(SF_t_val, SF_val, K_t_val, K_val, save_name, confirm):
    
    fontsize_title = 18
    fontsize_label = 14
    fontsize_tick = 12
    linewidth = 2
    fontsize_legend=12

    color_SF = '#A00000'
    color_K = "black"

    linewidth_SF = 2.5
    linewidth_K = 1.8
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(SF_t_val, SF_val, color=color_SF, linestyle=(0, (3, 1, 1, 1)), linewidth=2.5, label=r"$\overline{\theta}$(0,t)")
    ax.set_ylabel(r"temperature anomaly [K]", color=color_SF)

    ax2 = ax.twinx()
    ax2.plot(K_t_val, K_val, color=color_K, linewidth=1.8, alpha=0.6, label="K")
    ax2.set_ylabel("K [m$^2$/s]", color=color_K, fontsize=fontsize_label)

    ax.grid()
    ax.set_xlabel('t [hour]')
    ax.set_xticks(np.arange(0, 86401, 3600*5))
    ax.set_xticklabels(np.arange(0, 25, 5))
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, frameon=True, fontsize=fontsize_legend)
    ax.set_title(r"Diurnal distribution of $\overline{\theta}$(0,t) and K")
    if confirm==False:
        fig.savefig(save_name+".png" ,bbox_inches="tight", pad_inches=0.05, dpi=300)
    else:
        plt.show()
        
if __name__ == '__main__':
    #input
    surface_forcing_file = "surface_forcing_parametric/lat-30.0_Ls90.0/210.5_40.0_0.0.nc" #you need nz!!!
    K_min, K_max = 3, 100
    
    #output
    save_name = "K_parametric/lat-30.0_Ls90.0/210.5_40.0_0.0" #output name you need n!!!
    nc_path = save_name+".nc"
    png_path = save_name+".png"

    try:
        n_arr = np.arange(0, space_res)    
        #t_arr = np.arange(0, DAY_SEC, time_res)
        #len_tarr = len(t_arr)

        t, theta = load_data(surface_forcing_file)
        K_1dim = make_K_1dim(theta, K_min, K_max)
        K_arr = make_K_2dim(t,space_res, K_1dim) 
        
        K = K_arr.reshape(space_res, len(t))
        
        ds = makeDataset(space_res, time_res, t, n_arr, K)
        saveToNetcdf(ds, nc_path, confirm=True)
        """
        print("t.shape", t.shape)
        print("theta.shape", theta.shape)
        print("K_arr.shape", K_arr.shape)
        print("K_1dim.shape", K_1dim.shape)
        """
        save_name_K_SF = save_name + "_K_SF.png"
        make_K_SF_fig(t, theta, t, K_1dim, save_name_K_SF, confirm=False)
        #plotK(surface_forcing_file)
        """
        ds.K.plot()
        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig(png_path, dpi=300)
        """
    except Exception as e:
        print(e)
        
