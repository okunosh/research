import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
from scipy.optimize import curve_fit


#parameters-----------
DAY_SEC= 24*3600
time_res = 0.1
space_res = 780+1# 260+1
#----------------------

#generate diurnal K distribution. write NetCDF.
def surface_T_Model():
    T = np.array([
    184, 183, 181, 178.5, 177, 174, 170, 192, 219, 234, 248, 259,
    260, 261, 258.5, 252, 240, 225,
    210, 195, 185, 180, 178, 184
    ])
    return T

def fourier_series(t, *a):
    result = a[0]
    for n in range(1, 6):
        result += a[2*n-1] * np.cos(2 * np.pi * n * t / 24) + a[2*n] * np.sin(2 * np.pi * n * t / 24)
    return result

def generate_K_1dim(T):
    hours = np.arange(24)

    # 3. フィッティング
    initial_guess = [np.mean(T)] + [0] * (2 * 5)
    popt, _ = curve_fit(fourier_series, hours, T, p0=initial_guess)
    
    # 4. 高解像度の時間配列（0.1秒刻み）
    t_sec = np.linspace(0, 86400, 864000, endpoint=False)
    t_hour = t_sec / 3600
    theta_fit = fourier_series(t_hour, *popt)
    
    # 5. 正規化してKを作成
    theta_min, theta_max = np.min(theta_fit), np.max(theta_fit)
    K = 3 + (theta_fit - theta_min) / (theta_max - theta_min) * (100 - 3)
    K_min = np.min(K)
    
    # 6. 夜間（0〜5時）：K_minで固定
    K[(t_hour >= 0) & (t_hour < 5)] = K_min
    
    # 7. 最大値の要素番号と時刻を取得
    idx_max = np.argmax(K)
    t_max = t_hour[idx_max]
    
    # 8. 置き換え対象：t_max 〜 22時 に変更
    t_target = t_hour[(t_hour >= t_max) & (t_hour <= 22)]
    idx_start = np.where(t_hour == t_target[0])[0][0]
    idx_end = np.where(t_hour == t_target[-1])[0][0]
    
    # 9. 三角関数（cos）で滑らかに減少：x=0→最大、x=1→最小（22時）
    x = (t_target - t_max) / (22 - t_max)
    K_max = K[idx_start]
    K_range = K_max - K_min
    K_replacement = K_min + K_range * 0.5 * (1 + np.cos(np.pi * x))

    # 10. 置き換え
    K[idx_start:idx_end+1] = K_replacement
    
    # 11. 22時以降をK_minで固定
    K[t_hour > 22] = K_min

    return K

def make_K_2dim(space_num, K): #num is the number of grid points
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
       
        
if __name__ == '__main__':
    #input

    #output
    rate = 100
    save_name = f"K/K_{space_res}_{rate}"#K/testdata/num10_test1" #output name
    nc_path = save_name+".nc"
    png_path = save_name+".png"

    try:
        n_arr = np.arange(0, space_res)
        t = np.linspace(0, 86400, 864000, endpoint=False)

        T = surface_T_Model()
        K = generate_K_1dim(T)*rate*0.01
        K = make_K_2dim(space_res, K)
        Ks = K.reshape(space_res, len(t))
        print(Ks.shape)
        
        ds = makeDataset(space_res, time_res, t, n_arr, K)
        saveToNetcdf(ds, nc_path, confirm=True)      

        ds.K.plot()
        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig(png_path, dpi=300)
    except Exception as e:
        print(e)
        
