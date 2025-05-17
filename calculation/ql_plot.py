import os
import re
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import TwoSlopeNorm

class NetCDFProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.netcdf_files = self.extract_netcdf_files()

    def extract_netcdf_files(self):
        """extract all netcdf in the directory"""
        files = [f for f in os.listdir(self.directory) if f.endswith('.nc')]
        sorted_files = sorted(files, key=lambda x: int(x.split('_t')[1].split('.')[0]))
        return sorted_files

    def filter_files_by_time(self):
        """extract files in each 3 hours"""
        filtered_files = []
        for file in self.netcdf_files:
            match = re.search(r't(\d+)\.nc', file)
            if match:
                time_in_seconds = int(match.group(1))
                if time_in_seconds % 3600==0:#10800 == 0:  # 10800sec = 3 hours
                    filtered_files.append(file)
        return filtered_files

    def plot_variables(self):
        """plot u_bar & theta_bar"""
        filtered_files = self.filter_files_by_time()
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        for i, file in enumerate(filtered_files):
            file_path = os.path.join(self.directory, file)
            ds = xr.open_dataset(file_path)
            
            u_bar = ds.u_bar.values
            theta_bar = ds.theta_bar.values
            time = ds.time.values[0] / 3600
            l_plus = ds.l_plus.values[0]
            upper_alt = 2 * np.pi * l_plus
            altitude = ds.altitude.values
            
            const_u = ds.Theta.values[0] * ds.N.values[0] / ds.gamma.values[0] / 2
            const_theta = ds.Theta.values[0]
            
            axs[0].plot(u_bar + i * const_u, altitude, color='darkblue')
            axs[0].vlines(const_u*i, -0.1, upper_alt, color='black', linestyle='--', linewidth=1)
            axs[0].text(const_u*i, 2.0*np.pi*l_plus, '{}'.format(time))
            
            axs[1].plot(theta_bar + i * const_theta, altitude, color='red')
            axs[1].vlines(const_theta*i, -0.5, upper_alt, color='black', linestyle='--', linewidth=1)
            axs[1].text(const_theta*i, 2.0*np.pi*l_plus, '{}'.format(time))
        
        axs[0].set_title(r'$\overline{u}$ [m/s]')
        axs[1].set_title(r'$\overline{\theta}$ [K]')
        axs[0].set_xticks([0, const_u])
        axs[1].set_xticks([0, const_theta])
        axs[0].set_ylabel('Altitude [m]')
        axs[1].set_ylabel('Altitude [m]')
        
        plt.tight_layout()

        base_name = os.path.basename(self.directory.rstrip('/'))
        save_file_name = f"{self.directory}/{base_name}.png"

        #save_dir2 = ""
        #save_file_name2 = f"{}/{base_name}.png"
        plt.savefig(save_file_name, dpi=300)

    def load_data(self): #netcdf
        filtered_files = self.filter_files_by_time()
        files_num = len(filtered_files) #times
        
        first_file_path = os.path.join(self.directory, filtered_files[0])
        ds0 = xr.open_dataset(first_file_path)

        spatial_length = ds0.altitude.values.shape[0]
        l_plus = ds0.l_plus.values[0]
        altitude = ds0.altitude.values
        upper_alt = 2 * np.pi * l_plus


        #K------------------------------------
        K_file = ds0.K_file.values[0]
        print(K_file)
        dsK = xr.open_dataset(K_file, engine='netcdf4')
        K = dsK.K.values[0,::36000]
        
        range_number = int((files_num-1) / K.shape[0])
        Ks = dsK.K.values[0,::36000]
        for _ in range(range_number-1):
            Ks = np.hstack((Ks,K))
        Ks = np.hstack((Ks, K[-1])) #last 
        #Ks = Ks.reshape((files_num))
        #print("K:", Ks.shape)
        #--------------------------------------
        Ks = dsK.K.values[0,:]
        #u_bars, theta_bars = np.zeros((files_num, spatial_length,1))*np.nan, np.zeros((files_num, spatial_length,1))*np.nan
        u_bars, theta_bars = np.zeros((spatial_length,1, files_num))*np.nan, np.zeros((spatial_length,1, files_num))*np.nan
        print(u_bars.shape)
        t = np.arange(0, files_num)
        ts = np.repeat(t, spatial_length)
        altitudes = np.ones((spatial_length,1, files_num))*np.nan
        
        for i, file in enumerate(filtered_files):
            file_path = os.path.join(self.directory, file)
            ds = xr.open_dataset(file_path)
            
            u_bar = ds.u_bar.values
            theta_bar = ds.theta_bar.values
            alt = ds.altitude.values

            #u_bars[i,:,:] = u_bar
            #theta_bars[i,:,:] = theta_bar

            u_bars[:,:,i] = u_bar
            theta_bars[:,:,i] = theta_bar
            altitudes[:,:,i] = alt.reshape((alt.shape[0], 1))

        u_bars = u_bars.reshape((spatial_length, files_num))
        theta_bars = theta_bars.reshape((spatial_length, files_num))
        altitudes = altitudes.reshape((spatial_length, files_num))
        print(altitude.shape, t.shape, u_bars.shape, theta_bars.shape, altitudes.shape)

        return altitudes, ts, u_bars, theta_bars ,Ks

    def load_data_analytical(self): #netcdf for analytical not use K_file
        filtered_files = self.filter_files_by_time()
        files_num = len(filtered_files) #times
        
        first_file_path = os.path.join(self.directory, filtered_files[0])
        ds0 = xr.open_dataset(first_file_path)

        spatial_length = ds0.altitude.values.shape[0]
        l_plus = ds0.l_plus.values[0]
        altitude = ds0.altitude.values
        upper_alt = 2 * np.pi * l_plus

        u_bars, theta_bars = np.zeros((spatial_length,1, files_num))*np.nan, np.zeros((spatial_length,1, files_num))*np.nan
        print(u_bars.shape)
        t = np.arange(0, files_num)
        ts = np.repeat(t, spatial_length)
        altitudes = np.ones((spatial_length,1, files_num))*np.nan
        
        for i, file in enumerate(filtered_files):
            file_path = os.path.join(self.directory, file)
            ds = xr.open_dataset(file_path)
            
            u_bar = ds.u_bar.values
            theta_bar = ds.theta_bar.values
            alt = ds.altitude.values

            #u_bars[i,:,:] = u_bar
            #theta_bars[i,:,:] = theta_bar

            u_bars[:,:,i] = u_bar
            theta_bars[:,:,i] = theta_bar
            altitudes[:,:,i] = alt.reshape((alt.shape[0], 1))

        u_bars = u_bars.reshape((spatial_length, files_num))
        theta_bars = theta_bars.reshape((spatial_length, files_num))
        altitudes = altitudes.reshape((spatial_length, files_num))
        print(altitude.shape, t.shape, u_bars.shape, theta_bars.shape)

        return altitudes, ts, u_bars, theta_bars
#
def reshape_array_for_scatter(arr):
    spatial_length, time_length = arr.shape[0], arr.shape[1]
    return arr.T.reshape((1,time_length*spatial_length))

def extract_4th_day(altitude, t, u_bars, theta_bars, Ks):
    # 4日目のデータだけを抽出
    altitudes = altitude[:,:25]
    t_4th_day = t[:25*261]#np.arange(0, 25,1)#t[-25:]
    t_4th_day = t_4th_day.reshape((1,t_4th_day.shape[0]))
    u_bars_4th_day = u_bars[:, -25:]
    theta_bars_4th_day = theta_bars[:, -25:]
    Ks_4th_day = Ks[-25*864000:]
    """
    print('---')
    print(altitude.shape, altitudes.shape)
    print(t_4th_day)
    print("---")
    """
    return altitudes, t_4th_day, u_bars_4th_day, theta_bars_4th_day, Ks_4th_day,

import numpy as np

def get_max_min_info(t_4th_day, altitudes, u_bars_4th_day):
    #Find the maximum and minimum values in an array and their element numbers
    u_max = np.max(u_bars_4th_day)
    u_min = np.min(u_bars_4th_day)

    # 最大値・最小値のインデックス（すべて）
    max_indices = np.flatnonzero(u_bars_4th_day == u_max)
    min_indices = np.flatnonzero(u_bars_4th_day == u_min)

    # 複数あった場合に警告を出す
    if len(max_indices) > 1:
        print(f"警告: 最大値 {u_max:.3g} が {len(max_indices)} 箇所に存在します。最初の1つのみ使用します。")
    if len(min_indices) > 1:
        print(f"警告: 最小値 {u_min:.3g} が {len(min_indices)} 箇所に存在します。最初の1つのみ使用します。")

    # 最初のインデックスを使用
    idx_max = max_indices[0]
    idx_min = min_indices[0]

    # インデックスに対応する時刻・高度を取得
    t_max = t_4th_day[0, idx_max]
    t_min = t_4th_day[0, idx_min]
    alt_max = altitudes[0, idx_max]
    alt_min = altitudes[0, idx_min]

    return {
        'max_value': u_max,
        'min_value': u_min,
        't_at_max': t_max,
        'altitude_at_max': alt_max,
        't_at_min': t_min,
        'altitude_at_min': alt_min
    }



def plot_4th_day(directory, altitude, t, u_bars, theta_bars, Ks, confirm):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    
    # 4日目のデータだけを抽出
    altitudes, t_4th_day, u_bars_4th_day, theta_bars_4th_day, Ks_4th_day =  extract_4th_day(altitude, t, u_bars, theta_bars, Ks)

    day = np.unique(t_4th_day).shape[0]
    t_K = np.arange(0, 3600*(day-1), 0.1)
    t_K_hour = t_K/3600
    K_days = np.tile((Ks), int((day-1)/24) )

    altitudes = reshape_array_for_scatter(altitudes)
    #t_4th_day = reshape_array_for_scatter(t_4th_day)
  
    u_bars_4th_day = reshape_array_for_scatter(u_bars_4th_day)
    theta_bars_4th_day = reshape_array_for_scatter(theta_bars_4th_day)

    u_max_min =  get_max_min_info(t_4th_day, altitudes, u_bars_4th_day)
    theta_max_min =  get_max_min_info(t_4th_day, altitudes, theta_bars_4th_day)

    print("-----")
    print("altitudes: ", altitudes.shape)
    print("t_4th_day: ", t_4th_day.shape)
    print("u_bars_4th_day: ", u_bars_4th_day.shape)
    print("theta_bars_4th_day: ", theta_bars_4th_day.shape)
    print("K: ", Ks_4th_day.shape)
    print("-----")

    print(f"u最大値: {u_max_min['max_value']}（t={u_max_min['t_at_max']}, altitude={u_max_min['altitude_at_max']}）")
    print(f"u最小値: {u_max_min['min_value']}（t={u_max_min['t_at_min']}, altitude={u_max_min['altitude_at_min']}）")
    print(f"theta最大値: {theta_max_min['max_value']}（t={theta_max_min['t_at_max']}, altitude={theta_max_min['altitude_at_max']}）")
    print(f"theta最小値: {theta_max_min['min_value']}（t={theta_max_min['t_at_min']}, altitude={theta_max_min['altitude_at_min']}）")

    fontsize_title = 20
    fontsize_label = 16
    fontsize_tick = 14
    fontsize_colorbar = 18
    linewidth = 2
    
    vmin = min(np.nanmin(u_bars_4th_day), np.nanmin(theta_bars_4th_day))
    vmax = max(np.nanmax(u_bars_4th_day), np.nanmax(theta_bars_4th_day))
    cmap = "bwr"
    
    sc0 = ax[0].scatter(t_4th_day, altitudes, c=u_bars_4th_day, cmap=cmap, s=20, marker="o", vmin=vmin, vmax=vmax)
    sc1 = ax[1].scatter(t_4th_day, altitudes, c=theta_bars_4th_day, cmap=cmap, s=20, marker="o", vmin=vmin, vmax=vmax)

    #plot Max,Min values
    u_max_label = f"Max: {u_max_min['max_value']:.3g} m/s"
    u_min_label = f"Min: {u_max_min['min_value']:.3g} m/s"
    theta_max_label = f"Max: {theta_max_min['max_value']:.3g} K"
    theta_min_label = f"Min: {theta_max_min['min_value']:.3g} K"
    markersize=120

    ax[0].scatter(u_max_min["t_at_max"], u_max_min["altitude_at_max"], s=markersize, marker="*", color="white", edgecolors="crimson", label=u_max_label)
    ax[0].scatter(u_max_min["t_at_min"], u_max_min["altitude_at_min"], s=markersize, marker="*", color="white", edgecolors="navy", label=u_min_label)

    ax[1].scatter(theta_max_min["t_at_max"], theta_max_min["altitude_at_max"], s=markersize, marker="*", color="white", edgecolors="darkred", label=theta_max_label)
    ax[1].scatter(theta_max_min["t_at_min"], theta_max_min["altitude_at_min"], s=markersize, marker="*", color="white", edgecolors="darkblue", label=theta_min_label) 
    
    # カラーバー
    cbar = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    #cbar.set_label("Color scale", fontsize=fontsize_colorbar)
    cbar.ax.tick_params(labelsize=fontsize_tick)

    days = np.arange(0, 25, 6)

    for column in range(2):
        twinx = ax[column].twinx()
        twinx.plot(t_K_hour, K_days, color="black", alpha=0.7, linewidth=linewidth, label="K")
        twinx.set_ylabel("K", fontsize=fontsize_label)
        twinx.set_yticks(np.arange(0, 101, 20))
        twinx.tick_params(axis='y', labelsize=fontsize_tick)
        twinx.set_ylabel(r"K [m$^2$/s]", fontsize=fontsize_label)
        twinx.legend(loc="upper right", fontsize=fontsize_tick)


        # X軸
        ax[column].set_xlabel('t [hour]', fontsize=fontsize_label)
        ax[column].set_xticks(days)
        ax[column].tick_params(axis='x', labelsize=fontsize_tick)
        
        # Y軸（左）
        ax[column].set_ylabel('altitude [m]', fontsize=fontsize_label)
        ax[column].tick_params(axis='y', labelsize=fontsize_tick)
        
        ax[column].grid()
        ax[column].legend(loc="upper left", fontsize=fontsize_tick)
        # タイトル
        ax[0].set_title(r'$\overline{u}$', fontsize=fontsize_title)
        ax[1].set_title(r"$\overline{\theta}$", fontsize=fontsize_title)
    
    base_name = os.path.basename(directory.rstrip('/'))
    save_file_name = f"{directory}/{base_name}_4th_day_scatter.png"
    if confirm==True:
        plt.savefig(save_file_name, dpi=300)
    else:
        plt.show()
    plt.close()


#
def scatter_plot_u_theta(directory, altitudes, ts, u_bars, theta_bars, Ks,  confirm):

    altitudes = reshape_array_for_scatter(altitudes)
    #ts = reshape_array_for_scatter(ts)
    u_bars = reshape_array_for_scatter(u_bars)
    theta_bars = reshape_array_for_scatter(theta_bars)

    day = np.unique(ts).shape[0]

    t_K = np.arange(0, 3600*(day-1), 0.1)
    t_K_hour = t_K/3600

    K_days = np.tile((Ks), int((day-1)/24) )
    #print(day, t_K_hour.shape, K_days.shape)
    
    days = np.arange(0,day,24)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    # カラーバーの追加
    vmin = min(np.nanmin(u_bars), np.nanmin(theta_bars))
    vmax = max(np.nanmax(u_bars), np.nanmax(theta_bars))
    cmap = "bwr"
    sc0 = ax[0].scatter(ts, altitudes, c=u_bars, cmap=cmap, s=5, marker="o")
    sc1 = ax[1].scatter(ts, altitudes, c=theta_bars, cmap=cmap, s=5, marker="o")
    cbar = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)

    for column in range(2):
        twinx = ax[column].twinx()
        twinx.scatter(t_K_hour, K_days, s=1,color="black", alpha=0.7, marker=".")
    twinx.set_ylabel("K")
    twinx.set_yticks(np.arange(0,101,20))
    #labels and titles
    #ax[0].scatter(t_K_hour, K_days)
    ax[0].set_xlabel('t [hour]')
    ax[0].set_xticks(days)
    ax[0].set_ylabel('altitude [m]')
    ax[0].set_title(r'$\overline{u}$')
    ax[0].grid()
    ax[1].set_xlabel('t [hour]')
    ax[1].set_xticks(days)
    ax[1].set_ylabel('altitude [m]')
    ax[1].set_title(r"$\overline{\theta}$")
    ax[1].grid()
    base_name = os.path.basename(directory.rstrip('/'))
    save_file_name = f"{directory}/{base_name}.png"
        
    #save_dir2 = ""
    #save_file_name2 = f"{}/{base_name}.png"
    if confirm==True:
        print(altitudes.shape)
        print()
        plt.savefig(save_file_name, dpi=300)

    else:
        
        plt.show()
        
    plt.close()

#temporal file for analytical plot
def scatter_plot_u_theta_analytical(directory, altitudes, ts, u_bars, theta_bars,confirm):

    altitudes = reshape_array_for_scatter(altitudes)
    #ts = reshape_array_for_scatter(ts)
    u_bars = reshape_array_for_scatter(u_bars)
    theta_bars = reshape_array_for_scatter(theta_bars)

    day = np.unique(ts).shape[0]

    #t_K = np.arange(0, 3600*(day-1), 0.1)
    #t_K_hour = t_K/3600

    #K_days = np.tile((Ks), int((day-1)/24) )
    #print(day, t_K_hour.shape, K_days.shape)
    
    days = np.arange(0,day,24)
    if confirm==True:
        print(altitudes.shape)
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        # カラーバーの追加
        vmin = min(np.nanmin(u_bars), np.nanmin(theta_bars))
        vmax = max(np.nanmax(u_bars), np.nanmax(theta_bars))
        cmap = "bwr"
        sc0 = ax[0].scatter(ts, altitudes, c=u_bars, cmap=cmap, s=5, marker="o")
        sc1 = ax[1].scatter(ts, altitudes, c=theta_bars, cmap=cmap, s=5, marker="o")
        cbar = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        """
        for column in range(2):
            twinx = ax[column].twinx()
            twinx.scatter(t_K_hour, K_days, s=1,color="black", alpha=0.7, marker=".")
            twinx.set_ylabel("K")
            twinx.set_yticks(np.arange(0,101,20))
        """
        #labels and titles
        #ax[0].scatter(t_K_hour, K_days)
        ax[0].set_xlabel('t [hour]')
        ax[0].set_xticks(days)
        ax[0].set_ylabel('altitude [m]')
        ax[0].set_title(r'$\overline{u}$')
        ax[1].set_xlabel('t [hour]')
        ax[1].set_xticks(days)
        ax[1].set_ylabel('altitude [m]')
        ax[1].set_title(r"$\overline{\theta}$")
        base_name = os.path.basename(directory.rstrip('/'))
        save_file_name = f"{directory}/{base_name}.png"
        
        #save_dir2 = ""
        #save_file_name2 = f"{}/{base_name}.png"
        plt.savefig(save_file_name, dpi=300)
        plt.close()


def plot_u_theta(directory, altitude, t, u_bars, theta_bars, Ks, confirm):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    
    """
    print("altitude: ", altitude.shape)
    print("t:" , t.shape)
    print("u_bars: ",u_bars.shape)
    print("theta_bars: ", theta_bars.shape)
    #input('check the u_bars')
    """
    
    T, Alt = np.meshgrid(t, altitude)
    # カラーバーの追加
    vmin = min(np.nanmin(u_bars), np.nanmin(theta_bars))
    vmax = max(np.nanmax(u_bars), np.nanmax(theta_bars))

    # pcolormeshの描画
    color = "bwr"
    cmap = plt.get_cmap(color)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    shading = "auto"
    
    c1 = ax[0].pcolormesh(T, Alt, u_bars, norm=norm, cmap=cmap) 
    c2 = ax[1].pcolormesh(T, Alt, theta_bars, norm=norm, cmap=cmap)
    
    cbar = fig.colorbar(c1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    
    days = np.arange(0, t.shape[0], 24)

    #right axes for K plot
    for column in range(2):
        twinx = ax[column].twinx()
        twinx.scatter(t,Ks, s=5,color="black", alpha=0.7)
        twinx.set_ylabel("K")
        twinx.set_yticks(np.arange(0,101,20))
    
    #labels and titles
    ax[0].set_xlabel('t [hour]')
    ax[0].set_xticks(days)
    ax[0].set_ylabel('altitude [m]')
    ax[0].set_title(r'$\overline{u}$')
    ax[1].set_xlabel('t [hour]')
    ax[1].set_xticks(days)
    ax[1].set_ylabel('altitude [m]')
    ax[1].set_title(r"$\overline{\theta}$")
    
    if confirm==True:
        print("If you want to save the fig, you need -> confirm==False")
        plt.show()
    else:
        base_name = os.path.basename(directory.rstrip('/'))
        save_file_name = f"{directory}/{base_name}.png"
        
        #save_dir2 = ""
        #save_file_name2 = f"{}/{base_name}.png"
        plt.savefig(save_file_name, dpi=300)





        

def load_datasets(directory):
    processor = NetCDFProcessor(directory)
    altitude, t, u_bars, theta_bars = processor.load_data()
    return altitude, t, u_bars, theta_bars

    
def process_netcdf_directory(directory, confirm=False):
    processor = NetCDFProcessor(directory)
    altitude, t, u_bars, theta_bars, Ks = processor.load_data()
    plot_u_theta(directory, altitude, t, u_bars, theta_bars, Ks,confirm)

#
def process_netcdf_directory_scatter(directory, confirm=True):
    processor = NetCDFProcessor(directory)
    altitude, t, u_bars, theta_bars, Ks = processor.load_data()
    #scatter_plot_u_theta(directory,altitude, t, u_bars, theta_bars, Ks, confirm)
    plot_4th_day(directory, altitude, t, u_bars, theta_bars, Ks, confirm)

def process_netcdf_directory_scatter_analytical(directory, confirm=True):
    processor = NetCDFProcessor(directory)
    altitude, t, u_bars, theta_bars = processor.load_data_analytical()
    scatter_plot_u_theta_analytical(directory,altitude, t, u_bars, theta_bars, confirm)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    
    args = parser.parse_args()
    process_netcdf_directory_scatter(args.directory, confirm=True)
