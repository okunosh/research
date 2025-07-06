import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import xarray as xr
import glob

from ql_plot import NetCDFProcessor, process_netcdf_directory_scatter, extract_4th_day, reshape_variables_for_scatter, get_max_min_idxs, get_max_min_vals, plot_4th_day

#get parameters from netcdf 
def get_t0_file(directory):
    path = directory + "/*t0000000*" #t0 file
    t0_file = glob.glob(path)
    return t0_file[0]

def get_params_from_netcdf(netcdf_name):
    ds = xr.open_dataset(netcdf_name)
    return {var: ds[var].values for var in ds.data_vars}
#==========================


def get_data(directory, confirm):
    processor = NetCDFProcessor(directory)
    altitude, t, u_bars, theta_bars, Ks = processor.load_data()
    print("initial vals-----")
    print("u_bars_ini: ", u_bars.shape)
    print("theta_bars_ini: ", theta_bars.shape)
    print("altitude ini: ", altitude.shape)
    print("-----")
    #only 4th day data
    altitudes, t_4th_day, u_bars_4th_day, theta_bars_4th_day, Ks_4th_day =  extract_4th_day(altitude, t, u_bars, theta_bars, Ks)
    altitudes4scp, t_4th_day4scp, u_bars_4th_day4scp, theta_bars_4th_day4scp = reshape_variables_for_scatter(altitudes, t_4th_day, u_bars_4th_day, theta_bars_4th_day)

     #max, min
    u_max, u_min, u_max_idx, u_min_idx =  get_max_min_idxs(t_4th_day4scp, altitudes4scp, u_bars_4th_day4scp, "u")
    #print("umax_idx is :", u_max_idx)
    u_max_min =  get_max_min_vals(t_4th_day4scp, altitudes4scp, u_bars_4th_day4scp, u_max, u_min, u_max_idx, u_min_idx)

    theta_max, theta_min, theta_max_idx, theta_min_idx = get_max_min_idxs(t_4th_day4scp, altitudes4scp, theta_bars_4th_day4scp, "theta")
    theta_max_min =  get_max_min_vals(t_4th_day4scp, altitudes4scp, u_bars_4th_day4scp, theta_max, theta_min, theta_max_idx, theta_min_idx)

    
    #plot_4th_day(directory, altitudes4scp, t_4th_day4scp, u_bars_4th_day4scp, theta_bars_4th_day4scp, Ks_4th_day, u_max_min, theta_max_min, confirm)
    print("-----")
    print("altitude.shape: ", altitudes4scp.shape)
    print("t_4th_day4scp.shape: ", t_4th_day4scp.shape)
    print("u_bars_4th_day4scp.shape: ", u_bars_4th_day4scp.shape)
    print("theta_bars_4th_day4scp.shape: ", theta_bars_4th_day4scp.shape)
    print("-----")

    return altitudes4scp, t_4th_day4scp, u_bars_4th_day4scp, theta_bars_4th_day4scp

import numpy as np
"""

def compute_vertical_gradients(altitude, u, theta, gamma,  n_levels=261, n_times=25):
    # 2次元（高度 × 時間）に reshape
    altitude_2d = altitude.reshape((n_times, n_levels)).T
    u_2d = u.reshape((n_times, n_levels)).T
    theta_2d = theta.reshape((n_times, n_levels)).T

    # 微分結果格納用
    du_dz = np.full_like(u_2d, np.nan)
    dtheta_dz = np.full_like(theta_2d, np.nan)

    vmin = min(np.nanmin(u), np.nanmin(theta))
    vmax = max(np.nanmax(u), np.nanmax(theta))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


    for t in range(n_times):
        alt = altitude_2d[:, t]
        u_t = u_2d[:, t]
        theta_t = theta_2d[:, t]
        print("-----")
        print("t={}".format(t))
        print("alt: ", alt.shape)
        print("u_t: ", u_t.shape)
        print("t: ", np.ones_like(n_levels)*t)
        print("-----")
        
        dz = np.gradient(alt)
        du = np.gradient(u_t)
        dtheta = np.gradient(theta_t)

        du_dz[:, t] = du / dz
        dtheta_dz[:, t] = dtheta / dz + gamma

        plt.scatter(np.ones((n_levels))*t, alt, c=u_t, cmap="bwr", s=5, marker="o",norm=norm)
   
    cbar = plt.colorbar()
    cbar.set_label('u Value')
    plt.grid()
    plt.ylabel('alt')
    plt.show()

    return du_dz, dtheta_dz, altitude_2d
"""

def compute_vertical_gradients(altitude, u, theta, time_array, gamma, theta0):
    # reshape back to
    times_num = 261
    spaces_num = 6525
    n_time = spaces_num // times_num
    z = altitude.reshape(n_time, times_num)
    u_t = u.reshape(n_time, times_num)
    theta_t = theta.reshape(n_time, times_num)
    t = time_array.reshape(n_time, times_num)
    # z = altitude.reshape(times_num, n_time)
    # u_t = u.reshape(times_num, n_time)
    # theta_t = theta.reshape(times_num, n_time)
    # t = time_array.reshape(times_num, n_time)

    dz = z[:, 1:] - z[:, :-1]
    du = u_t[:, 1:] - u_t[:, :-1]
    dtheta = theta_t[:, 1:] - theta_t[:, :-1]
    for j in range(0, 25):
        print("theta_lower value at top layer at  LT{}:{} ".format(j, theta_t[j, -2]))
        print("u_lower value at top layer at  LT{}:{} ".format(j, u_t[j, -2]))
        print("---")
    dudz = du / dz
    dthetadz = dtheta / dz + gamma

    z_mid = 0.5 * (z[:, :-1] + z[:, 1:])
    t_mid = 0.5 * (t[:, :-1] + t[:, 1:])
    #t_mid = t

    #theta0 = 210
    theta_bottom = theta0 + gamma * z[:, :-1] + theta_t[:, :-1]
    theta_top = theta0 + gamma * z[:, 1:] + theta_t[:, 1:]
    
    theta_mid = 0.5 * (theta_top + theta_bottom)

    #print("ut:", u_t.shape, u_t[:,13].shape, u_t[:,13])


    #plot
    vmin = min(np.nanmin(u), np.nanmin(theta))
    vmax = max(np.nanmax(u), np.nanmax(theta))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.scatter(t, z, c=u_t, cmap="bwr", s=10, marker="o",norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('u Value')
    plt.grid()
    plt.ylabel('alt')
    plt.show()

    return dudz, dthetadz, z_mid, theta_mid, t_mid

def compute_richardson_number_from_gradients(dthetadz, dudz, theta_mid, z_mid):
    """
    勾配からリチャードソン数を計算する関数。
    theta_mid, z_mid は中点（層の中間）で定義された値。
    """
    # 定数
    g = 3.721  # 火星重力加速度 [m/s^2]

    # リチャードソン数の式: Ri = (g/θ) * (dθ/dz) / (du/dz)^2
    Ri = (g / theta_mid) * (dthetadz) / (dudz ** 2 + 1e-8)  # 分母がゼロに近づくのを防ぐ

    return Ri



def plot_vertical_gradients(directory, du_dz, dtheta_dz, altitude_2d, time_2d):
    # フラット化
    time_flat = time_2d.flatten() # [hours]
    altitude_flat = altitude_2d.flatten()
    du_dz_flat = du_dz.flatten()
    dtheta_dz_flat = dtheta_dz.flatten()
    print(time_flat.shape, altitude_flat.shape, du_dz_flat.shape) #time_flat

    # 共通カラーバーの min/max を決定
    vmin = min(np.nanmin(du_dz_flat), np.nanmin(dtheta_dz_flat))
    vmax = max(np.nanmax(du_dz_flat), np.nanmax(dtheta_dz_flat))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # プロット
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1], 'right': 0.85})
    
    sc1 = axs[0].scatter(time_flat, altitude_flat, c=du_dz_flat, cmap='bwr', norm=norm, s=5)
    axs[0].set_ylabel('Altitude [m]')
    axs[0].set_title('Vertical Wind Shear ∂u/∂z [1/s]')
    axs[0].grid(True)

    sc2 = axs[1].scatter(time_flat, altitude_flat, c=dtheta_dz_flat, cmap='bwr',norm=norm, s=5)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Altitude [m]')
    axs[1].set_title('Potential Temperature Gradient ∂θ/∂z [K/m]')
    axs[1].grid(True)

    # カラーバーを個別に定義された軸に追加
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.76])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label('Gradient Value')
    #plt.show()
    plt.savefig(directory+"/H)vertical_gradient.png",dpi=300, bbox_inches="tight", pad_inches=0.05)

def plot_vertical_gradients_eachfig(directory, du_dz, dtheta_dz, altitude_2d, time_2d, theta_mid):
    print("altitude2d: ", altitude_2d.shape)
    print("time2d:", time_2d.shape)
    #convert 2dim arrays into 2dim for scatter plot.
    time_flat = time_2d.flatten()  # [hours]
    altitude_flat = altitude_2d.flatten()
    du_dz_flat = du_dz.flatten()
    dtheta_dz_flat = dtheta_dz.flatten()
    theta_mid_flat = theta_mid.flatten()

    # common colorbar min/max
    vmin = min(np.nanmin(du_dz_flat), np.nanmin(dtheta_dz_flat))
    vmax = max(np.nanmax(du_dz_flat), np.nanmax(dtheta_dz_flat))
    #norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    norm = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)
    cmap="bwr"
    

    # === fig1: ∂u/∂z ===
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sc1 = ax1.scatter(time_flat, altitude_flat, c=du_dz_flat, cmap=cmap, norm=norm, s=5)
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('Vertical Wind Shear ∂u/∂z [1/s]')
    ax1.set_xlabel('Time')
    ax1.grid(True)
    cbar1 = fig1.colorbar(sc1, ax=ax1)
    cbar1.set_label('Gradient Value')

    # === fig2: ∂θ/∂z ===
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sc2 = ax2.scatter(time_flat, altitude_flat, c=dtheta_dz_flat, cmap=cmap, norm=norm, s=5)
    ax2.set_ylabel('Altitude [m]')
    ax2.set_title('Potential Temperature Gradient ∂θ/∂z [K/m]')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    cbar2 = fig2.colorbar(sc2, ax=ax2)
    cbar2.set_label('Gradient Value')
    #confirm
    #for j in range(0, 25):
        #print("dtheta/dz value at top layer at  LT{}:{} ".format(j, dtheta_dz[j, -1]))

    vmin_theta = min(theta_mid_flat)
    vmax_theta = max(theta_mid_flat)
    cmap_theta = "jet"
    #==fig3: theta_mid ===
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sc3 = ax3.scatter(time_flat, altitude_flat, c=theta_mid_flat, cmap=cmap_theta, vmax=vmax_theta, vmin=vmin_theta, s=5)
    ax3.set_ylabel('Altitude [m]')
    ax3.set_title('theta_mid [K]')
    ax3.set_xlabel('Time')
    ax3.grid(True)
    cbar3 = fig3.colorbar(sc3, ax=ax3)
    cbar3.set_label('theta')

    #fig4: theta_mid some times
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    plot_hours = [2, 7, 9, 12, 15, 18, 20]

    for i in range(time_2d.shape[0]):
        hour = time_2d[i, 0]

        if hour in plot_hours:
            ax4.plot(theta_mid[i,:], altitude_2d[i,:], label=f"{int(hour):02d}:00")
            
    ax4.set_ylabel('Altitude [m]')
    ax4.set_title('theta [K]')
    ax4.set_xlabel('Potential Temperature [K]')
    ax4.legend(title="Local Time")
    ax4.grid(True)
    #plt.show()

    fig1.savefig(directory+"/G)u_gradient.png",dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig2.savefig(directory+"/F)theta_gradient.png",dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig3.savefig(directory+"/E)theta_mid.png",dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig4.savefig(directory+"/C)theta_mid_some.png",dpi=300, bbox_inches="tight", pad_inches=0.05)
    
def plot_richardson_number(directory, Ri, t_mid, z_mid):
    """
    リチャードソン数を散布図で可視化する。
    横軸：時間、縦軸：高度、色：Ri値（0.25が白になるよう調整）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # フラット化
    x = t_mid.flatten()
    y = z_mid.flatten()
    c = Ri.flatten()

    # 色設定（Ri=0.25を白に）
    vmin = -5#np.nanmin(c)
    vmax = 5#np.nanmax(c)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.25, vmax=vmax)

    # 散布図描画
    sc = ax.scatter(x, y, c=c, cmap='bwr', norm=norm, s=5)

    # カラーバー
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Ri')

    # 軸ラベルなど
    ax.set_xlabel('Time')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Richardson Number')
    # カラーバーの目盛り位置を手動指定（例：vmin, 0.25, vmax）
    ticks = [-4, -2, 0.25, 2, 4]
    tick_labels = ['-4', '-2', '0.25', '2', '4']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    
    plt.tight_layout()
    plt.grid()
    plt.savefig(directory+"/D)RichardsonNumber_plot.png",dpi=300, bbox_inches="tight", pad_inches=0.05)
    
    #plt.show()


    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    
    args = parser.parse_args()
    #process_netcdf_directory_scatter(args.directory, confirm=False)
    altitudes4scp, t_4th_day4scp, u_bars_4th_day4scp, theta_bars_4th_day4scp = get_data(args.directory, confirm=False)

    print(args.directory)
    t0_file = get_t0_file(args.directory)
    print(t0_file)
    netcdf_vals = get_params_from_netcdf(t0_file)
    gamma = netcdf_vals["gamma"]
    theta0 = netcdf_vals["theta_0"]
    
    du_dz, dtheta_dz, z_mid, theta_mid, t_mid= compute_vertical_gradients(
    altitudes4scp,
    u_bars_4th_day4scp,
    theta_bars_4th_day4scp,
    t_4th_day4scp,
    gamma,
    theta0
    )
    """
    n_levels=261,
    n_times=25,
   ) 
    """
    #time_2d = t_4th_day4scp.reshape((25, 261)).T  # 時刻配列の整形
    plot_vertical_gradients(args.directory, du_dz, dtheta_dz, z_mid, t_mid)
    plot_vertical_gradients_eachfig(args.directory, du_dz, dtheta_dz, z_mid, t_mid, theta_mid)
    # calc Richardson Number
    Ri = compute_richardson_number_from_gradients(dtheta_dz, du_dz, theta_mid, z_mid)

    plot_richardson_number(args.directory, Ri, t_mid, z_mid)
