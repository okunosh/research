import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import re

def load_K_and_z_from_netcdf(filepath, K_varname='K', z_varname='altitude'):
    ds = xr.open_dataset(filepath)
    K_z_t = ds[K_varname].values  # shape = (261, 864000)
    z_z_t = ds[z_varname].values  # shape = (261, 864000)
    ds.close()
    return K_z_t, z_z_t

def load_surface_forcing(filepath):
    ds = xr.open_dataset(filepath)
    surf_tmp = ds["theta_0"].values
    ds.close()
    return surf_tmp

#----------------------------------
def extract_netcdf_files(directory):
    """extract all netcdf in the directory"""
    files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    sorted_files = sorted(files, key=lambda x: int(x.split('_t')[1].split('.')[0]))
    #print("sorted_files: ", sorted_files)
    return sorted_files

def filter_files_by_time(netcdf_files):
    """extract files in each 3 hours"""
    filtered_files = []
    for file in netcdf_files:
        match = re.search(r't(\d+)\.nc', file)
        if match:
            time_in_seconds = int(match.group(1))

            if time_in_seconds % 3600==0:#10800 == 0:  # 10800sec = 3 hours
                filtered_files.append(file)
    #print("files: ", filtered_files)
    return filtered_files

def load_data(directory): #netcdf
    extracted_files = extract_netcdf_files(directory)
    filtered_files = filter_files_by_time(extracted_files)
    files_num = len(filtered_files) #times

    first_file_path = os.path.join(directory, filtered_files[0])
    print(first_file_path)
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
        file_path = os.path.join(directory, file)
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


#----------------------------------------------------

def generate_gamma_variable_z(K_z_t, z_z_t, gamma1=0.003, gamma2=0.0060,
                              delta1=20, delta2=30, z1_min=30, z1_max=80,
                              z2_min=150, z2_max=300):

    print(f"K_z_t.shape: {K_z_t.shape}, z_z_t.shape: {z_z_t.shape}")
    nt, nz = K_z_t.shape[0], z_z_t.shape[0]
    print(f"nz: {nz}, nt: {nt}")
    gamma_zt = np.zeros((nz, nt))
    z1_t = np.zeros(nt)
    z2_t = np.zeros(nt)

    #K_t_mean = np.mean(K_z_t)
    Kmin = np.min(K_z_t)
    Kmax = np.max(K_z_t)

    #z_z_t = np.tile(np.arange(nz), (nt, 1)).T
    print(f"z_z_t: {z_z_t.shape}")
    
            
    for t in range(nt):
        z_t = z_z_t[:, t]
        z_t_max = np.max(z_t)
        z_t_min = np.min(z_t)
        z_t_range = (z_t_max - z_t_min)
        
        K_mean = K_z_t[t]
        

        ratio = np.clip((K_mean - Kmin) / (Kmax - Kmin), 0, 1)

        rate = 1.e-3
        z1 = z_t_min + rate * (1 + ratio) * z_t_range         # 地表層厚
        neutral_thickness = (rate + (1 - rate) * ratio) * z_t_range   # 中立層厚
        z2 = z1 + neutral_thickness

        #print(f"z1: {z1}, z2: {z2}")
        # 安全なクリップ処理（z2がz1より下回らない、上限を越えない）
        z2 = np.clip(z2, z1 + 0.01 * z_t_range, z_t_max - 0.01 * z_t_range)

        z1_t[t] = z1
        z2_t[t] = z2

        """
        print("t: LT", t)
        print("ratio: ", ratio)
        print("K_mean", K_mean)
        print("zmin: ", z_t_min)
        print("z_t_range: ", z_t_range)
        print("z1: ", z1)
        print("z2: ", z2)
        print("-------------")
        """
        S1 = 0.5 * (1 + np.tanh((z_t - z1) / delta1))
        S2 = 0.5 * (1 + np.tanh((z_t - z2) / delta2))
        gamma_zt[:, t] = gamma1 * (1 - S1)# + gamma2# * S2

    return gamma_zt, z1_t, z2_t
def compute_theta_variable_z(gamma_zt, z_z_t, theta_bars):
    nz, nt = gamma_zt.shape
    theta_zt = np.zeros_like(gamma_zt)

    for t in range(nt):
        #print("theta0_t: ", theta0_t[t])
        #print(f"theta_bar: {theta_bars.shape}")
        z_t = z_z_t[:, t]
        dz_t = np.diff(z_t, prepend=z_t[0])  # 差分で積分
        theta_zt[:, t] = 210 + np.cumsum(gamma_zt[:, t] * dz_t) + theta_bars[:, t]

    return theta_zt

def plot_theta_profiles_variable_z(theta_zt, gamma_zt, z_z_t, local_times,
                                   z1_t, z2_t, dt=0.1, colors=None):
    nt = theta_zt.shape[1]
    time_axis = np.arange(nt) * dt
    if colors is None:
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'gray']

    fig1, ax1 = plt.subplots(figsize=(6,8))
    fig2, ax2 = plt.subplots(figsize=(6,8))
    
    for i, hour in enumerate(local_times):
        #t_sec = hour * 3600
        #idx = np.argmin(np.abs(time_axis - t_sec))
        idx = int(hour)
        #print("idx: ", idx)
        #print("z_max", z_z_t[-1, idx])

        gamma = gamma_zt[:, idx]
        theta = theta_zt[:, idx]
        z = z_z_t[:, idx]
        z1 = z1_t[idx]
        z2 = z2_t[idx]
        ax1.plot(gamma, z, label=f"LT{hour}", color=colors[i % len(colors)])
        ax2.plot(theta, z, label=f"LT{hour}", color=colors[i % len(colors)])

        ax1.axhline(z1, color=colors[i % len(colors)], linestyle='--', linewidth=1, alpha=0.6)
        ax1.axhline(z2, color=colors[i % len(colors)], linestyle=':', linewidth=1, alpha=0.6)

        #ax2.axhline(z1, color=colors[i % len(colors)], linestyle='--', linewidth=1, alpha=0.6)
        #ax2.axhline(z2, color=colors[i % len(colors)], linestyle=':', linewidth=1, alpha=0.6)

    ax1.set_xlabel("gamma [K/m]")
    ax1.set_ylabel("z [m]")
    ax2.set_xlabel("theta [K]")
    ax2.set_ylabel("z [m]")

    #plt.ylim(0, 1000)
    #plt.title("温位プロファイルと遷移高度（z₁:--, z₂::）")
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show(block=False)

def plot_neutral_layer_thickness(z1_t, z2_t, z_z_t, dt=3600):
    """
    z1_t, z2_t: shape = (nt,)   各時刻における遷移高度 [m]
    dt: 時間刻み（秒） 例: 3600（1時間ごと）

    出力：中立層厚さ (z2 - z1) の日変化グラフ
    """

    nt = len(z1_t)
    time_hours = np.arange(nt) * dt / 3600  # 単位を「時間」に変換

    neutral_thickness = z2_t - z1_t
    z_max_t = z_z_t[-1, :]

    #print("z_z_t_shape: ", z_z_t.shape)
    #print("z_max_t_shape: ", z_max_t.shape)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(time_hours, neutral_thickness, color='blue', linewidth=2)
    ax1.fill_between(time_hours, neutral_thickness, color='blue', alpha=0.2)
    ax1.set_xlabel("LT [hour]")
    ax1.set_ylabel("neutral thickness [m]")
    #plt.title("中立層の厚さの時間変化")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(time_hours, z1_t, color="green", label="z1")
    ax2.plot(time_hours, z2_t, color="blue", label="z2")
    ax2.plot(time_hours, z_max_t, color="red", label="zmax")
    ax2.set_xlabel("LT [hour]")
    ax2.set_ylabel("altitude [m]")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    
    plt.show()

#====================



def plot_gamma_index_simple(gamma_zt, title="γ(z, t) distribution (Index View)"):
    """
    γ(z, t) の格子点インデックス同士でのカラーマップ
    """
    plt.figure(figsize=(10, 6))
    im = plt.imshow(gamma_zt, aspect='auto', origin='lower', cmap='plasma')
    plt.colorbar(im, label="γ [K/m]")
    plt.xlabel("Time index")
    plt.ylabel("Grid index (z)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


#=====================


def save_gamma_to_netcdf(gamma_zt: np.ndarray, filename: str):
    """
    時間・高度依存の温位傾度 γ(z, t) を NetCDF ファイルとして保存する。

    Parameters:
        gamma_zt : np.ndarray
            shape = (nz, nt) の2次元配列。温位傾度 [K/m]。
        filename : str
            保存する NetCDF ファイル名（例: "gamma_variable.nc"）
    """

    # ---------- 形状チェック ----------
    expected_shape = (261, 864000)
    if gamma_zt.shape != expected_shape:
        raise ValueError(f"[ERROR] gamma_zt.shape = {gamma_zt.shape} が想定と異なります。必要な形状: {expected_shape}")
    nz, nt = gamma_zt.shape

    ds = xr.Dataset(
        {"gamma": (("t", "z"), gamma_zt.T)},
        coords={
            "t": np.arange(nt),
            "z": np.arange(nz),
        }
    )

    # メタデータ（NetCDFビューアでの可読性向上）
    ds["gamma"].attrs["units"] = "K/m"
    ds["gamma"].attrs["long_name"] = "potential temperature gradient"
    ds["z"].attrs["long_name"] = "vertical grid index"
    ds["t"].attrs["long_name"] = "time index"

    # 保存処理
    ds.to_netcdf(filename)
    print(f"[INFO] gamma(z, t) saved to {filename} (shape: {gamma_zt.shape})")


if __name__ == "__main__":
    import argparse as ar
    parser = ar.ArgumentParser(description='NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    args = parser.parse_args()

    z_z_t, ts, u_bars, theta_bars ,Ks = load_data(args.directory)
    z_z_t, ts, u_bars, theta_bars ,Ks = extract_4th_day(z_z_t, ts, u_bars, theta_bars, Ks)
    print("Ks: ", Ks.shape)#864000
    # 各時間に対応するインデックス（1時間 = 3600秒 ⇒ 36000ステップ）
    dt = 0.1  # 秒

    steps_per_hour = int(3600 / dt)  # = 36000
    #selected_indices = np.arange(0, 864001, steps_per_hour)  # 0〜828000まで24点
    Ks25 = Ks[::steps_per_hour]
    Ks25 = np.append(Ks25, Ks25[-1])
    
    # --- 1. データ読み込み ---
    filepath = "K/K_261.nc"
    K_z_t, z_z_t_nouse = load_K_and_z_from_netcdf(filepath)
    
    # --- 2. 地表温位 θ₀(t) を与える（日変化モデル）---
    
    # --- 3. γ, z₁, z₂ を計算 ---
    #idx
    z_z_t_idx = np.tile(np.arange(261), (864000, 1)).T
    gamma_zt_idx, z1_t_idx, z2_t_idx = generate_gamma_variable_z(Ks, z_z_t_idx)
    gamma_zt, z1_t, z2_t = generate_gamma_variable_z(Ks25, z_z_t)
    print("gamma_zt: ", gamma_zt.shape, gamma_zt)
    print("z_z_t: ", z_z_t.shape)
    print("Ks: ", Ks.shape)
    #input('stop')
    # --- 4. θ(z, t) を計算 ---
    theta_zt = compute_theta_variable_z(gamma_zt, z_z_t, theta_bars)
    
    # --- 5. 指定時刻でプロット ---
    plot_theta_profiles_variable_z(theta_zt, gamma_zt, z_z_t,
                                   local_times=[0, 6, 9, 12, 15, 18],
                                   z1_t=z1_t, z2_t=z2_t)
    plot_neutral_layer_thickness(z1_t, z2_t, z_z_t, dt=3600)

    #---6 save ---
    filename = "test_spaciotmp_gamma0717.nc"
    save_gamma_to_netcdf(gamma_zt_idx, filename)


    
    
