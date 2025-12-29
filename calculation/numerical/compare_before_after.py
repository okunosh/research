import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import TwoSlopeNorm

# ディレクトリパスを指定
dir_path = "output/validation_dn_after/K_variable/Earth"
before_dir = "20250501T191206_SubCritical_num11_alp0-24_gamma3-00_dt0.1" #OK

dir_path2 = "output/validation_dn_after/Earth"
after_dir = "20250502T164610_SubCritical_num11_alp0-24_gamma3-00_dt0.1"

before_path = os.path.join(dir_path, before_dir)
after_path = os.path.join(dir_path2, after_dir)

# ディレクトリ内のすべてのファイルを取得
before_files = [f for f in os.listdir(before_path) if f.endswith('.nc')]
after_files = [f for f in os.listdir(after_path) if f.endswith('.nc')]

#print(before_files)
#print('-----------')
#print(after_files)
#input('stop')
# tの値を抽出してソートする関数
def extract_t_value(file_name):
    return int(file_name.split('_t')[1].split('.')[0])
def extract_t_value_after(file_name):
    return int(file_name.split('_t')[1].split('_n')[0])

# ファイル名をtの値でソート
before_files_sorted = sorted(before_files, key=extract_t_value)
after_files_sorted = sorted(after_files, key=extract_t_value)

#print(before_files_sorted)
#print("-------------")
#print(after_files_sorted)

#input("OK")

#TEST FILE
before_test_file = os.path.join(before_path, before_files_sorted[0])
after_test_file = os.path.join(after_path, after_files_sorted[0])

ds_before_test = xr.open_dataset(before_test_file)
ds_after_test = xr.open_dataset(after_test_file)

u_bar_before_test = ds_before_test.u_bar.values
theta_bar_before_test = ds_before_test.theta_bar.values

u_bar_after_test = ds_after_test.u_bar.values
theta_bar_after_test = ds_after_test.theta_bar.values

if u_bar_before_test.shape != u_bar_after_test.shape:
    print("The shapes are different! You need check the directories which you compare. ")

u_bar_diffs = np.ones( (u_bar_before_test.shape[0],len(before_files)) )*np.nan
theta_bar_diffs = np.ones( (theta_bar_before_test.shape[0], len(before_files)) )*np.nan

# ファイル名が一致するファイルに対して処理を行う
for k, (before_file, after_file) in enumerate(zip(before_files_sorted, after_files_sorted)):
    before = os.path.join(before_path, before_file)
    after = os.path.join(after_path, after_file)
    
    # データセットを読み込む
    ds_a = xr.open_dataset(after, decode_times=False)
    ds_b = xr.open_dataset(before, decode_times=False)
    
    # u_bar と theta_bar の値が一致しているかを確認
    u_bar_equal = np.array_equal(ds_a['u_bar'].values, ds_b['u_bar'].values)
    theta_bar_equal = np.array_equal(ds_a['theta_bar'].values, ds_b['theta_bar'].values)
    
    print(f"Comparing {before_file} and {after_file}:")
    print(f"u_bar values are identical: {u_bar_equal}")
    print(f"theta_bar values are identical: {theta_bar_equal}")
    
    # 一致しない場合の差分を確認
    if not u_bar_equal:
        print("Differences in u_bar:")
        u_bar_diffs[:,k] = (ds_a['u_bar'].values - ds_b['u_bar'].values).reshape(u_bar_diffs.shape[0])
        
    
    if not theta_bar_equal:
        print("Differences in theta_bar:")
        theta_bar_diffs[:,k] = (ds_a['theta_bar'].values - ds_b['theta_bar'].values).reshape(theta_bar_diffs.shape[0])

print(u_bar_diffs)


def plot_u_theta(altitude, t, u_bars, theta_bars, confirm):
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
    """
    #right axes for K plot
    for column in range(2):
        twinx = ax[column].twinx()
        twinx.scatter(t,Ks, s=5,color="black", alpha=0.7)
        twinx.set_ylabel("K")
        twinx.set_yticks(np.arange(0,101,20))
    """
    #labels and titles
    ax[0].set_xlabel('t [hour]')
    ax[0].set_xticks(days)
    ax[0].set_ylabel('altitude [m]')
    ax[0].set_title(r'$\overline{u}$')
    ax[1].set_xlabel('t [hour]')
    ax[1].set_xticks(days)
    ax[1].set_ylabel('altitude [m]')
    ax[1].set_title(r"$\overline{\theta}$")
    plt.show()
    
n = np.arange(0,ds_before_test.altitude.values.shape[0]+1,1)
t = np.arange(0, len(before_files)+1,1)
plot_u_theta(n, t, u_bar_diffs, theta_bar_diffs, confirm=True)
