import os
import numpy as np
import xarray as xr
import argparse
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import TwoSlopeNorm


from ql_plot import NetCDFProcessor  # ql_plot.pyからインポート

def read_variable(file_path, variable_name):
    ds = xr.open_dataset(file_path, decode_times=False)
    return ds[variable_name].values

def main(analytical_dir, numerical_dir):
    analytical_processor = NetCDFProcessor(analytical_dir)
    numerical_processor = NetCDFProcessor(numerical_dir)

    analytical_files = analytical_processor.extract_netcdf_files()
    numerical_files = numerical_processor.extract_netcdf_files()
    #print(numerical_files)
    #input('stop')
    num_files = len(analytical_files)
    #print(num_files, len(numerical_files))
    #num = read_variable(os.path.join(numerical_dir, analytical_files[0]), '')
    num = 11
    diffs_u_bar = np.zeros((num, 1, num_files))*np.nan  # change the numbers
    diffs_theta_bar = np.zeros((num,1, num_files))*np.nan  # change the numbers

    surface_theta = np.ones((num_files))*np.nan

    for i, a_file in enumerate(analytical_files):
        #print(a_file)
        t_index = a_file.split('_t')[-1].split('.')[0]
        #print(t_index)
        #input('stop')
        
        # 対応するn_fileを探す
        pattern = f'N_*_t{t_index}.nc'  # ワイルドカードを使用
        matching_files = fnmatch.filter(numerical_files, pattern)
        print(matching_files)

        if matching_files:
            n_file = matching_files[0]  # 最初の一致するファイルを使用
            #print(f"Processing: {a_file} and {n_file}")
        
            a_u_bar = read_variable(os.path.join(analytical_dir, a_file), 'u_bar')
            a_theta_bar = read_variable(os.path.join(analytical_dir, a_file), 'theta_bar')
            n_u_bar = read_variable(os.path.join(numerical_dir, n_file), 'u_bar')
            n_theta_bar = read_variable(os.path.join(numerical_dir, n_file), 'theta_bar')

            print(a_theta_bar[0,0])
            surface_theta[i] = a_theta_bar[0][0]

            #difference between analytical and numerical
            dif_u_bar = a_u_bar - n_u_bar
            dif_theta_bar = a_theta_bar - n_theta_bar

            diffs_u_bar[:,:,i] = dif_u_bar
            diffs_theta_bar[:,:,i] = dif_theta_bar
        else:
            print(f"No matching numerical file for {a_file}")

    diffs_u_bar_reshaped = diffs_u_bar.reshape(num, num_files)
    diffs_theta_bar_reshaped = diffs_theta_bar.reshape(num, num_files)
    altitude = read_variable(os.path.join(numerical_dir, n_file), 'altitude')
    t = np.arange(0, num_files)

    plt.plot(t, surface_theta)
    plt.grid()
    plt.title("surface theta")
    plt.xticks(np.arange(0,97,24))
    plt.xlabel('t[hour]')
    plt.show()

    # 最終的な結果を表示
    print("Final u_bar differences:")
    print(diffs_u_bar_reshaped.shape)
    print("Final theta_bar differences:")
    print(diffs_theta_bar_reshaped.shape)
    return diffs_u_bar_reshaped, diffs_theta_bar_reshaped, altitude, t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NetCDF files from analytical and numerical directories.')
    parser.add_argument('analytical_directory', type=str, help='Directory containing analytical NetCDF files')
    parser.add_argument('numerical_directory', type=str, help='Directory containing numerical NetCDF files')

    args = parser.parse_args()

    diffs_u_bar_reshaped, diffs_theta_bar_reshaped, altitude, t = main(args.analytical_directory, args.numerical_directory)
    T, Alt = np.meshgrid(t, altitude)
    # サブプロットの作成
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

     # カラーバーの追加
    vmin = min(np.min(diffs_u_bar_reshaped), np.min(diffs_theta_bar_reshaped))
    vmax = max(np.max(diffs_u_bar_reshaped), np.max(diffs_theta_bar_reshaped))

    # pcolormeshの描画
    color = "bwr"
    cmap = plt.get_cmap(color)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    shading = "auto"

    c1 = ax[0].pcolormesh(T, Alt, diffs_u_bar_reshaped, norm=norm, cmap=cmap) 
    c2 = ax[1].pcolormesh(T, Alt, diffs_theta_bar_reshaped, norm=norm, cmap=cmap)

    cbar = fig.colorbar(c1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.ax.set_yscale('linear')

    # 軸ラベルとタイトルの設定
    ax[0].set_xticks(np.arange(0, 97, 24))
    ax[0].set_xlabel('t [hour]')
    ax[0].set_ylabel('altitude [m]')
    ax[0].set_title(r'$\overline{u}_{analytical}$ - $\overline{u}_{numerical}$')
    ax[1].set_xticks(np.arange(0, 97, 24))
    ax[1].set_xlabel('t [hour]')
    ax[1].set_ylabel('altitude [m]')
    ax[1].set_title(r"$\overline{\theta}_{analytical}$ - $\overline{\theta}_{numerical}$")
    
    # プロットの表示
    plt.show()
