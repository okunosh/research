import os
import numpy as np
import xarray as xr
import argparse
import fnmatch
import matplotlib.pyplot as plt
from ql_plot import NetCDFProcessor  # ql_plot.pyからインポート

def read_variable(file_path, variable_name):
    ds = xr.open_dataset(file_path, decode_times=False)
    return ds[variable_name].values

def main(analytical_dir, numerical_dir):
    analytical_processor = NetCDFProcessor(analytical_dir)
    numerical_processor = NetCDFProcessor(numerical_dir)

    analytical_files = analytical_processor.extract_netcdf_files()
    numerical_files = numerical_processor.extract_netcdf_files()
    print(analytical_files)
    input('stop')
    num_files = len(analytical_files)
    #input('stop')

    diffs_u_bar = np.zeros((701, 1, num_files))*np.nan  # change the numbers
    diffs_theta_bar = np.zeros((701,1, num_files))*np.nan  # change the numbers

    for i, a_file in enumerate(analytical_files):
        t_index = a_file.split('_t')[-1].split('.')[0]
        
        # 対応するn_fileを探す
        pattern = f'N_*_t{t_index}.nc'  # ワイルドカードを使用
        matching_files = fnmatch.filter(numerical_files, pattern)
        print(matching_files)

        if matching_files:
            n_file = matching_files[0]  # 最初の一致するファイルを使用
            print(f"Processing: {a_file} and {n_file}")
        
            a_u_bar = read_variable(os.path.join(analytical_dir, a_file), 'u_bar')
            a_theta_bar = read_variable(os.path.join(analytical_dir, a_file), 'theta_bar')
            n_u_bar = read_variable(os.path.join(numerical_dir, n_file), 'u_bar')
            n_theta_bar = read_variable(os.path.join(numerical_dir, n_file), 'theta_bar')

            dif_u_bar = a_u_bar - n_u_bar
            dif_theta_bar = a_theta_bar - n_theta_bar
            #print(dif_theta_bar.shape)
            #input('ss')
            # diffs_u_barとdiffs_theta_barのi番目に格納
            diffs_u_bar[:,:,i] = dif_u_bar
            diffs_theta_bar[:,:,i] = dif_theta_bar

            #print(f"u_bar difference for index {i}: {diffs_u_bar[i, :]}")
            #print(f"theta_bar difference for index {i}: {diffs_theta_bar[i, :]}")
        else:
            print(f"No matching numerical file for {a_file}")

    # 最終的な結果を表示
    print("Final u_bar differences:")
    print(diffs_u_bar)
    print("Final theta_bar differences:")
    print(diffs_theta_bar)

    #u_dif_mean = np.mean(diffs_u_bar,axis=2)
    #theta_dif_mean = np.mean(diffs_theta_bar,axis=2)
    altitude = read_variable(os.path.join(numerical_dir, n_file), 'altitude')
    diffs_u_bar_reshaped = diffs_u_bar.reshape(701, num_files)
    diffs_theta_bar_reshaped = diffs_theta_bar.reshape(701, num_files)

    return diffs_u_bar_reshaped, diffs_theta_bar_reshaped, altitude #return reshaped arrays and t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NetCDF files from analytical and numerical directories.')
    parser.add_argument('analytical_directory', type=str, help='Directory containing analytical NetCDF files')
    parser.add_argument('numerical_directory', type=str, help='Directory containing numerical NetCDF files')

    args = parser.parse_args()

    diffs_u_bar_reshaped, diffs_theta_bar_reshaped, altitude = main(args.analytical_directory, args.numerical_directory)

    #diffs_u_bar_reshaped = diffs_u_bar.reshape(701, 49)
    #diffs_theta_bar_reshaped = diffs_theta_bar.reshape(701, 49)
    t = np.arange(0, 97)

    T, Alt = np.meshgrid(t, altitude)

    # サブプロットの作成
    fig, ax = plt.subplots(1, 2)
    #fig.subplots_adjust(hspace=0)
    # pcolormeshの描画
    c1 = ax[0].pcolor(T, Alt, diffs_u_bar_reshaped, cmap='coolwarm')
    c2 = ax[1].pcolor(T, Alt, diffs_theta_bar_reshaped, cmap='coolwarm')
    
    # 等高線の描画
    #contour1 = ax[0].contour(T, Alt, diffs_u_bar_reshaped, colors='black', linewidths=0.5)
    #contour2 = ax[1].contour(T, Alt, diffs_theta_bar_reshaped, colors='black', linewidths=0.5)
    # カラーバーの追加
    pp = plt.colorbar(c1, ax=ax[0], orientation="vertical")
    cbar = fig.colorbar(c1, ax=ax[1], orientation="vertical")
    # 軸ラベルとタイトルの設定
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('altitude')
    ax[0].set_title("analytical - numerical, u")
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('altitude')
    ax[1].set_title("analytical - numerical, theta")
    
    # プロットの表示
    plt.tight_layout()
    plt.show()
