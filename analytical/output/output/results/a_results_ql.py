import xarray as xr
import os
import glob
import matplotlib.pyplot as plt
import argparse

def plot_netcdf_files(directory):
    # ディレクトリ内のすべてのNetCDFファイルを取得
    file_names = glob.glob(os.path.join(directory, "*.nc"))

    # t0から3時間分ごとのファイルのみを抜き出す
    filtered_file_names = [file_name for file_name in file_names if int(file_name.split('_')[-1].split('t')[1].split('.')[0]) % 10800 == 0]

    # プロットの準備
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # 各ファイルに対して処理を行う
    for file_name in filtered_file_names:
        # データセットを読み込む
        ds = xr.open_dataset(file_name, decode_times=False)
        
        # u_bar と theta_bar をプロット
        t_value = int(file_name.split('_')[-1].split('t')[1].split('.')[0])/ 3600
        ax1.plot(ds['u_bar'].values, ds['altitude'].values, label=str(t_value))
        ax2.plot(ds['theta_bar'].values, ds['altitude'].values, label=str(t_value))

    # プロットのラベルとタイトルを設定
    ax1.set_xlabel('u_bar (m/s)')
    ax1.set_ylabel('Altitude (meters)')
    ax1.set_title('u_bar vs Altitude')
    ax1.grid()
    ax1.legend()

    ax2.set_xlabel('theta_bar (K)')
    ax2.set_ylabel('Altitude (meters)')
    ax2.set_title('theta_bar vs Altitude')
    ax2.grid()
    ax2.legend()
    

    # プロットを表示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    
    args = parser.parse_args()
    
    plot_netcdf_files(args.directory)
