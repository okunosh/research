import argparse
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt

def plot_profiles(directory):
    # ディレクトリ内のすべてのNetCDFファイルを取得
    file_names = glob.glob(os.path.join(directory, "*.nc"))

    filtered_file_names = [file_name for file_name in file_names if int(file_name.split('_')[-1].split('t')[1].split('.')[0]) % 10800 == 0]

    # サブプロットを2個作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # uのプロファイルを描画
    for file_name in filtered_file_names:
        ds = xr.open_dataset(file_name)
        u = ds['u']
        theta = ds['theta']
        altitude = ds['altitude']
        
        ax1.plot(u, altitude, label=f't={ds["time"].values[0]/3600}')
        ax2.plot(theta, altitude, label=f't={ds["time"].values[0]/3600}')

    ax1.set_xlabel(r'$\overline{u}$ [m/s]')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title(r'$\overline{u}$ Profile')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel(r'$\overline{\theta}$ [K]')
    ax2.set_ylabel('Altitude [m]')
    ax2.set_title(r'$\overline{\theta}$ Profile')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot profiles from NetCDF files in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing NetCDF files")
    args = parser.parse_args()
    
    plot_profiles(args.directory)
