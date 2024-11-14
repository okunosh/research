import os
import re
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

class NetCDFProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.netcdf_files = self.extract_netcdf_files()

    def extract_netcdf_files(self):
        """extract all netcdf in the directory"""
        files = [f for f in os.listdir(self.directory) if f.endswith('.nc')]
        sorted_files = sorted(files, key=lambda x: int(x.split('_t')[1].split('.')[0]))
        return files

    def filter_files_by_time(self):
        """extract files in each 3 hours"""
        filtered_files = []
        for file in self.netcdf_files:
            match = re.search(r't(\d+)\.nc', file)
            if match:
                time_in_seconds = int(match.group(1))
                if time_in_seconds % 10800 == 0:  # 10800sec = 3 hours
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

def process_netcdf_directory(directory):
    processor = NetCDFProcessor(directory)
    processor.plot_variables()

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    
    args = parser.parse_args()
    
    process_netcdf_directory(args.directory)
