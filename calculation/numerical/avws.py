import sys
sys.path.append("../")
import ql_plot as ql
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    
    args = parser.parse_args()
    #ql.process_netcdf_directory(args.directory, confirm=True)

    altitude, t, u_bars, theta_bars = ql.load_datasets(args.directory)

    print(altitude.shape, t.shape, u_bars.shape, theta_bars.shape)
    
    lists = os.listdir(args.directory)
    nc_files = list(filter(lambda file: file.endswith('.nc'), lists))
    print(nc_files[0])
    ds = xr.open_dataset(args.directory+"/"+nc_files[0])
    #print(ds.variables)

    avws_height = np.zeros(len(t))*np.nan
    avws_height2 = np.zeros(len(t))*np.nan
    avws_height3 = np.zeros(len(t))*np.nan
    avws_height4 = np.zeros(len(t))*np.nan

    b1, t1 = 0, 3
    b2, t2 = 10, 20
    b3, t3 = 29, 57
    b4, t4 = 58, 86
    for i in range(len(t)):
        avws_height[i] = u_bars[b1:t1,i].mean()
        avws_height2[i] = u_bars[b2:t2,i].mean()
        avws_height3[i] = u_bars[b3:t3,i].mean()
        avws_height4[i] = u_bars[b4:t4,i].mean()

    plt.plot(t, avws_height, label="{:.2f}m-{:.2f}m".format(altitude[b1:t1][0], altitude[b1:t1][-1]))
    plt.plot(t, avws_height2, label="{:.2f}m-{:2f}m".format(altitude[b2:t2][0], altitude[b2:t2][-1]))
    plt.plot(t, avws_height3, label="{:.2f}m-{:.2f}m".format(altitude[b3:t3][0], altitude[b3:t3][-1]))
    plt.plot(t, avws_height4, label="{:.2f}m-{:.2f}m".format(altitude[b4:t4][0], altitude[b4:t4][-1]))
    plt.xticks(np.arange(0,len(t),24))
    plt.ylabel('average wind speed')
    plt.xlabel('t')
    plt.title(args.directory)
    plt.grid()
    plt.legend()
    plt.minorticks_on()
    plt.show()
    
