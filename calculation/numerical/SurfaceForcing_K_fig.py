import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# デフォルトのフォントサイズと線の太さを設定
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 2
fontsize_legend=12

SF_nc = "surface_forcing/test_surface_forcing.nc"
K_nc = "K/K_261_25.nc"

def load_data(SF_nc, K_nc):
    SF_ds = xr.open_dataset(SF_nc)
    K_ds = xr.open_dataset(K_nc)
    
    SF_t_val = SF_ds.time.values
    SF_val = SF_ds.theta_0.values
    
    K_t_val = K_ds.time.values
    K_val = K_ds.K.values[0,:]
    return SF_t_val, SF_val, K_t_val, K_val

def make_fig(SF_t_val, SF_val, K_t_val, K_val, save_name, confirm):
    
    fontsize_title = 18
    fontsize_label = 14
    fontsize_tick = 12
    linewidth = 2

    color_SF = '#A00000'
    color_K = "black"

    linewidth_SF = 2.5
    linewidth_K = 1.8
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(SF_t_val, SF_val, color=color_SF, linestyle=(0, (3, 1, 1, 1)), linewidth=2.5, label=r"$\overline{\theta}$(0,t)")
    ax.set_ylabel(r"temperature anomaly [K]", color=color_SF)

    ax2 = ax.twinx()
    ax2.plot(K_t_val, K_val, color=color_K, linewidth=1.8, alpha=0.6, label="K")
    ax2.set_ylabel("K [m$^2$/s]", color=color_K, fontsize=fontsize_label)

    ax.grid()
    ax.set_xlabel('t [hour]')
    ax.set_xticks(np.arange(0, 86401, 3600*5))
    ax.set_xticklabels(np.arange(0, 25, 5))
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, frameon=True, fontsize=fontsize_legend)
    ax.set_title(r"Diurnal distribution of $\overline{\theta}$(0,t) and K")
    if confirm==True:
        fig.savefig(save_name+".png" ,bbox_inches="tight", pad_inches=0.05, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    SF_t_val, SF_val, K_t_val, K_val = load_data(SF_nc, K_nc)

    save_name = "SurfaceForcing_K_sub_80"

    confirm=True #save fig
    make_fig(SF_t_val, SF_val, K_t_val, K_val, save_name, confirm)
