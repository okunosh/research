import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path

def plot_ubar_thetabar(
    t_array,             # (nt,) 1d [seconds]
    altitude,            # (nz,) or (nt, nz)
    u_bar,               # (nt, nz)
    theta_bar,
    theta_0,
    g,
    K=None,              # (nt,) optional
    period=24*3600,
    cmap="bwr",
    save_path=None,
    title=None,
    show_colorbar=True,          # 既存フラグ（優先: Falseなら非表示）
    colorbar="shared",           # "shared" | "each" | "none"（デフォルト: shared）
    method="pcolormesh"          # "scatter" or "pcolormesh"
):
    """
    横軸を自動判定：
      - max(t_array) > 3600: 地方時 [hour] に変換
      - それ以外: 秒 [s] のまま

    カラーバー表示:
      - show_colorbar=False → 強制的に非表示
      - colorbar="shared"   → 共通カラーバー（u・θで共通のvmin/vmax、中心0）
      - colorbar="each"     → 各パネルに個別カラーバー（各配列のvmin/vmax、中心0）
      - colorbar="none"     → 非表示
    """
    # ========= 入力整形 =========
    nt, nz = u_bar.shape

    # Z（高度）の2次元展開
    if altitude.ndim == 1 and altitude.shape[0] == nz:
        Z = np.broadcast_to(altitude[None, :], (nt, nz))
    elif altitude.ndim == 2 and altitude.shape == (nt, nz):
        Z = altitude
    else:
        raise ValueError(f"altitude shape {altitude.shape} is incompatible with u_bar {u_bar.shape}")

    # 横軸X（2次元）と凡例用x_1d（1次元）を用意
    t_max = float(np.nanmax(t_array))
    if t_max > 3600.0:
        # 地方時 [hour]
        t_hours = t_array / period * 24.0           # (nt,)
        X = np.broadcast_to(t_hours[:, None], (nt, nz))
        x_1d = t_hours
        x_label = 'Local Time [hour]'
    else:
        # 秒 [s]
        X = np.broadcast_to(t_array[:, None], (nt, nz))
        x_1d = t_array
        x_label = 'Time [s]'

    # ========= カラースケール設計 =========
    # show_colorbar=False のときは colorbar を none 扱いにする
    if not show_colorbar:
        colorbar = "none"

    # 共通スケール用（全体の最小・最大）
    if colorbar == "shared":
        vmin_all = np.nanmin([np.nanmin(u_bar), np.nanmin(theta_bar)])
        vmax_all = np.nanmax([np.nanmax(u_bar), np.nanmax(theta_bar)])
        # vmin==vmax の安全策（定数配列のとき）
        if not np.isfinite(vmin_all) or not np.isfinite(vmax_all) or vmin_all == vmax_all:
            vmin_all, vmax_all = -1.0, 1.0
        norm_shared = TwoSlopeNorm(vmin=vmin_all, vcenter=0.0, vmax=vmax_all)
    else:
        norm_shared = None  # 個別で計算する

    # ========= 描画 =========
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    fontsize_label = 14
    fontsize_title = 18
    fontsize_tick = 12


    arrays = [u_bar, theta_bar]
    titles_u = r'(a) wind speed $\overline{u}$ [m/s]'
    titles_theta =  r"(b) potential temperature anomaly $\overline{\theta}$ [K]"
    titles = [titles_u, titles_theta]
    
    parts_lower = [p.lower() for p in Path(str(save_path)).parts]
    use_b = any("dns" in p for p in parts_lower)
    #print("parts_lower: ", parts_lower)
    #print("use_b: ", use_b)
    if use_b:
        #print("calclate  and plot bouyancy")
        #print("g", g)
        #print("theta_0", theta_0[0])
        b = (float(g) / theta_0[0]) * theta_bar
        titles_b = r"(b) bouyancy anomaly $b$ [m s$^{-2}$]"
        b_label = r"$b$ [m s$^{-2}$"

        arrays[1] = b
        titles[1] = titles_b
    
    artists = []

    for i, (arr, y_title) in enumerate(zip(arrays, titles)):
        ax = axes[i]

        # 個別カラーバー用に個別norm（中心0）を作成／共通なら共通norm
        if colorbar == "each":
            vmin_i = float(np.nanmin(arr))
            vmax_i = float(np.nanmax(arr))
            if not np.isfinite(vmin_i) or not np.isfinite(vmax_i) or vmin_i == vmax_i:
                vmin_i, vmax_i = -1.0, 1.0
            norm_i = TwoSlopeNorm(vmin=vmin_i, vcenter=0.0, vmax=vmax_i)
        else:
            norm_i = norm_shared

        if method == "scatter":
            c = ax.scatter(X.flatten(), Z.flatten(), c=arr.flatten(), cmap=cmap, norm=norm_i, s=20, marker="s")
        elif method == "pcolormesh":
            c = ax.pcolormesh(X, Z, arr, cmap=cmap, norm=norm_i, shading='auto')
        else:
            raise ValueError(f"Unknown method: {method}")

        artists.append(c)

        ax.set_ylabel('Altitude [m]', fontsize=fontsize_label)
        ax.set_title(y_title, fontsize=fontsize_title)
        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.grid()
        ax.minorticks_on()

        # K（最下層など）の重ね描画（任意）
        if K is not None:
            ax2 = ax.twinx()
            ax2.plot(x_1d, K, linewidth=1.8, alpha=0.4,color="black")
            ax2.set_ylabel("K (bottom) [m$^2$/s]", fontsize=fontsize_label)
            ax2.tick_params(axis='y', labelsize=fontsize_tick)
            if np.nanmin(K) < 1.e-2:
                from matplotlib.ticker import StrMethodFormatter
                ax2.set_yscale("log")
                ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:.2e}"))

        # 個別カラーバー
        if colorbar == "each":
            cb = fig.colorbar(c, ax=ax, orientation='vertical', fraction=0.045, pad=0.04)
            cb.ax.tick_params(labelsize=fontsize_tick)

    axes[1].set_xlabel(x_label, fontsize=fontsize_label)

    # 共通カラーバー
    if colorbar == "shared":
        # pcolormesh/scatterのどちらでも使えるように ScalarMappable を用意
        sm = ScalarMappable(norm=norm_shared, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.045, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize_tick)

    if title:
        fig.suptitle(title, fontsize=fontsize_title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"Fig is saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

        # time & vals


if __name__ == "__main__":
    import argparse
    import os
    from visualization.common.io_utils import load_all_data, read_global_attr_values, stack_by_variable, convert_to_standard_shapes
    
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    args = parser.parse_args()

    varnames = ["u_bar", "theta_bar", "altitude", "time", "K", "theta_0"]
    attrs_names = ["g"]
    
    data_list = load_all_data(args.directory, varnames)
    stacked = stack_by_variable(data_list, varnames)
    reshaped_stacked = convert_to_standard_shapes(stacked)

    attr_dic = read_global_attr_values(args.directory, attrs_names)

    """
    print(reshaped_stacked["time"].shape) #(nt,)
    print(reshaped_stacked["altitude"].shape) #(nz,)
    print(reshaped_stacked["u_bar"].shape) #(nt, nz)
    print(reshaped_stacked["K"].shape)#(nt,) 
    """
    
    # 可視化
    for method in ["scatter", "pcolormesh"]:
        save_filename = f"time_alt_map_{method}.png"
        plot_ubar_thetabar(
            t_array=reshaped_stacked["time"],
            altitude=reshaped_stacked["altitude"],
            u_bar=reshaped_stacked["u_bar"],
            theta_bar=reshaped_stacked["theta_bar"],
            K = reshaped_stacked["K"],
            theta_0 = reshaped_stacked["theta_0"],
            g=attr_dic["g"],
            period=24*3600,
            save_path=os.path.join(args.directory, save_filename),
            method=method,
            title=None,
            colorbar="each"
        )

    save_filename2="timeseriesEachAltitude.png"
    plot_timeseries_u_b(
        reshaped_stacked,
        alt_arr = (0.07, 0.3, 0.6, 0.9),  # [m] 近傍の既存レベルをそのまま使用（補間なし）
        g = attr_dic["g"], 
        out_path=os.path.join(args.directory, save_filename2),
        show=False
    )
