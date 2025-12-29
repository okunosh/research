import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import Optional, Mapping


from visualization.last_day_plotters import _to_lt_hours, _centers_to_edges

#CONSTANTs
R = 191
Cp =  735
KAPPA = R / Cp

#pressure
import numpy as np

def calc_pressure(reshaped_stacked, p_surf, g):
    z = np.asarray(reshaped_stacked["altitude"], dtype=float)
    gamma = np.asarray(reshaped_stacked["gamma"], dtype=float)[0]
    theta_0 = np.asarray(reshaped_stacked["theta_0"], dtype=float)
    th0 = float(np.ravel(theta_0)[0])

    theta = th0 + gamma * z + reshaped_stacked["theta_bar"]

    inv_th = 1.0 / theta
    I = np.zeros_like(theta, dtype=float)

    if z.ndim == 1:
        dz = np.diff(z)
        I[1:] = np.cumsum(0.5 * (inv_th[:-1] + inv_th[1:]) * dz)
    else:
        dz = z[:, 1:] - z[:, :-1]
        I[:, 1:] = np.cumsum(0.5 * (inv_th[:, :-1] + inv_th[:, 1:]) * dz, axis=1)

    alpha = (float(KAPPA) * float(g)) / float(R)  # κ g / R
    pi = 1.0 - alpha * I  # p_ref = p_surf として π(z0)=1

    p0 = np.asarray(p_surf, dtype=float)
    if p0.ndim == 0:
        p = p0 * (pi ** (1.0 / float(KAPPA)))
    else:
        p = p0[:, None] * (pi ** (1.0 / float(KAPPA)))
    return p


def pressure_without_theta_bar(reshaped_stacked, p_surf, g):
    z = reshaped_stacked["altitude"]
    th0 = reshaped_stacked["theta_0"][0]
    gamma = reshaped_stacked["gamma"]
    
    # p_ref = p_surf として線形温位の閉形式
    I0 = (1.0/gamma) * np.log((th0 + gamma*z) / th0)
    pi = 1.0 - (KAPPA*g/R) * I0
    return p_surf * (pi**(1.0/KAPPA))

"""
def calc_pressure(reshaped_stacked, p_surf, g):

    

    theta_bar = reshaped_stacked["theta_bar"]
    z = reshaped_stacked["altitude"]
    gamma = reshaped_stacked["gamma"]
    theta_0 = reshaped_stacked["theta_0"]
    th0 = theta_0[0]
    g = float(g)
    
    
    theta = th0 + gamma * z #+ theta_bar
    print("th_min: ", np.nanmin(theta))
    print("th_max: ", np.nanmax(theta))

    coeff = KAPPA * g * p_surf**KAPPA /R / gamma
    p = (p_surf**KAPPA - coeff * np.log(theta / th0) )**(1/KAPPA)
    
    return p
"""
def calc_temperature(reshaped_stacked, p, p_surf):

    z = reshaped_stacked["altitude"]
    gamma = reshaped_stacked["gamma"][0]
    theta_bar = reshaped_stacked["theta_bar"]
    theta_0 = reshaped_stacked["theta_0"]
    th0 = theta_0[0]

    theta = th0 + gamma * z + theta_bar
    t = theta * (p_surf / p)**KAPPA
    
    return t

def calc_density(reshaped_stacked, p, temperature):
    T = temperature
    density = p / T / R

    return density

def calc_momentum(reshaped_stacked, rho):

    u = reshaped_stacked["u_bar"]
    mom = u * rho
    return mom

#Plot
from pathlib import Path
from typing import Optional, Mapping
import numpy as np
import matplotlib.pyplot as plt
from visualization.last_day_plotters import _to_lt_hours, _centers_to_edges

from pathlib import Path
from typing import Optional, Mapping
import numpy as np
import matplotlib.pyplot as plt
from visualization.last_day_plotters import _to_lt_hours, _centers_to_edges
from datetime import timedelta

SEC_PER_HOUR = float(timedelta(hours=1).total_seconds())

def plot_last_day_generic(
    reshaped_stacked: Mapping[str, np.ndarray],
    var_name: str,
    fields: Mapping[str, np.ndarray],
    *,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    units: Optional[str] = None,
    save_path: Optional[str] = None,
    invert_cbar: Optional[bool] = None,
) -> None:
    """
    計算済み2D配列を可視化（横=LT[hour],縦=z）。“最後の1日”だけ描画。
    - A がすでに最後の1日（nt_A == nt_last）に切り出されている場合、追加のマスクは行わない。
    - A が全期間（nt_A == nt_full）の場合のみ最後の1日マスクを適用する。
    """
    # 既定の表示設定
    cmap_map  = {"p":"Blues", "T":"Reds", "rho":"viridis", "u_rho":"cividis", "theta":"Greens"}
    title_map = {"p":"Pressure", "T":"Temperature", "rho":"Density", "u_rho":"Momentum (u·ρ)", "theta":"Potential temperature"}
    unit_map  = {"p":"Pa", "T":"K", "rho":"kg m$^{-3}$", "u_rho":"kg m$^{-2}$ s$^{-1}$", "theta":"K"}

    _cmap  = cmap  or cmap_map.get(var_name, "viridis")
    _title = title or title_map.get(var_name, var_name)
    _units = units or unit_map.get(var_name, "")
    _invert = invert_cbar if invert_cbar is not None else (var_name == "p")

    # 入力取り出し
    t_full = np.asarray(reshaped_stacked["time"], dtype=np.float64)   # (nt_full,)
    z_raw  = np.asarray(reshaped_stacked["altitude"])                 # (nt_full,nz) or (nz,)
    # 可視化対象 A を取得（u_rho はそのまま fields から受け取る運用で）
    if var_name in fields:
        A = np.asarray(fields[var_name])
    elif var_name == "u_rho" and "u_rho" in fields:
        A = np.asarray(fields["u_rho"])
    else:
        # フォールバック：そのままキーアクセス
        A = np.asarray(fields[var_name])  # ここで KeyError が出れば呼び出し側で修正してください

    if A.ndim != 2:
        raise ValueError(f"{var_name} は (nt, nz) の2次元配列が必要です（shape={A.shape}）。")

    nt_full = t_full.size
    nt_A, nz_A = A.shape

    # 最後の1日マスク（t_full 基準）
    period = 24.0 * SEC_PER_HOUR
    mask_last = (t_full > (float(t_full[-1]) - period))
    nt_last = int(mask_last.sum())

    # --- 時間合わせのロジック ---
    # ケース1) A が全期間 → 最後の1日のみ切り出し
    if nt_A == nt_full:
        t_sel = t_full[mask_last]                # (nt_last,)
        A_sel = A[mask_last, :]                  # (nt_last, nz)
    # ケース2) A がすでに最後の1日（推定） → そのまま使う
    elif nt_A == nt_last:
        t_sel = t_full[mask_last]                # (nt_last,)
        A_sel = A                                # (nt_last, nz)
    else:
        # ケース3) それ以外（不一致） → 具体的にエラーを出す
        raise ValueError(
            f"時間次元が合いません: A.shape[0]={nt_A}, 全期間={nt_full}, 最後の1日={nt_last}。\n"
            "A を全期間配列か最後の1日配列のいずれかに揃えてください。"
        )

    # z を 1D に整形（(nt,nz)なら最後の1日部分の列平均）
    if z_raw.ndim == 2 and z_raw.shape[0] == nt_full and z_raw.shape[1] == nz_A:
        z_centers = np.nanmean(z_raw[mask_last, :], axis=0)   # (nz,)
    elif z_raw.ndim == 2 and z_raw.shape[0] == nt_A and z_raw.shape[1] == nz_A:
        z_centers = np.nanmean(z_raw, axis=0)                 # (nz,)
    elif z_raw.ndim == 1 and z_raw.size == nz_A:
        z_centers = z_raw                                     # (nz,)
    else:
        raise ValueError(f"altitude 形状不整合: {z_raw.shape} と A.shape={A.shape}")

    # x軸=LT[hour] に変換 → 昇順ソート → エッジ化
    t_hours   = t_sel / SEC_PER_HOUR
    t_centers = _to_lt_hours(t_hours)
    sort_idx  = np.argsort(t_centers)
    t_edges   = _centers_to_edges(t_centers[sort_idx])
    z_edges   = _centers_to_edges(z_centers)

    # 描画
    A_sorted = A_sel[sort_idx, :].T
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    m  = ax.pcolormesh(t_edges, z_edges, A_sorted, cmap=_cmap, shading="auto")
    cb = fig.colorbar(m, ax=ax, fraction=0.045, pad=0.04)
    if _units:
        cb.set_label(_units)
    if _invert:
        cb.ax.invert_yaxis()

    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(_title)
    ax.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)

    """
def plot_last_day_generic(
    reshaped_stacked: Mapping[str, np.ndarray],
    var_name: str,
    fields: Mapping[str, np.ndarray],
    *,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    units: Optional[str] = None,
    save_path: Optional[str] = None,
    invert_cbar: Optional[bool] = None,   # ← 追加：Noneなら"p"のとき自動で反転
) -> None:
    cmap_map  = {"p":"Blues", "T":"Reds", "rho":"viridis", "u_rho":"cividis"}
    title_map = {"p":"Pressure", "T":"Temperature", "rho":"Density", "u_rho":"Momentum (u·ρ)"}
    unit_map  = {"p":"Pa", "T":"K", "rho":"kg m$^{-3}$", "u_rho":"kg m$^{-2}$ s$^{-1}$"}

    _cmap  = cmap  or cmap_map.get(var_name, "viridis")
    _title = title or title_map.get(var_name, var_name)
    _units = units or unit_map.get(var_name, "")
    _invert = invert_cbar if invert_cbar is not None else (var_name == "p")  # ← ここ

    t = np.asarray(reshaped_stacked["time"])
    z = np.asarray(reshaped_stacked["altitude"])

    if var_name == "u_rho":
        rho = np.asarray(fields["rho"])
        u   = np.asarray(reshaped_stacked["u_bar"])
        if rho.shape != u.shape:
            raise ValueError(f"u_bar{u.shape} と rho{rho.shape} の形状不一致です。")
        A = u * rho
    else:
        if var_name not in fields:
            raise KeyError(f"fields['{var_name}'] がありません。")
        A = np.asarray(fields[var_name])

    if A.ndim != 2:
        raise ValueError(f"{var_name} は (nt, nz) の2次元配列が必要です（shape={A.shape}）。")

    period = 86400.0
    t_end = float(t[-1])
    mask = (t > (t_end - period))
    t_sel = t[mask]
    A_sel = A[mask, :]

    if z.ndim == 2 and z.shape[0] == t.shape[0]:
        z_centers = np.nanmean(z[mask, :], axis=0)
    elif z.ndim == 1 and z.size == A.shape[1]:
        z_centers = z
    else:
        raise ValueError("altitude は (nt, nz) または (nz,) で、対象配列と整合している必要があります。")

    t_hours   = t_sel / 3600.0
    t_centers = _to_lt_hours(t_hours)
    sort_idx  = np.argsort(t_centers)
    t_edges   = _centers_to_edges(t_centers[sort_idx])
    z_edges   = _centers_to_edges(z_centers)

    A_sorted = A_sel[sort_idx, :].T
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    m  = ax.pcolormesh(t_edges, z_edges, A_sorted, cmap=_cmap, shading="auto")
    cb = fig.colorbar(m, ax=ax, fraction=0.045, pad=0.04)
    if _units:
        cb.set_label(_units)
    if _invert:
        cb.ax.invert_yaxis()

    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(_title)
    ax.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)
"""

    
def plot_pressure_t_alt_map(reshaped_stacked, p, cmap="Blues", save_path=None):
    """
    気圧 p (shape=(nt, nz)) を、最後の1日だけ切り出して pcolormesh 描画。
    x軸は Local Time[hour]（0–24）に整形し、セルエッジは共有ヘルパを利用。
    """
    t = reshaped_stacked["time"]        # (nt,)  [sec] もしくは [hour] でもよいが、ここでは秒想定
    z = reshaped_stacked["altitude"]    # (nt, nz) or (nz,)
    period = 86400.0

    # --- 直近1日のマスク ---
    t_end = float(t[-1])
    mask = (t > (t_end - period))
    t_sel = t[mask]                     # (nt_last,)

    # --- 形状をそろえる ---
    if p.ndim != 2:
        raise ValueError("p must be 2D with shape (nt, nz).")
    p_sel = p[mask, :]                  # (nt_last, nz)

    # altitude: (nt,nz) の場合はマスクして列平均で (nz,) に代表化。 (nz,) の場合はそのまま。
    if z.ndim == 2 and z.shape[0] == t.shape[0]:
        z_sel = z[mask, :]              # (nt_last, nz)
        z_centers = np.nanmean(z_sel, axis=0)   # (nz,)
    elif z.ndim == 1 and z.size == p.shape[1]:
        z_centers = z                    # (nz,)
    else:
        raise ValueError("altitude must be (nt, nz) or (nz,) and consistent with p.")

    # --- x軸: Local Time[hour] に整形し、0跨ぎを避けるために昇順ソート ---
    # t は秒想定。hourへ換算して共有関数で 0–24 へ正規化
    t_hours = t_sel / 3600.0
    t_centers_lt = _to_lt_hours(t_hours)          # (nt_last,)
    sort_idx = np.argsort(t_centers_lt)
    t_centers = t_centers_lt[sort_idx]            # 昇順（中心）
    t_edges   = _centers_to_edges(t_centers)      # (nt_last+1,)

    # --- y軸: z も中心→エッジへ ---
    z_edges = _centers_to_edges(z_centers)        # (nz+1,)

    # --- 値を (y, x) = (nz, nt_last) にして pcolormesh ---
    p_sorted = p_sel[sort_idx, :].T               # (nz, nt_last)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    m = ax.pcolormesh(t_edges, z_edges, p_sorted, cmap=cmap, shading="auto")
    cb = fig.colorbar(m, ax=ax, orientation="vertical", fraction=0.045, pad=0.04)
    cb.ax.invert_yaxis()

    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Pressure [hPa]")
    ax.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


def plot_temperature_t_alt_map(reshaped_stacked, temperature, cmap="Reds", save_path=None):
    """
    温度 T (shape=(nt, nz)) を、最後の1日だけ切り出して pcolormesh 描画。
    """
    t = reshaped_stacked["time"]        # (nt,)
    z = reshaped_stacked["altitude"]    # (nt, nz) or (nz,)
    period = 86400.0

    # --- 直近1日のマスク ---
    t_end = float(t[-1])
    mask = (t > (t_end - period))
    t_sel = t[mask]

    if temperature.ndim != 2:
        raise ValueError("temperature must be 2D with shape (nt, nz).")
    T_sel = temperature[mask, :]        # (nt_last, nz)

    if z.ndim == 2 and z.shape[0] == t.shape[0]:
        z_sel = z[mask, :]
        z_centers = np.nanmean(z_sel, axis=0)   # (nz,)
    elif z.ndim == 1 and z.size == temperature.shape[1]:
        z_centers = z
    else:
        raise ValueError("altitude must be (nt, nz) or (nz,) and consistent with temperature.")

    t_hours = t_sel / 3600.0
    t_centers_lt = _to_lt_hours(t_hours)
    sort_idx = np.argsort(t_centers_lt)
    t_centers = t_centers_lt[sort_idx]
    t_edges   = _centers_to_edges(t_centers)
    z_edges   = _centers_to_edges(z_centers)

    T_sorted = T_sel[sort_idx, :].T

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    m = ax.pcolormesh(t_edges, z_edges, T_sorted, cmap=cmap, shading="auto")
    cb = fig.colorbar(m, ax=ax, orientation="vertical", fraction=0.045, pad=0.04)

    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Temperature [K]")
    ax.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


def plot_potential_temperature_t_alt_map(reshaped_stacked, cmap="Greens", save_path=None):
    """
    温位 theta (shape=(nt, nz)) を、最後の1日だけ切り出して pcolormesh 描画。
    x軸は Local Time [hour]（024）に整形し、セルエッジは共有ヘルパを利用。
    """
    t = reshaped_stacked["time"]        # (nt,) [sec]
    z = reshaped_stacked["altitude"]    # (nt, nz) or (nz,)

    theta_bar = reshaped_stacked["theta_bar"]
    z = reshaped_stacked["altitude"]
    gamma = reshaped_stacked["gamma"]
    theta_0 = reshaped_stacked["theta_0"]
    th0 = theta_0[0]
    
    
    theta = th0 + gamma * z + theta_bar
    period = 86400.

    # --- 直近1日のマスク ---
    t_end = float(t[-1])
    mask = (t > (t_end - period))
    t_sel = t[mask]                     # (nt_last,)

    if theta.ndim != 2:
        raise ValueError("theta must be 2D with shape (nt, nz).")
    th_sel = theta[mask, :]             # (nt_last, nz)

    # altitude: (nt,nz) の場合はマスクして列平均で (nz,) に代表化。 (nz,) の場合はそのまま。
    if z.ndim == 2 and z.shape[0] == t.shape[0]:
        z_sel = z[mask, :]                           # (nt_last, nz)
        z_centers = np.nanmean(z_sel, axis=0)        # (nz,)
    elif z.ndim == 1 and z.size == theta.shape[1]:
        z_centers = z                                 # (nz,)
    else:
        raise ValueError("altitude must be (nt, nz) or (nz,) and consistent with theta.")

    # --- x軸: Local Time [hour] に整形し、0跨ぎ回避のため昇順ソート ---
    t_hours = t_sel / 3600.0
    t_centers_lt = _to_lt_hours(t_hours)             # (nt_last,)
    sort_idx = np.argsort(t_centers_lt)
    t_centers = t_centers_lt[sort_idx]               # 昇順（中心）
    t_edges   = _centers_to_edges(t_centers)         # (nt_last+1,)

    # --- y軸: z も中心→エッジへ ---
    z_edges = _centers_to_edges(z_centers)           # (nz+1,)

    # --- 値を (y, x) = (nz, nt_last) にして pcolormesh ---
    th_sorted = th_sel[sort_idx, :].T                # (nz, nt_last)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    m = ax.pcolormesh(t_edges, z_edges, th_sorted, cmap=cmap, shading="auto")
    cb = fig.colorbar(m, ax=ax, orientation="vertical", fraction=0.045, pad=0.04)

    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Potential temperature [K]")
    ax.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


    

if __name__ == "__main__":
    import argparse
    import os
    from visualization.common.io_utils import load_all_data, read_global_attr_values, stack_by_variable, convert_to_standard_shapes
    
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    args = parser.parse_args()

    varnames = ["u_bar", "theta_bar", "altitude", "time", "K", "theta_0", "gamma"]
    attrs_names = ["g"]
    
    data_list = load_all_data(args.directory, varnames)
    stacked = stack_by_variable(data_list, varnames)
    reshaped_stacked = convert_to_standard_shapes(stacked)

    attr_dic = read_global_attr_values(args.directory, attrs_names)


    p_surf = 610 #Pa
    p = calc_pressure(reshaped_stacked, p_surf, attr_dic["g"])
    temperature = calc_temperature(reshaped_stacked, p, p_surf)
    rho = calc_density(reshaped_stacked, p, temperature)
    momentum = calc_momentum(reshaped_stacked, rho)

    p_wo_pt = pressure_without_theta_bar(reshaped_stacked, p_surf, float(attr_dic["g"]))

    print("p - p w/o pt", np.nanmax(p - p_wo_pt))
    fields = {"p": p, "T": temperature, "rho": rho, "u_rho": momentum}
        
    save_filename_p = "test3_pressure_map"
    save_filename_T = "test3_temperature_map"
    save_filename_rho = "test3_density_map"
    save_filename_mom = "test3_momentum_map"
    
    
    plot_last_day_generic(reshaped_stacked, "p",    fields, save_path=os.path.join(args.directory, save_filename_p))
    plot_last_day_generic(reshaped_stacked, "T",    fields, save_path=os.path.join(args.directory, save_filename_T))
    plot_last_day_generic(reshaped_stacked, "rho",  fields, save_path=os.path.join(args.directory, save_filename_rho))
    plot_last_day_generic(reshaped_stacked, "u_rho",fields, save_path=os.path.join(args.directory, save_filename_mom))

    """
    save_filename = "test_pressure_map"
    plot_pressure_t_alt_map(reshaped_stacked, p,
                            save_path=os.path.join(args.directory, save_filename)
                            )
    save_filename2 = "test_temperature_map"
    plot_temperature_t_alt_map(reshaped_stacked, temperature,
                            save_path=os.path.join(args.directory, save_filename2)
                            )

    save_filename3 = "test_potential_temperature"
    plot_potential_temperature_t_alt_map(reshaped_stacked,
                                         save_path=os.path.join(args.directory, save_filename3)
                                         )
    """
