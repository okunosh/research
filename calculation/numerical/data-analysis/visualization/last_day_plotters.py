# visualization/last_day_plotters.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from typing import Optional, Tuple
from common.extract_extremas import LastDaySlice, ExtremaResult

# --------- 内部ユーティリティ ---------
LT_PERIOD = 24.0
facecolor="#E0E0E0"

def _to_lt_hours(t_hours: np.ndarray) -> np.ndarray:
    """
    絶対時刻[hour] → Local Time [0..24] に正規化。
    pcolormesh/scatterのX座標として使用。
    """
    t = np.asarray(t_hours, dtype=float).ravel()
    t_lt = np.mod(t, LT_PERIOD)

    # 右端が 0 に巻き込まれてしまうケースへの微調整（ほぼ24なら24に寄せる）
    if np.isclose(t_lt[-1], 0.0, atol=1e-9) and (t[-1] - t[0]) >= 23.9:
        t_lt[-1] = LT_PERIOD
    return t_lt

def _x_on_axis(t_val: float, use_lt: bool) -> float:
    return (_to_lt_hours([t_val])[0]) if use_lt else float(t_val)



def _make_diverging_norm(vmin: float, vmax: float) -> Tuple[object, float, float]:
    """
    TwoSlopeNorm の安全版（vmin==vmax, 逆順, 非有限値を吸収）。
    返り値: (norm, vmin_used, vmax_used)
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return Normalize(vmin=-1.0, vmax=1.0), -1.0, 1.0
    if vmin == vmax:
        vm = max(1.0, abs(vmin))
        return Normalize(vmin=-vm, vmax=vm), -vm, vm
    lo, hi = (vmin, vmax) if vmin < vmax else (vmax, vmin)
    try:
        return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi), lo, hi
    except Exception:
        span = hi - lo
        if span <= 1e-9:
            return Normalize(vmin=lo - 1.0, vmax=lo + 1.0), lo - 1.0, lo + 1.0
        return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi), lo, hi

def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """
    1D中心点配列 -> 境界配列（len+1）。端は外挿。
    """
    c = np.asarray(centers, dtype=float).ravel()
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)
    if c.size < 1:
        raise ValueError("centers must have length >= 1")
    if c.size == 2:
        mid = 0.5 * (c[0] + c[1])
        left = c[0] - (mid - c[0])
        right = c[1] + (c[1] - mid)
        return np.array([left, mid, right], dtype=float)
    mid = 0.5 * (c[1:] + c[:-1])
    left = c[0] - (mid[0] - c[0])
    right = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[left], mid, [right]])

# --------- 散布図 ---------
def plot_last_day_scatter(
    out_path: Optional[str],
    last: LastDaySlice,
    extrema_u: Optional[ExtremaResult] = None,
    extrema_theta: Optional[ExtremaResult] = None,
    *,
    title_u: str = r'(a) wind speed $\overline{u}$ [m/s]',
    title_theta: str = r"potential temperature anomaly $\overline{\theta}$ [K]",
    cmap: str = "bwr",
    point_size: float = 12.0,
    dpi: int = 300,
    use_lt_axis: bool = True,
    show: bool = False,
) -> None:
    """
    最後の1日分を散布図で 2行1列（上=ū, 下=θ̄）に描画。
    K があれば両段右軸に重畳。
    """
    # 共通カラースケール（0中心）
    vmin_all = np.nanmin([np.nanmin(last.u_day), np.nanmin(last.theta_day)])
    vmax_all = np.nanmax([np.nanmax(last.u_day), np.nanmax(last.theta_day)])
    norm, _, _ = _make_diverging_norm(vmin_all, vmax_all)

    fig, (ax_u, ax_th) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # プロット
    #T = np.broadcast_to(last.t_day[None, :], last.u_day.shape)  # [nz, nt]

    # X座標（LT or 絶対時刻）
    x_vals = _to_lt_hours(last.t_day) if use_lt_axis else last.t_day
    T = np.broadcast_to(x_vals[None, :], last.u_day.shape)

    sc0 = ax_u.scatter(T, last.z_day, c=last.u_day, s=point_size, cmap=cmap, norm=norm)
    sc1 = ax_th.scatter(T, last.z_day, c=last.theta_day, s=point_size, cmap=cmap, norm=norm)

    # カラーバー（下段に紐づけ）
    cbar = fig.colorbar(sc1, ax=(ax_u, ax_th), orientation="vertical", fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    def _x_on_axis(t_val: float) -> float:
        return _to_lt_hours(np.array([t_val]))[0] if use_lt_axis else float(t_val)
    # 極値
    star_kw = dict(s=180, marker="*", linewidths=1.8,
                   facecolor="white", zorder=6)

    # 軸に合わせた x 座標変換関数
    if extrema_u is not None:
        x_umax = _x_on_axis(extrema_u.t_at_max)
        x_umin = _x_on_axis(extrema_u.t_at_min)
        ax_u.scatter(x_umax, extrema_u.z_at_max, edgecolors="darkred", **star_kw,
                     label=f"Upslope max: {extrema_u.max_value:.2f} m/s")
        ax_u.scatter(x_umin, extrema_u.z_at_min, edgecolors="darkblue", **star_kw,
                     label=f"Downslope min: {extrema_u.min_value:.2f} m/s")
        ax_u.legend(loc="upper left", fontsize=11, frameon=True)
        
    if extrema_theta is not None:
        x_thmax = _x_on_axis(extrema_theta.t_at_max)
        x_thmin = _x_on_axis(extrema_theta.t_at_min)
        ax_th.scatter(x_thmax, extrema_theta.z_at_max, edgecolors="darkred", **star_kw,
                      label=f"Max: {extrema_theta.max_value:.2f} K")
        ax_th.scatter(x_thmin, extrema_theta.z_at_min, edgecolors="darkblue", **star_kw,
                      label=f"Min: {extrema_theta.min_value:.2f} K")
        ax_th.legend(loc="upper left", fontsize=11, frameon=True)    

    # 体裁
    for ax in (ax_u, ax_th):
        ax.set_xlabel("Local time [hour]" if use_lt_axis else "Time [hour]", fontsize=13)
        ax.set_facecolor(facecolor)
        if use_lt_axis:
            #ax.set_xlim(0.0, 24.0)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_ylabel("Altitude [m]", fontsize=13)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=12)
        if last.K_day is not None:
            ax2 = ax.twinx()
            x_k = x_vals  # Kも同じXで重ねる
            ax2.plot(x_k, last.K_day, linewidth=2.0, alpha=0.8, color="black")
            ax2.set_ylabel(r"K [m$^2$/s]", fontsize=12)
            ax2.tick_params(labelsize=12)

    ax_u.set_title(title_u, fontsize=16)
    ax_th.set_title(title_theta, fontsize=16)

    # 出力
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    if show or not out_path:
        plt.show()
    plt.close(fig)

# --------- pcolormesh ---------
def plot_last_day_pcolormesh(
    out_path: Optional[str],
    last: LastDaySlice,
    extrema_u: Optional[ExtremaResult] = None,
    extrema_theta: Optional[ExtremaResult] = None,
    *,
    title_u: str = r'(a) wind speed $\overline{u}$ [m/s]',
    title_theta: str = r"(b) potential temperature anomaly $\overline{\theta}$ [K]",
    cmap: str = "bwr",
    dpi: int = 300,
    use_lt_axis=True,
    show: bool = False,
) -> None:
    """
    最後の1日分を pcolormesh で 2行1列（上=ū, 下=θ̄）に描画。
    pcolormesh はセル境界座標が必要なので、t と z の中心→境界へ変換。
    K があれば両段右軸に重畳。
    """
    # 共通カラースケール（0中心）
    vmin_all = np.nanmin([np.nanmin(last.u_day), np.nanmin(last.theta_day)])
    vmax_all = np.nanmax([np.nanmax(last.u_day), np.nanmax(last.theta_day)])
    norm, _, _ = _make_diverging_norm(vmin_all, vmax_all)

    # 時間は昇順に（pcolormesh で綺麗に敷くため）
    sort_idx_abs = np.argsort(last.t_day)
    t_centers_abs = last.t_day[sort_idx_abs]

    if use_lt_axis:
        # LT 化してから再ソート（0 跨ぎ対策）
        t_centers_lt = _to_lt_hours(t_centers_abs)
        sort_idx_lt = np.argsort(t_centers_lt)
        t_centers = t_centers_lt[sort_idx_lt]
        final_col_idx = sort_idx_abs[sort_idx_lt]
    else:
        t_centers = t_centers_abs
        final_col_idx = sort_idx_abs

    t_edges = _centers_to_edges(t_centers)

    # z は列ごとの差が小さい前提で列平均を代表化
    z_centers_1d = np.nanmean(last.z_day[:, final_col_idx], axis=1)  # [nz]
    z_edges = _centers_to_edges(z_centers_1d)

    # u, theta を時間順に並べ替え
    u_sorted = last.u_day[:, final_col_idx]
    th_sorted = last.theta_day[:, final_col_idx]

    # K も同じ並びに（右軸用）
    K_sorted = (last.K_day[final_col_idx] if last.K_day is not None else None)

    fig, (ax_u, ax_th) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    pm0 = ax_u.pcolormesh(t_edges, z_edges, u_sorted, shading="auto", cmap=cmap, norm=norm)
    pm1 = ax_th.pcolormesh(t_edges, z_edges, th_sorted, shading="auto", cmap=cmap, norm=norm)

    # カラーバー（下段に紐づけ）
    cbar = fig.colorbar(pm1, ax=(ax_u, ax_th), orientation="vertical", fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    # 極値重畳（任意） X は軸に合わせて（LT/絶対時刻）
    star_kw = dict(s=150, marker="*", linewidths=1.8, facecolor="white")
    if extrema_u is not None:
        x_umax = (_to_lt_hours([extrema_u.t_at_max])[0] if use_lt_axis else extrema_u.t_at_max)
        x_umin = (_to_lt_hours([extrema_u.t_at_min])[0] if use_lt_axis else extrema_u.t_at_min)
        ax_u.scatter(x_umax, extrema_u.z_at_max, edgecolors="darkred", **star_kw,
                     label=f"Upslope max: {extrema_u.max_value:.2f} m/s")
        ax_u.scatter(x_umin, extrema_u.z_at_min, edgecolors="darkblue", **star_kw,
                     label=f"Downslope max: {extrema_u.min_value:.2f} m/s")
        ax_u.legend(loc="upper left", fontsize=11, frameon=True)
    if extrema_theta is not None:
        x_thmax = (_to_lt_hours([extrema_theta.t_at_max])[0] if use_lt_axis else extrema_theta.t_at_max)
        x_thmin = (_to_lt_hours([extrema_theta.t_at_min])[0] if use_lt_axis else extrema_theta.t_at_min)
        ax_th.scatter(x_thmax, extrema_theta.z_at_max, edgecolors="darkred", **star_kw,
                      label=f"Max: {extrema_theta.max_value:.2f} K")
        ax_th.scatter(x_thmin, extrema_theta.z_at_min, edgecolors="darkblue", **star_kw,
                      label=f"Min: {extrema_theta.min_value:.2f} K")
        ax_th.legend(loc="upper left", fontsize=11, frameon=True)

    # 体裁
    for ax in (ax_u, ax_th):
        ax.set_xlabel("Local time [hour]", fontsize=13)
        ax.set_ylabel("Altitude [m]", fontsize=13)
        #ax.set_xlim(0.0, 24.0)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True, alpha=0.35)
        ax.set_facecolor(facecolor)
        ax.tick_params(labelsize=12)
        if last.K_day is not None:
            ax2 = ax.twinx()
            x_k = t_centers  # 軸に合わせた X（LT or 絶対時刻）
            ax2.plot(x_k, K_sorted, linewidth=2.0, alpha=0.8, color="black")
            ax2.set_ylabel(r"K [m$^2$/s]", fontsize=12)
            ax2.tick_params(labelsize=12)

    ax_u.set_title(title_u, fontsize=16)
    ax_th.set_title(title_theta, fontsize=16)

    # 出力
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    if show or not out_path:
        plt.show()
    plt.close(fig)
