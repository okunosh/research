from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from visualization.common.io_utils import (
    load_all_data, stack_by_variable, convert_to_standard_shapes, read_global_attr_values
)
from visualization.plot_Ri import compute_vertical_gradients_and_ri
from visualization.common.poster_utils import (
    apply_poster_mode, is_poster_enabled, poster_scale, save_if_poster
)

SEC_PER_HOUR = 3600.0

# ---------- helpers ----------
def _last_day_mask(t: np.ndarray, period: float) -> np.ndarray:
    if t.size == 0: return np.zeros(0, bool)
    t_end = float(t[-1]); t0 = max(float(t[0]), t_end - period + 1e-9)
    return t >= t0

def _roll_to_monotonic_lt(t_sec: np.ndarray, *, period: float):
    lt = (t_sec % period) / SEC_PER_HOUR
    k0 = int(np.argmin(lt))
    return k0, np.roll(lt, -k0)

def _edges_1d(xc: np.ndarray) -> np.ndarray:
    x = np.asarray(xc, float)
    if x.size == 1:
        dx = 1.0; return np.array([x[0]-dx/2, x[0]+dx/2])
    dx = np.diff(x)
    return np.concatenate(([x[0]-dx[0]/2], 0.5*(x[:-1]+x[1:]), [x[-1]+dx[-1]/2]))

def _z_edges_timevarying(Zc: np.ndarray) -> np.ndarray:
    nt, nz = Zc.shape
    Ze_row = np.empty((nt, nz+1))
    for i in range(nt):
        Ze_row[i, :] = _edges_1d(Zc[i, :])
    Ze = np.empty((nt+1, nz+1))
    Ze[0,:] = Ze_row[0,:]; Ze[-1,:] = Ze_row[-1,:]
    Ze[1:-1,:] = 0.5*(Ze_row[:-1,:] + Ze_row[1:,:])
    return Ze

def _fallback_g(attrs: dict, p_surf: float) -> float:
    try:
        g = float(attrs.get("g", np.nan))
    except Exception:
        g = np.nan
    if np.isfinite(g): return g
    planet = (attrs.get("planet") or "").lower()
    if "mars" in planet: return 3.71
    if "earth" in planet: return 9.81
    return 3.71 if p_surf <= 2000 else 9.81

# ---------- main ----------
def plot_ri_gradients_summary(
    directory: str,
    *,
    period: float = 24*SEC_PER_HOUR,
    cmap_grad: str = "bwr",
    cmap_ri: str = "coolwarm",
    ri_cbar_limit: float = 5.0,
    ri_cont_step: float = 0.25,
    ri_cont_limit: float = 2.0,
    grad_u_step: float = 0.01,
    grad_theta_step: float = 0.01,
    save_name: str = "Ri_analysis_summary.png",
) -> Path:
    out_dir = Path(directory)

    # 入力
    varnames = ["u_bar", "theta_bar", "altitude", "time", "K", "theta_0", "gamma"]
    attrs    = ["g", "planet"]
    rs = convert_to_standard_shapes(stack_by_variable(load_all_data(str(out_dir), varnames), varnames))
    g  = _fallback_g(read_global_attr_values(str(out_dir), attrs), p_surf=610.0)

    # 勾配・Ri（z_mid,t_mid は中心座標）
    du_dz, dtheta_dz, Ri, z_mid, t_mid, _ = compute_vertical_gradients_and_ri(rs, g=g)
    # t_mid: (nt, nz_mid) だが time 軸は列ごと同じはず → 1列目を採用
    t_all = np.asarray(t_mid[:, 0], float)      # (nt,)
    Zc_all = np.asarray(z_mid, float)           # (nt, nz_mid) をそのまま使う

    # 最後の1日
    mask = _last_day_mask(t_all, period)
    if not np.any(mask): mask = np.ones_like(t_all, bool)

    t_last = t_all[mask]
    Ri_l   = Ri[mask]
    du_l   = du_dz[mask]
    dth_l  = dtheta_dz[mask]
    Zc_l   = Zc_all[mask]

    # LT 単調化・並べ替え
    k0, lt_roll = _roll_to_monotonic_lt(t_last, period=period)
    lt_sorted = lt_roll
    Ri_sorted  = np.roll(Ri_l,  -k0, axis=0)
    du_sorted  = np.roll(du_l,  -k0, axis=0)
    dth_sorted = np.roll(dth_l, -k0, axis=0)
    Zc_sorted  = np.roll(Zc_l,  -k0, axis=0)

    # エッジ
    t_edges = _edges_1d(lt_sorted)
    time_var_z = not np.allclose(Zc_sorted, Zc_sorted[0, :])
    if time_var_z:
        Zy = _z_edges_timevarying(Zc_sorted)
        Tx = np.broadcast_to(t_edges[:, None], Zy.shape)
        TT = np.broadcast_to(lt_sorted[:, None], Zc_sorted.shape)
        ZZ = Zc_sorted
    else:
        z_edges = _edges_1d(Zc_sorted[0, :])
        TT, ZZ = np.meshgrid(lt_sorted, Zc_sorted[0, :], indexing="ij")

    # ノルム・レベル
    ri_norm = TwoSlopeNorm(vmin=-ri_cbar_limit, vcenter=0.0, vmax=ri_cbar_limit)
    def _sym_norm(A): 
        vmax = float(np.nanmax(np.abs(A))); 
        if not np.isfinite(vmax) or vmax == 0: vmax = 1.0
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    norm_u  = _sym_norm(du_sorted)
    norm_th = _sym_norm(dth_sorted)

    def _levels_by_step(lo, hi, step, max_levels=400):
        if step <= 0 or not (np.isfinite(lo) and np.isfinite(hi)): return np.array([])
        lo2 = np.floor(lo/step)*step; hi2 = np.ceil(hi/step)*step
        if (hi2-lo2)/step > max_levels: step = (hi2-lo2)/(max_levels//2)
        return np.arange(lo2, hi2 + 0.5*step, step)

    ri_levels = np.arange(-ri_cont_limit, ri_cont_limit + 0.5*ri_cont_step, ri_cont_step)
    lev_u  = _levels_by_step(np.nanmin(du_sorted),  np.nanmax(du_sorted),  grad_u_step)
    lev_th = _levels_by_step(np.nanmin(dth_sorted), np.nanmax(dth_sorted), grad_theta_step)

    # 描画
    # 余白は constrained_layout のパラメータで調整
    scale = poster_scale(1.5) 
    fig, axes = plt.subplots(1, 3, figsize=(15*scale, 4.2*scale),
                             constrained_layout=is_poster_enabled()
    )
    # 横方向のパネル間余白を少し広げる（必要なら 0.06〜0.12 の範囲で微調整可）
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.08, hspace=0.02)

    # y ラベルは少し内側へ（色バーとの干渉を避ける）
    for ax in axes[1:]:
        ax.yaxis.labelpad = 0  # デフォルトより内側へ

    def _draw(ax, A, cmap, norm, title, cbar_label, levels, *, cbar_pad=0.03):
        if time_var_z:
            im = ax.pcolormesh(Tx, Zy, A, shading="auto", cmap=cmap, norm=norm)
        else:
            im = ax.pcolormesh(t_edges, z_edges, A.T, shading="auto", cmap=cmap, norm=norm)

        if levels.size >= 2:
            cs = ax.contour(TT, ZZ, A, levels=levels, colors="k", linewidths=0.6, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"{v:.2f}")

        ax.set_title(title)
        ax.set_xlabel("Local time [h]")
        ax.set_ylabel("Altitude [m]")

        # 色バー。pad は控えめ（余白は上の wspace で確保する）
        cbar = fig.colorbar(im, ax=ax, pad=cbar_pad)
        cbar.set_label(cbar_label)

        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True)

    # 左・中央は cbar_pad をやや小さめに（0.02〜0.03 程度）
    _draw(axes[0], Ri_sorted, cmap_ri, ri_norm,
          f"Ri (contours within ±{ri_cont_limit:g} step {ri_cont_step:g})", "Ri",
          ri_levels, cbar_pad=0.025)

    _draw(axes[1], du_sorted, cmap_grad, norm_u,
          r"$\partial\overline{u}/\partial z$ [s$^{-1}$]",
          r"$\partial\overline{u}/\partial z$ [s$^{-1}$]",
          lev_u, cbar_pad=0.025)

    # 右端は余裕があるのでデフォルト
    _draw(axes[2], dth_sorted, cmap_grad, norm_th,
          r"$\partial\overline{\theta}/\partial z$ [K m$^{-1}$]",
          r"$\partial\overline{\theta}/\partial z$ [K m$^{-1}$]",
          lev_th, cbar_pad=0.035)

    out_path = out_dir / save_name
    # 右端のカラーバーラベル見切れ対策：tight + 少しだけ余白
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    save_if_poster(fig, out_path, dpi=500)

    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="1x3 summary (Ri, du/dz, dtheta/dz) for the last day.")
    p.add_argument("directory", type=str)
    p.add_argument("--ri-cbar-limit", type=float, default=5.0)
    p.add_argument("--ri-cont-step",  type=float, default=0.25)
    p.add_argument("--ri-cont-limit", type=float, default=0.5)
    p.add_argument("--grad-u-step",     type=float, default=0.05)
    p.add_argument("--grad-theta-step", type=float, default=0.05)
    p.add_argument(
        "--poster", action="store_true",
        help="ポスター用の図を追加保存（*_poster.pdf + フォント拡大）"
    )
    
    args = p.parse_args()
    apply_poster_mode(args.poster, title=24, label=20, tick=18)

    plot_ri_gradients_summary(
        args.directory,
        ri_cbar_limit=args.ri_cbar_limit,
        ri_cont_step=args.ri_cont_step,
        ri_cont_limit=args.ri_cont_limit,
        grad_u_step=args.grad_u_step,
        grad_theta_step=args.grad_theta_step,
    )

if __name__ == "__main__":
    main()
