# ===== mix_momentum_profiles.py =====
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# 既存ヘルパ（あなたの環境のパスに合わせて調整）
from visualization.last_day_plotters import _to_lt_hours, _centers_to_edges
from visualization.plot_Ri import compute_vertical_gradients_and_ri
from visualization.pressure_t_alt_map import plot_last_day_generic

from visualization.common.poster_utils import apply_poster_mode, save_if_poster, is_poster_enabled, poster_scale


# 秒/時（直値3600は避ける）
SEC_PER_HOUR = float(timedelta(hours=1).total_seconds())


# ───────────────────────────────────────────────────────────
# 1) 最後の1日データの準備（u, ρ, ρu と Ri を揃える）
# ───────────────────────────────────────────────────────────
def prepare_last_day_inputs(
    reshaped_stacked: Dict[str, np.ndarray],
    *,
    rho: np.ndarray,  
    ri_results: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    g: float = 3.721,
) -> Dict[str, np.ndarray]:
    """
    最後の1日ぶんを抽出し、形状を (nt_last, nz) に揃えて返す。
    ρu（運動量密度）と、必要なら Ri を計算。
    戻り値: dict(time, z, u, rho, mom, Ri, z_mid)
    """
    to_f64 = lambda x: np.asarray(x, dtype=np.float64)

    t_full   = to_f64(reshaped_stacked["time"])       # (nt_full,)
    u_full   = to_f64(reshaped_stacked["u_bar"])      # (nt_full, nz)
    z_raw    = to_f64(reshaped_stacked["altitude"])   # (nt_full,nz) or (nz,)
    rho_full = to_f64(rho)                             # (nt_full, nz)

    period = 24.0 * SEC_PER_HOUR
    t_end  = float(t_full[-1])
    mask   = (t_full > (t_end - period))

    t_last = t_full[mask]                              # (nt_last,)
    u_last = u_full[mask, :]                           # (nt_last, nz)
    rho_last = rho_full[mask, :]                       # (nt_last, nz)

    nt_last, nz = u_last.shape

    # z を (nt_last, nz) に整形
    if z_raw.ndim == 2 and z_raw.shape[0] == t_full.size and z_raw.shape[1] == nz:
        z_last = z_raw[mask, :]
    elif z_raw.ndim == 1 and z_raw.size == nz:
        z_last = np.broadcast_to(z_raw[None, :], (nt_last, nz))
    elif z_raw.ndim == 2 and z_raw.shape == (1, nz):
        z_last = np.broadcast_to(z_raw, (nt_last, nz))
    else:
        raise ValueError(f"altitude 形状不整合: {z_raw.shape} -> (nt_last,{nz})")

    mom_last = rho_last * u_last                       # (nt_last, nz)

    # Ri（未指定なら計算）
    if ri_results is None:
        du_dz, dtheta_dz, Ri, z_mid, t_mid, theta_mid = compute_vertical_gradients_and_ri(
            reshaped_stacked, g=g
        )
    else:
        du_dz, dtheta_dz, Ri, z_mid, t_mid, theta_mid = ri_results

    return dict(
        time=t_last, z=z_last, u=u_last, rho=rho_last, mom=mom_last,
        Ri=np.asarray(Ri, dtype=np.float64), z_mid=np.asarray(z_mid, dtype=np.float64)
    )


# ───────────────────────────────────────────────────────────
# 2) 混合ロジック：各時刻で条件を満たすときのみ層内一様化
# ───────────────────────────────────────────────────────────
def _layer_mean_by_height(z: np.ndarray, f: np.ndarray) -> Tuple[float, float]:
    """
    1本のプロファイル z(レベル) と f(同) から台形則で ∫ f dz / H を返す。
    戻り値: (平均値, 層厚 H)
    """
    H = float(z[-1] - z[0])
    if H <= 0.0:
        return np.nan, np.nan
    integ_f = np.trapz(f, z)
    return integ_f / H, H

"""
def apply_momentum_mixing_per_time(
    z_row: np.ndarray, u_row: np.ndarray, rho_row: np.ndarray,
    Ri_row_mid: np.ndarray,                      # (nz-1,) セル間（mid）
    *,
    ri_upper: float = 0.25,
    require_positive: bool = False,             # ← 緩和条件: 0<Ri を要求しない
) -> Tuple[np.ndarray, np.ndarray, Optional[int], bool]:
    
    #1時刻ぶんに混合を適用。
    #戻り値: (mom_mixed_row, u_mixed_row, top_idx, applied)
      #- top_idx: 最大|u|のレベル index。条件不成立時は None もあり
      #- applied: 条件を満たして混合を適用したか
   
    nz = u_row.size
    j_star = int(np.nanargmax(np.abs(u_row)))          # 最大|u|
    if j_star < 1:
        return rho_row * u_row, u_row.copy(), None, False

    # 条件: 地面〜j_star までの全セルで Ri < 0.25
    Ri_slice = Ri_row_mid[:j_star]                     # (j_star,)
    if require_positive:
        cond = np.all((Ri_slice > 0.0) & (Ri_slice < ri_upper))
    else:
        cond = np.all(Ri_slice < ri_upper)

    if (not cond) or (not np.isfinite(Ri_slice).all()):
        return rho_row * u_row, u_row.copy(), j_star, False

    # 層 0..j_star の平均運動量密度・平均ρ → 一様化
    z_layer   = z_row[:j_star+1]
    mom_layer = (rho_row * u_row)[:j_star+1]
    rho_layer = rho_row[:j_star+1]

    mom_mean, H = _layer_mean_by_height(z_layer, mom_layer)
    rho_mean, _ = _layer_mean_by_height(z_layer, rho_layer)
    if not (np.isfinite(mom_mean) and np.isfinite(rho_mean) and rho_mean > 0.0):
        return rho_row * u_row, u_row.copy(), j_star, False

    u_mean = mom_mean / rho_mean

    mom_out = rho_row * u_row
    u_out   = u_row.copy()
    mom_out[:j_star+1] = mom_mean
    u_out[:j_star+1]   = u_mean
    return mom_out, u_out, j_star, True
"""

def apply_momentum_mixing_per_time(
    z_row: np.ndarray,
    u_row: np.ndarray,
    rho_row: np.ndarray,
    Ri_row_mid: np.ndarray,                    # (nz-1,) セル間（mid）
    *,
    ri_upper: float = 0.25,                    # ← ri_uper → ri_upper に修正
    require_positive: bool = False,            # True のとき 0 < Ri < ri_upper を要求
) -> Tuple[np.ndarray, np.ndarray, Optional[int], bool]:
    """
    1時刻ぶんに運動量混合を適用（地面から連続して Ri < ri_upper を満たす層まで）。
    返り値:
      mom_mixed_row, u_mixed_row, top_idx, applied
        - top_idx: 混合層の最上位レベル index（セル中心） / None
        - applied: 混合を適用したかどうか
    実装メモ:
      - Ri はセル間(mid)なので、mid=0..k-1 が条件を満たすなら
        混合するセル中心レベルは 0..k（= k+1 層）になる。
    """
    nz = u_row.size
    # 入力チェック
    if nz < 2 or Ri_row_mid.size != nz - 1:
        return rho_row * u_row, u_row.copy(), None, False

    # 1) 地面から「連続して」条件を満たす mid の本数を数える
    if require_positive:
        ok = np.isfinite(Ri_row_mid) & (Ri_row_mid > 0.0) & (Ri_row_mid < ri_upper)
    else:
        ok = np.isfinite(Ri_row_mid) & (Ri_row_mid < ri_upper)

    # 最初に条件を破る mid の位置（無ければ len(ok)）
    first_bad = np.argmax(~ok) if (~ok).any() else ok.size
    # 連続して満たす mid の本数
    n_mid_ok = int(first_bad)

    # mid が 1 本未満 → 混合しない（厚みが無い）
    if n_mid_ok < 1:
        return rho_row * u_row, u_row.copy(), None, False

    # 2) 混合するセル中心レベルは 0..top_idx
    top_idx = n_mid_ok  # mid k-1 までOK ⇒ レベルは 0..k
    # 安全装置
    top_idx = min(top_idx, nz - 1)

    # 3) 高さ重み（台形則）で層平均の (ρu) と ρ を取り ū を作る
    z_layer   = z_row[:top_idx + 1]          # (top_idx+1,)
    mom_layer = (rho_row * u_row)[:top_idx + 1]
    rho_layer = rho_row[:top_idx + 1]

    mom_mean, H = _layer_mean_by_height(z_layer, mom_layer)
    rho_mean, _ = _layer_mean_by_height(z_layer, rho_layer)
    if not (np.isfinite(mom_mean) and np.isfinite(rho_mean) and rho_mean > 0.0):
        return rho_row * u_row, u_row.copy(), None, False

    #divide by rho mean 
    u_mean = mom_mean / rho_mean

    # 4) 出力に反映（層 0..top_idx を一様化）
    mom_out = (rho_row * u_row).copy()
    u_out   = u_row.copy()
    mom_out[:top_idx + 1] = mom_mean
    u_out[:top_idx + 1]   = u_mean

    return mom_out, u_out, top_idx, True

    

def apply_momentum_mixing_last_day(
    prepared: Dict[str, np.ndarray],
    *,
    ri_upper: float = 0.25,
    require_positive: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    最後の1日全時刻に対して混合適用。
    戻り値: (mom_mixed, u_mixed, applied_indices, top_idx_list)
    """
    t   = prepared["time"]
    z   = prepared["z"]
    u   = prepared["u"]
    rho = prepared["rho"]
    Ri  = prepared["Ri"]

    nt, nz = u.shape
    mom_mixed = np.empty_like(u)
    u_mixed   = np.empty_like(u)
    applied_indices: List[int] = []
    top_idx_list: List[int] = []

    for i in range(nt):
        mom_i, u_i, j_star, applied = apply_momentum_mixing_per_time(
            z[i, :], u[i, :], rho[i, :], Ri[i, :],
            ri_upper=ri_upper, require_positive=require_positive
        )
        mom_mixed[i, :] = mom_i
        u_mixed[i, :]   = u_i
        top_idx_list.append(-1 if j_star is None else j_star)
        if applied:
            applied_indices.append(i)

    return mom_mixed, u_mixed, applied_indices, top_idx_list


# ───────────────────────────────────────────────────────────
# 3) 可視化：A/B pcolormesh（時間×高度）、C 時刻別プロファイル
# ───────────────────────────────────────────────────────────
def plot_mixing_results_pcolormesh(
    reshaped_stacked: Dict[str, np.ndarray],
    prepared: Dict[str, np.ndarray],
    mom_mixed: np.ndarray,
    u_mixed: np.ndarray,
    *,
    out_dir: str,
) -> None:
    """
    A: 混合後の運動量密度（ρu） pcolormesh
    B: 混合後の風速（u） pcolormesh
    既存の共通プロッタ plot_last_day_generic を使い回し。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # A: ρu（u_rho スロットを使用）
    fields_A = {"T": mom_mixed}
    plot_last_day_generic(
        reshaped_stacked, var_name="T", fields=fields_A,
        cmap="cividis", title="Mixed momentum density (ρu)", units="kg m$^{-2}$ s$^{-1}$",
        save_path=str(Path(out_dir, "A_mixed_momentum_density.png")), invert_cbar=False
    )

    # B: u（ラベル等は上書き）
    fields_B = {"T": u_mixed}  # 共通関数の既定を流用してタイトル等は上書き
    plot_last_day_generic(
        reshaped_stacked, var_name="T", fields=fields_B,
        cmap="bwr", title="Mixed wind speed (u)", units="m s$^{-1}$",
        save_path=str(Path(out_dir, "B_mixed_wind_speed.png")), invert_cbar=False
    )


def plot_mixed_u_like_reference( 
    reshaped_stacked: dict,
    u_mixed: np.ndarray,                    # (nt_full,nz) でも (nt_last,nz) でもOK
    *,
    save_path,
    cmap: str = "bwr",
    period: float = None,                   # Noneなら自動で24h
    title = r"$\overline{u}$ [m/s]",
    show_colorbar: bool = True,
    k_level_index: int = 0,                 # 右軸に重ねる K の高さindex（最下層=0）
) -> None:
    """
    参照図フォーマットで 混合後の u を描画（bwr, 右軸に K[時間]、最大/最小に★）。
    K は reshaped_stacked['K'] から取得。補間なし。最後の1日・時間1次元に揃える。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from pathlib import Path
    from datetime import timedelta
    from visualization.last_day_plotters import _to_lt_hours

    # 極値ユーティリティ（無ければフォールバック）
    try:
        from extract_extremas import LastDaySlice, get_extrema_on_last_day, get_minima_on_last_day  # type: ignore
        has_extrema = True
    except Exception:
        LastDaySlice = None
        get_extrema_on_last_day = None
        get_minima_on_last_day = None
        has_extrema = False

    SEC_PER_HOUR = float(timedelta(hours=1).total_seconds())

    # ---- 入力 ----
    t_full = np.asarray(reshaped_stacked["time"], dtype=np.float64)   # (nt_full,)
    z_raw  = np.asarray(reshaped_stacked["altitude"])                 # (nt_full,nz) or (nz,)
    U      = np.asarray(u_mixed, dtype=np.float64)                    # (nt_u, nz)
    if U.ndim != 2:
        raise ValueError(f"u_mixed must be 2D (nt, nz); got {U.shape}")

    # K を取得（後で 1D の時間列に還元）
    K_raw = reshaped_stacked.get("K", None)
    if K_raw is not None:
        K_raw = np.asarray(K_raw, dtype=np.float64)

    nt_full = t_full.size
    nt_u, nz = U.shape
    if period is None:
        period = 24.0 * SEC_PER_HOUR

    # 最後の1日マスク
    t_end = float(t_full[-1])
    mask_last = (t_full > (t_end - period))
    nt_last = int(mask_last.sum())

    # ---- u を最後の1日に整列 ----
    if nt_u == nt_full:
        t_last = t_full[mask_last]                     # (nt_last,)
        U_last = U[mask_last, :]                       # (nt_last, nz)
        # z を (nt_last,nz) or (nz,)
        if z_raw.ndim == 2 and z_raw.shape == (nt_full, nz):
            Z = z_raw[mask_last, :]
        elif z_raw.ndim == 1 and z_raw.size == nz:
            Z = np.broadcast_to(z_raw[None, :], (nt_last, nz))
        else:
            raise ValueError(f"altitude shape mismatch: {z_raw.shape}")
    elif nt_u == nt_last:
        t_last = t_full[mask_last]
        U_last = U
        if z_raw.ndim == 2 and z_raw.shape == (nt_full, nz):
            Z = z_raw[mask_last, :]
        elif z_raw.ndim == 2 and z_raw.shape == (nt_last, nz):
            Z = z_raw
        elif z_raw.ndim == 1 and z_raw.size == nz:
            Z = np.broadcast_to(z_raw[None, :], (nt_last, nz))
        else:
            raise ValueError(f"altitude shape mismatch: {z_raw.shape}")
    else:
        raise ValueError(
            f"Time length mismatch: u_mixed nt={nt_u}, full={nt_full}, last-day={nt_last}."
        )

    # ---- K を 1D(時間) に還元 → 最後の1日に揃える（補間なし）----
    def _k_lastday_1d(K_arr: np.ndarray) -> np.ndarray:
        """K を時間1次元にし、最後の1日(nt_last)だけ返す。高さは k_level_index を採用。"""
        nonlocal nz, nt_full, nt_last, mask_last
        j = int(np.clip(k_level_index, 0, nz - 1))
        if K_arr.ndim == 1:
            # (nt_full,) or (nt_last,) など
            if K_arr.size == nt_full:
                return K_arr[mask_last]
            elif K_arr.size == nt_last:
                return K_arr
            elif K_arr.size > nt_last:
                return K_arr[-nt_last:]  # 末尾採用（補間なし）
            else:
                raise ValueError(f"K length {K_arr.size} < last-day {nt_last}.")
        elif K_arr.ndim == 2:
            h, w = K_arr.shape
            # (nt, nz)
            if w == nz and h in (nt_full, nt_last):
                col = K_arr[:, j]
                return col[mask_last] if h == nt_full else col
            # (nz, nt)
            if h == nz and w in (nt_full, nt_last):
                row = K_arr[j, :]
                return row[mask_last] if w == nt_full else row
            # 典型外：高さと時間の位置が不明 → 高さ0を優先して時間軸らしき次元を採用
            if h in (nt_full, nt_last):
                col = K_arr[:, 0]
                return col[mask_last] if h == nt_full else col
            if w in (nt_full, nt_last):
                row = K_arr[0, :]
                return row[mask_last] if w == nt_full else row
            raise ValueError(f"Unsupported K shape {K_arr.shape}.")
        else:
            raise ValueError(f"Unsupported K ndim {K_arr.ndim}.")

    K_last = _k_lastday_1d(K_raw) if K_raw is not None else None  # → shape = (nt_last,)

    # ---- 時間軸：024 に正規化して昇順へ（u, z, K も同じ順へ）----
    t_centers = _to_lt_hours(t_last / SEC_PER_HOUR)  # unwrap で負が出ることがある
    t_mod     = np.mod(t_centers, 24.0)
    order     = np.argsort(t_mod)

    t_hours   = t_mod[order]                # (nt_last,)
    U_last    = U_last[order, :]            # (nt_last, nz)
    Z         = Z[order, :]                 # (nt_last, nz)
    if K_last is not None:
        if K_last.ndim != 1 or K_last.size != nt_last:
            raise ValueError(f"K still not 1D last-day: shape={K_last.shape}, expected ({nt_last},)")
        K_last = K_last[order]

    # ---- pcolormesh 入力（2D化）----
    X = np.broadcast_to(t_hours[:, None], (nt_last, nz))
    if Z.ndim == 1:
        Z = np.broadcast_to(Z[None, :], (nt_last, nz))

    # ---- カラースケール（中心0）----
    vmin_i = float(np.nanmin(U_last))
    vmax_i = float(np.nanmax(U_last))
    if not np.isfinite(vmin_i) or not np.isfinite(vmax_i) or vmin_i == vmax_i:
        vmin_i, vmax_i = -1.0, 1.0
    norm = TwoSlopeNorm(vmin=vmin_i, vcenter=0.0, vmax=vmax_i)

    # ---- 描画 ----
    Path(save_path or ".").parent.mkdir(parents=True, exist_ok=True)
    scale = poster_scale(1.5) 
    fig, ax = plt.subplots(1, 1, figsize=(10*scale, 4*scale),
                           constrained_layout=is_poster_enabled())
    print("plot_mixed_u_like_reference")
    m = ax.pcolormesh(X, Z, U_last, cmap=cmap, norm=norm, shading="auto")

    if show_colorbar:
        pad    = 0.09 if is_poster_enabled() else 0.08
        frac   = 0.05  if is_poster_enabled() else 0.045
        shrink = 0.92  if is_poster_enabled() else 1.0
        cb = fig.colorbar(m, ax=ax, fraction=frac, pad=pad, shrink=shrink)
        cb.set_label(r"$\overline{u}$ [m s$^{-1}$]")

    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(title or r"$\overline{u}$ [m/s]")
    ax.grid(True, alpha=0.4)

    # 右軸 K（1D・長さ nt_last のときだけ）
    if K_last is not None:
        ax2 = ax.twinx()
        ax2.plot(t_hours, K_last, color="k", lw=1.8, alpha=0.9)
        ax2.set_ylabel("K [m$^2$/s]")

    # ---- 極値（正の最大・負の最小）に★ ----
    def _fallback_extrema(U_nt_nz: np.ndarray):
        Upos = np.where(U_nt_nz > 0, U_nt_nz, -np.inf)
        Uneg = np.where(U_nt_nz < 0, U_nt_nz, +np.inf)
        idx_up = np.unravel_index(np.nanargmax(Upos), Upos.shape)
        idx_dn = np.unravel_index(np.nanargmin(Uneg), Uneg.shape)
        return idx_up, U_nt_nz[idx_up], idx_dn, U_nt_nz[idx_dn]

    if has_extrema:
        slice_u = LastDaySlice(
            t_day=t_hours, z_day=Z.T, u_day=U_last.T,
            theta_day=np.zeros_like(U_last.T), K_day=None,
            idx_mask=np.ones_like(t_hours, dtype=bool),
        )
        ext_max = get_extrema_on_last_day(slice_u, field="u")
        if get_minima_on_last_day is not None:
            ext_min = get_minima_on_last_day(slice_u, field="u")
            up_t, up_z, up_val = float(ext_max.t_at_max), float(ext_max.z_at_max), float(ext_max.max_value)
            dn_t, dn_z, dn_val = float(ext_min.t_at_min), float(ext_min.z_at_min), float(ext_min.min_value)
        else:
            (iu, ju), up_val, (idn, jdn), dn_val = _fallback_extrema(U_last)
            up_t, up_z = float(t_hours[iu]), float(Z[iu, ju])
            dn_t, dn_z = float(t_hours[idn]), float(Z[idn, jdn])
    else:
        (iu, ju), up_val, (idn, jdn), dn_val = _fallback_extrema(U_last)
        up_t, up_z = float(t_hours[iu]), float(Z[iu, ju])
        dn_t, dn_z = float(t_hours[idn]), float(Z[idn, jdn])

    # ★（白抜き・凡例に線を出さない）
    up_star, = ax.plot(up_t, up_z, marker="*", ms=12, mfc="white", mec="red",
                   mew=1.2, linestyle="None", zorder=5)
    dn_star, = ax.plot(dn_t, dn_z, marker="*", ms=12, mfc="white", mec="blue",
                   mew=1.2, linestyle="None", zorder=5)

    #legend
    leg = ax.legend(
        [up_star, dn_star],
        [f"Upslope max: {up_val:.2f} m/s", f"Downslope max: {dn_val:.2f} m/s"],
        loc="upper left",
        frameon=True,
        handletextpad=0.6,
    )


    if not is_poster_enabled():
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        save_if_poster(fig, save_path, enabled=False, dpi=500)

        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)




"""    
def plot_before_after_profiles(
    prepared: Dict[str, np.ndarray],
    mom_mixed: np.ndarray,
    applied_indices: List[int],
    top_idx_list: List[int],
    *,
    out_dir: str,
    max_plots: Optional[int] = None,
) -> None:
    
    #C: 条件を満たした各地方時で、運動量密度（ρu）の混合前/後プロファイルを保存。
    #   横軸=ρu, 縦軸=z。前=赤破線、後=青実線。
    #  1時刻=1ファイルで保存（多い場合は max_plots で上限）。
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t   = prepared["time"]
    z   = prepared["z"]
    mom = prepared["mom"]

    if max_plots is not None:
        applied_indices = applied_indices[:max_plots]

    lt_hours = _to_lt_hours(t / SEC_PER_HOUR)

    for i in applied_indices:
        j_star = top_idx_list[i]
        if j_star is None or j_star <= 0:
            continue
        z_layer    = z[i, :j_star+1]
        mom_before = mom[i, :j_star+1]
        mom_after  = mom_mixed[i, :j_star+1]

        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        ax.plot(mom_before, z_layer, "r--", lw=2, label="before")
        ax.plot(mom_after,  z_layer, "b-",  lw=2, label="after")
        ax.set_xlabel(r"Momentum density  $\rho u$  [kg m$^{-2}$ s$^{-1}$]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title(f"Momentum profile before/after mixing  (LT={lt_hours[i]:.2f} h)")
        ax.grid(True)
        ax.legend(loc="best")
        save_path = Path(out_dir) / "mixed_momentum_profile"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        save_path = Path(save_path, f"C_momentum_profile_LT_{lt_hours[i]:05.2f}h.png")
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

"""
def plot_near_surface_wind_timeseries(
        reshaped_stacked: dict,
        prepared: dict,
        u_mixed: np.ndarray,
        *,
        level_index: int = 0,         # 0: 最下層, 1: 最下層から1つ上…にしたい場合は 1 を指定
        out_path: str = "out/mixing_lastday/near_surface_wind_timeseries.png",
) -> None:
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import timedelta
    # 既存ヘルパを利用（モジュール先頭などで import 済み想定）
    # from visualization.last_day_plotters import _to_lt_hours
    # SEC_PER_HOUR は既存の定数を利用（未定義なら timedelta から取得）
    try:
        _SEC_PER_HOUR = float(SEC_PER_HOUR)
    except NameError:
        _SEC_PER_HOUR = float(timedelta(hours=1).total_seconds())

    # 入力
    t_last = np.asarray(prepared["time"], dtype=np.float64)     # (nt_last,)
    u_last = np.asarray(prepared["u"],    dtype=np.float64)     # (nt_last, nz)
    u_mix  = np.asarray(u_mixed,           dtype=np.float64)    # (nt_last, nz)

    nt_last, nz = u_last.shape
    if not (u_mix.shape == (nt_last, nz)):
        raise ValueError(f"u_mixed shape mismatch: {u_mix.shape} vs expected {(nt_last, nz)}")

    # レベル決定（範囲チェック＆フォールバック）
    j = int(level_index)
    if j < 0 or j >= nz:
        j = 0  # フォールバック：最下層

    # Local Time [hour] を 024 に正規化して昇順に並べ替え（折れ線をきれいに）
    lt = _to_lt_hours(t_last / _SEC_PER_HOUR)                   # (nt_last,)
    order = np.argsort(lt)
    x = lt[order]
    y_before = u_last[order, j]
    y_after  = u_mix[order,  j]

    # プロット
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scale = poster_scale(1.5) 
    fig, ax = plt.subplots(figsize=(8*scale, 4*scale),
                           constrained_layout=is_poster_enabled,
    )
    ax.plot(x, y_before, "r--", lw=1.8, label="before mixing")
    ax.plot(x, y_after,  "b-",  lw=2.0, label="after  mixing")
    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Wind speed u [m s$^{-1}$]")
    ax.set_title(f"Near-surface wind time series (level index = {j})")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    save_if_poster(fig, out_path, dpi=500) #poster
    plt.close(fig)

"""
def plot_before_after_wind_profiles(
    prepared: dict,
    u_mixed: np.ndarray,
    applied_indices: List[int],
    top_idx_list: List[int],
    *,
    directory: str,                      # 例: args.directory と同義で out_dir を渡す
    max_plots: Optional[int] = None,
) -> None:
    
    #Ri<0.25 条件が成立した地方時のみ、風速 u の混合前/後プロファイルを保存。
    #横軸=u [m/s], 縦軸=高度 [m]。前=赤破線, 後=青実線。
    #保存先: <directory>/mixed_wind_speed_profile/u_profile_LT_XX.XXh.png
    
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import timedelta

    try:
        _SEC_PER_HOUR = float(SEC_PER_HOUR)
    except NameError:
        _SEC_PER_HOUR = float(timedelta(hours=1).total_seconds())

    out_dir = Path(directory) / "mixed_wind_speed_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    t   = np.asarray(prepared["time"], dtype=np.float64)  # (nt_last,)
    z   = np.asarray(prepared["z"],    dtype=np.float64)  # (nt_last, nz)
    u0  = np.asarray(prepared["u"],    dtype=np.float64)  # (nt_last, nz)
    u1  = np.asarray(u_mixed,          dtype=np.float64)  # (nt_last, nz)

    nt_last, nz = u0.shape
    if u1.shape != (nt_last, nz):
        raise ValueError(f"u_mixed shape mismatch: {u1.shape} vs {(nt_last, nz)}")

    if max_plots is not None:
        applied_indices = applied_indices[:max_plots]

    lt_hours = _to_lt_hours(t / _SEC_PER_HOUR)  # 024

    for i in applied_indices:
        j_star = top_idx_list[i]
        if j_star is None or j_star <= 0:
            continue

        z_layer  = z[i, :j_star+1]
        u_before = u0[i, :j_star+1]
        u_after  = u1[i, :j_star+1]

        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        ax.plot(u_before, z_layer, "r--", lw=2.0, label="before")
        ax.plot(u_after,  z_layer, "b-",  lw=2.0, label="after")
        ax.set_xlabel("Wind speed  u  [m s$^{-1}$]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title(f"Wind profile before/after mixing  (LT={lt_hours[i]:.2f} h)")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best")

        save_path = out_dir / f"u_profile_LT_{lt_hours[i]:05.2f}h.png"
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
"""
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

def plot_before_after_wind_profiles(
    prepared: dict,
    u_mixed: np.ndarray,
    applied_indices: List[int],
    top_idx_list: List[Optional[int]],
    *,
    directory: str,
    max_plots: Optional[int] = None,
    mom_mixed: Optional[np.ndarray] = None,   # （あれば）混合後の ρu 全層
) -> None:

    print(len(applied_indices), applied_indices[:15])
    """
    Ri<ri_upper を満たして混合を適用した地方時のみ、風速(必須)と運動量(あれば)の
    前後プロファイルを「全高度」で描画・保存する。

    'after' プロファイルは、元の全高度プロファイルに対し、
    地面〜top_idx までを混合後値で置き換えたもの。
    運動量は prepared['rho'] があれば before=ρ*u_before、after は
      - mom_mixed が与えられたら層内を mom_mixed で置換（それ以外は元の値）
      - 無ければ ρ * u_after_full を用いる（厳密な一様(ρu)と異なる可能性あり）
    保存先: <directory>/mixed_wind_speed_profile/profiles_LT_XX.XXh.png
    """
    # --- 時間単位（秒→時） ---
    try:
        _SEC_PER_HOUR = float(SEC_PER_HOUR)  # 既存定数があれば使う
    except NameError:
        _SEC_PER_HOUR = float(timedelta(hours=1).total_seconds())

    # --- 出力先 ---
    out_dir = Path(directory) / "mixed_wind_speed_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 入力取り出し ---
    t   = np.asarray(prepared["time"], dtype=np.float64)  # (nt_last,)
    z   = np.asarray(prepared["z"],    dtype=np.float64)  # (nt_last, nz)
    u0  = np.asarray(prepared["u"],    dtype=np.float64)  # (nt_last, nz)
    u1i = np.asarray(u_mixed,          dtype=np.float64)  # (nt_last, nz)

    nt_last, nz = u0.shape
    if u1i.shape != (nt_last, nz):
        raise ValueError("u_mixed shape mismatch: {} vs {}".format(u1i.shape, (nt_last, nz)))

    rho = prepared.get("rho", None)
    if rho is not None:
        rho = np.asarray(rho, dtype=np.float64)
        if rho.shape != (nt_last, nz):
            raise ValueError("rho shape mismatch: {} vs {}".format(rho.shape, (nt_last, nz)))

    if mom_mixed is not None:
        mom_mixed = np.asarray(mom_mixed, dtype=np.float64)
        if mom_mixed.shape != (nt_last, nz):
            raise ValueError("mom_mixed shape mismatch: {} vs {}".format(mom_mixed.shape, (nt_last, nz)))

    # プロットする時刻の制限
    if max_plots is not None:
        applied_indices = applied_indices[:max_plots]

    # ローカル時 (024h)
    try:
        lt_hours = _to_lt_hours(t / _SEC_PER_HOUR)  # 既存ユーティリティがあれば
    except NameError:
        lt_hours = (t % (24.0 * _SEC_PER_HOUR)) / _SEC_PER_HOUR

    # --- 各時刻で描画 ---
    for i in applied_indices:
        top_idx = top_idx_list[i]
        if top_idx is None or top_idx <= 0:
            # 混合層が厚みゼロならスキップ
            continue
        if top_idx >= nz:
            top_idx = nz - 1  # 念のためガード

        # 全高度の before/after を構成
        z_full        = z[i, :]               # (nz,)
        u_before_full = u0[i, :].copy()       # (nz,)
        u_after_full  = u0[i, :].copy()
        # ← 地面〜top_idx を混合後で置換（層外は元の値を保持）
        u_after_full[:top_idx + 1] = u1i[i, :top_idx + 1]

        # 運動量（可能なら）NoT USE NOW
        have_mom = None #rho is not None
        if have_mom:
            mom_before_full = rho[i, :] * u_before_full
            if mom_mixed is not None:
                mom_after_full = mom_before_full.copy()
                mom_after_full[:top_idx + 1] = mom_mixed[i, :top_idx + 1]
            else:
                # 厳密な(ρu)一様化とは異なる可能性はあるが、情報が無い場合は ρ*u_after を描く
                mom_after_full = rho[i, :] * u_after_full

        # --- 描画 ---
        scale = poster_scale(1.5)
        if have_mom:
            fig, axes = plt.subplots(1, 2, figsize=(9.5, 6), constrained_layout=True)
            ax_u, ax_m = axes[0], axes[1]
        else:
            fig, ax_u = plt.subplots(1, 1, figsize=(5.2*scale, 6*scale), constrained_layout=True)
            ax_m = None

        # 風速
        ax = ax_u
        ax.plot(u_before_full, z_full, "r--", lw=2.0, label="before")
        ax.plot(u_after_full,  z_full, "b-",  lw=2.2, label="after")
        ax.axhspan(z_full[0], z_full[top_idx], color="tab:blue", alpha=0.08, lw=0)  # 混合層の帯
        ax.set_xlabel("Wind speed  u  [m s$^{-1}$]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Wind profile (LT={:.2f})".format(lt_hours[i]))
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best")

        # 運動量
        if ax_m is not None:
            ax = ax_m
            ax.plot(mom_before_full, z_full, "r--", lw=2.0, label="before")
            ax.plot(mom_after_full,  z_full, "b-",  lw=2.2, label="after")
            ax.axhspan(z_full[0], z_full[top_idx], color="tab:blue", alpha=0.08, lw=0)
            ax.set_xlabel(r"Momentum  $\rho u$  [kg m$^{-2}$ s$^{-1}$]")
            ax.set_ylabel("Altitude [m]")
            ax.set_title("Momentum profile (LT={:.2f} h)".format(lt_hours[i]))
            ax.grid(True, alpha=0.4)
            ax.legend(loc="best")

        # 保存
        save_path = out_dir / "profiles_LT_{:05.2f}h.png".format(lt_hours[i])
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", pad_inches=0.05)
        save_if_poster(fig, save_path, dpi=500) #poster
        plt.close(fig)



def run_last_day_mixing_pipeline(
    reshaped_stacked: Dict[str, np.ndarray],
    fields,
    *,
    rho: np.ndarray,   # (nt, nz) 計算済み密度
    ri_results: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    g: float = 3.721,
    ri_upper: float = 0.25,
    require_positive: bool = False,   # 緩和条件（Ri<0.25）
    out_dir: str = "out/mixing_lastday",
    profile_max_plots: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    最後の1日データで混合を適用し、A/B/C（運動量）と C’（風速）を出力。
    戻り値: (mom_mixed, u_mixed, applied_indices, top_idx_list)
    """
    prepared = prepare_last_day_inputs(
        reshaped_stacked, rho=rho, ri_results=ri_results, g=g
    )

    mom_mixed, u_mixed, applied_indices, top_idx_list = apply_momentum_mixing_last_day(
        prepared, ri_upper=ri_upper, require_positive=require_positive
    )

    # A/B の pcolormesh
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_mixing_results_pcolormesh(
        reshaped_stacked, prepared, mom_mixed, u_mixed, out_dir=out_dir
    )

    # C: 運動量の混合前後プロファイ
    """
    plot_before_after_profiles(
        prepared, mom_mixed, applied_indices, top_idx_list,
        out_dir=out_dir, max_plots=profile_max_plots
    )
    """

    # C’: 風速の混合前後プロファイル
    plot_before_after_wind_profiles(
        prepared, u_mixed, applied_indices, top_idx_list,
        directory=out_dir, max_plots=profile_max_plots
    )

    return mom_mixed, u_mixed, applied_indices, top_idx_list


if __name__ == "__main__":
    import argparse
    import os
    from visualization.common.io_utils import load_all_data, read_global_attr_values, stack_by_variable, convert_to_standard_shapes
    from visualization.common.io_utils import _fallback_g

    from visualization.pressure_t_alt_map import calc_pressure, calc_temperature, calc_density
    from visualization.plot_Ri import compute_vertical_gradients_and_ri
    
    
    parser = argparse.ArgumentParser(description='Plot u_bar and theta_bar from NetCDF files in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing NetCDF files')
    parser.add_argument('--poster', action='store_true',
                        help="ポスター用の図を追加保存（*_poster.pdf, フォント拡大）")

    args = parser.parse_args()
    apply_poster_mode(args.poster, title=24, label=20, tick=18) 
    

    varnames = ["u_bar", "theta_bar", "altitude", "time", "K", "theta_0", "gamma"]
    attrs_names = ["g"]
    
    data_list = load_all_data(args.directory, varnames)
    stacked = stack_by_variable(data_list, varnames)
    reshaped_stacked = convert_to_standard_shapes(stacked)

    attr_dic = read_global_attr_values(args.directory, attrs_names)

    p_surf = 610 #Pa
    g = _fallback_g(attr_dic, p_surf)

    p = calc_pressure(reshaped_stacked, p_surf, g)
    
    temperature = calc_temperature(reshaped_stacked, p, p_surf)
    rho = calc_density(reshaped_stacked, p, temperature)
    fields = {"p": p, "T": temperature, "rho": rho}
    
    ri_results = compute_vertical_gradients_and_ri(reshaped_stacked, g=g)

    mom_mixed, u_mixed, applied_idx, top_idx = run_last_day_mixing_pipeline(
        reshaped_stacked,
        fields,
        rho=rho,
        # ri_results=ri_results,  # 渡さなければ内部計算
        g=g,
        ri_upper=0.25,
        require_positive=False,   # 緩和条件（Ri<0.25）で適用
        out_dir=args.directory,
        profile_max_plots=24,
    )

    today = "20251024"
    level_index = 1
    save_filename_sf_wind = f"{today}_near_surface_wind_timeseries_lv{level_index}"
    # 最下層から1つ上の格子点j
    plot_near_surface_wind_timeseries(
        reshaped_stacked,
        prepared=prepare_last_day_inputs(reshaped_stacked, rho=rho,g=g),
        u_mixed=u_mixed,
        level_index=level_index,
        out_path=os.path.join(args.directory, save_filename_sf_wind)
    )

    save_filename_mw_profile = f"{today}_mixed_wind_profiles"
    # u_mixed は (nt_full,nz) でも (nt_last,nz) でもOK。K は任意。
    plot_mixed_u_like_reference(
        reshaped_stacked,
        u_mixed=u_mixed,
        save_path=os.path.join(args.directory,save_filename_mw_profile),
        cmap="bwr",
    )
