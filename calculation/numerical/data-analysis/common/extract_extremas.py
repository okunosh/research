# common/extrema.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0
SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR

@dataclass(frozen=True)
class LastDaySlice:
    """最後の1日分のスライス結果をまとめた構造体。"""
    t_day: np.ndarray          # shape: [nt_day]   単位は hours
    z_day: np.ndarray          # shape: [nz, nt_day] 
    u_day: np.ndarray          # shape: [nt, nz]
    theta_day: np.ndarray      # shape: [nt, nz]
    K_day: Optional[np.ndarray]  # shape: [nt_day] or None
    idx_mask: np.ndarray       # shape: [nt]      True=採用（最後の1日）

def _ensure_time_in_hours(t: np.ndarray, time_unit: Optional[str]) -> Tuple[np.ndarray, str]:
    """
    時間配列 t を hours に正規化する。time_unit が None の場合は推定する。
    """
    t = np.asarray(t).reshape(-1)  # [nt]
    if time_unit is None:
        # ざっくり推定：最大値が 48 より十分大きければ秒とみなす
        time_unit = "seconds" if np.nanmax(t) > 1000 else "hours"

    if time_unit == "seconds":
        return t / SECONDS_PER_HOUR, "hours"
    elif time_unit == "hours":
        return t, "hours"
    else:
        raise ValueError('time_unit must be "seconds" or "hours"')

def _broadcast_z(z: np.ndarray, nt: int) -> np.ndarray:
    """
    z が 1D([nz]) の場合は [nz, nt] にブロードキャスト。
    すでに [nz, nt] の場合はそのまま。
    """
    z = np.asarray(z)
    if z.ndim == 1:
        return np.repeat(z[:, None], nt, axis=1)
    if z.ndim == 2:
        return z
    raise ValueError("z must be 1D [nz] or 2D [nz, nt].")

def _mask_last_full_day(t_hours: np.ndarray, day_length_hours: float = HOURS_PER_DAY) -> np.ndarray:
    """
    最後の「ちょうど day_length_hours 時間」を含むブールマスクを返す。
    """
    t_end = np.nanmax(t_hours)
    t_start = t_end - day_length_hours
    #print("t_end: ", t_end,  "t_start", t_start)
    # t が単調増加でなくてもOKなように「範囲マスク」を使う
    mask = (t_hours > t_start - 1e-9) & (t_hours <= t_end + 1e-9)
    return mask

def extract_last_day(
    z: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    K: Optional[np.ndarray] = None,
    *,
    time_unit: Optional[str] = None,     # "seconds" | "hours" | None(自動推定)
    day_length_hours: float = HOURS_PER_DAY
) -> LastDaySlice:
    """
    入力（z, t, u, theta, K）から「最後の1日分」を抽出して返す。
    - t は秒 or 時を許容（自動推定）。返却は hours。
    - z は 1D or 2D を許容。返却は 2D（[nz, nt_day]）。
    - u, theta は [nz, nt] を想定。
    - K は [nt] or None。
    """
    # 時間正規化（hours）
    t_hours, _ = _ensure_time_in_hours(t, time_unit)

    # 形状チェック
    u = np.asarray(u)
    theta = np.asarray(theta)
    if u.ndim != 2 or theta.ndim != 2:
        raise ValueError("u, theta must be 2D arrays of shape [nz, nt].")
    if u.shape != theta.shape:
        raise ValueError("u and theta must have the same shape.")
    nz, nt = u.shape

    if t_hours.shape[0] != nt:
        raise ValueError(f"t length ({t_hours.shape[0]}) must match u/theta nt ({nt}).")

    if K is not None:
        K = np.asarray(K).reshape(-1)
        if K.shape[0] != nt:
            raise ValueError(f"K length ({K.shape[0]}) must match nt ({nt}).")

    # 最後の1日のマスク
    mask = _mask_last_full_day(t_hours, day_length_hours=day_length_hours)
    if not np.any(mask):
        raise ValueError("Could not extract the last-day segment (mask is empty).")

    idx = np.where(mask)[0]

    t_day = t_hours[idx]  # [nt_day]
    z_2d = _broadcast_z(z, nt)  # [nz, nt]
    z_day = z_2d[:, idx]        # [nz, nt_day]
    u_day = u[:, idx]
    theta_day = theta[:, idx]
    K_day = K[idx] if K is not None else None

    return LastDaySlice(
        t_day=t_day,
        z_day=z_day,
        u_day=u_day,
        theta_day=theta_day,
        K_day=K_day,
        idx_mask=mask
    )

@dataclass(frozen=True)
class ExtremaResult:
    max_value: float
    min_value: float
    t_at_max: float     # hours
    t_at_min: float     # hours
    z_at_max: float     # meters (同じ単位で返す)
    z_at_min: float
    idx_max: Tuple[int, int]  # (iz, it)
    idx_min: Tuple[int, int]

def finite_nanargmax(a: np.ndarray) -> Tuple[int, int]:
    """NaNを無視して (iz,it) の最大値位置を返す。"""
    if np.all(~np.isfinite(a)):
        raise ValueError("All values are non-finite; cannot take argmax.")
    flat_idx = np.nanargmax(a)
    return np.unravel_index(flat_idx, a.shape)

def finite_nanargmin(a: np.ndarray) -> Tuple[int, int]:
    """NaNを無視して (iz,it) の最小値位置を返す。"""
    if np.all(~np.isfinite(a)):
        raise ValueError("All values are non-finite; cannot take argmin.")
    flat_idx = np.nanargmin(a)
    return np.unravel_index(flat_idx, a.shape)

def get_extrema_on_last_day(
    last: LastDaySlice,
    field: str  # "u" or "theta"
) -> ExtremaResult:
    """
    最後の1日区間における field の極値情報を返す。
    """
    if field not in {"u", "theta"}:
        raise ValueError('field must be "u" or "theta".')

    V = last.u_day if field == "u" else last.theta_day  # [nz, nt_day]
    iz_max, it_max = finite_nanargmax(V)
    iz_min, it_min = finite_nanargmin(V)

    vmax = V[iz_max, it_max]
    vmin = V[iz_min, it_min]

    tmax = last.t_day[it_max]  # hours
    tmin = last.t_day[it_min]
    zmax = last.z_day[iz_max, it_max]
    zmin = last.z_day[iz_min, it_min]

    return ExtremaResult(
        max_value=float(vmax),
        min_value=float(vmin),
        t_at_max=float(tmax),
        t_at_min=float(tmin),
        z_at_max=float(zmax),
        z_at_min=float(zmin),
        idx_max=(iz_max, it_max),
        idx_min=(iz_min, it_min),
    )

def get_surface_theta_at_time(
    last: LastDaySlice,
    extrema_u: ExtremaResult,
    surface_index: int = 0
) -> Dict[str, float]:
    """
    u の極大・極小時刻における “地表（surface_index 行）” の θ̄ と、
    そのときの “同一格子 (iz,it)” の θ̄ を返す。
    """
    it_umax = extrema_u.idx_max[1]
    it_umin = extrema_u.idx_min[1]

    theta_umax_surface = float(last.theta_day[surface_index, it_umax])
    theta_umin_surface = float(last.theta_day[surface_index, it_umin])

    theta_umax_samecell = float(last.theta_day[extrema_u.idx_max])
    theta_umin_samecell = float(last.theta_day[extrema_u.idx_min])

    return dict(
        theta_surface_at_umax=theta_umax_surface,
        theta_surface_at_umin=theta_umin_surface,
        theta_samecell_at_umax=theta_umax_samecell,
        theta_samecell_at_umin=theta_umin_samecell,
    )
