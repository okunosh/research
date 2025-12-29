#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_surface_forcing_from_MCD_ver4.py

Pipeline (ver4):
  1) read MCD 24-point text (LT[h], value)
  2) anomaly (value - mean)
  3) periodic moving average (same as ver3)
  4) diagnose key times t_min/t_rise/t_max/t_set (same robust logic as ver3)
  5) build forcing:
      - low-order Fourier fit to selected points (default: raw, N=6)
     - evaluate the fitted curve at the hourly points (and at t_set/t_min)
     - enforce night monotone non-increasing on that point sequence
     - PCHIP (shape-preserving cubic) through the constrained points
  6) night monotone check + plot + save PNG

Notes
-----
- This file is intentionally independent from ver2/ver3 (no compatibility constraints).
- Goal: stabilize phase (LT) while guaranteeing no warming at night.
"""

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _maybe_use_agg_backend() -> None:
    """Use Agg backend on headless (non-Windows) environments."""

    # On many remote Linux servers, DISPLAY is unset and GUI backends fail.
    # On Windows, DISPLAY is typically unset even when GUI is available, so avoid switching.
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg", force=True)


def _float_token(x: float, *, digits: int = 6) -> str:
    """Format float into a filesystem-friendly token.

    Examples
    --------
    0.01 -> "0p01"
    0.4  -> "0p4"
    10.0 -> "10"
    """

    s = f"{float(x):.{int(digits)}g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _format_title_from_filename(path: str) -> str:
        """Format title from filename when it matches known MCD naming.

        Expected pattern (basename):
            Ls180_Lat-10_Lon260_Alt0.txt
        ->
            Ls: 180  Lat: -10  Lon: 260  Alt: 0
        """

        base = os.path.basename(path)
        name = os.path.splitext(base)[0]

        m = re.fullmatch(
                r"Ls(?P<Ls>-?\d+(?:\.\d+)?)_Lat(?P<Lat>-?\d+(?:\.\d+)?)_Lon(?P<Lon>-?\d+(?:\.\d+)?)_Alt(?P<Alt>-?\d+(?:\.\d+)?)",
                name,
        )
        if not m:
                return base

        return (
                f"Ls: {m.group('Ls')}  "
                f"Lat: {m.group('Lat')}  "
                f"Lon: {m.group('Lon')}  "
                f"Alt: {m.group('Alt')}"
        )


# -------------------------
# I/O
# 1列目: local time [hour], 2列目: value (T or theta)
# -------------------------

def load_lt_val_from_mcd_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            data.append((float(parts[0]), float(parts[1])))

    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError(f"No valid data lines found in: {path}")
    return arr[:, 0], arr[:, 1]


# -------------------------
# Periodic moving average (window points, odd recommended)
# -------------------------

def periodic_moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    k = window // 2
    out = np.empty_like(x)
    for i in range(n):
        idxs = [(i + j) % n for j in range(-k, k + 1)]
        out[i] = float(np.mean(x[idxs]))
    return out


# -------------------------
# Periodic linear interpolation on [0,24)
# -------------------------

def periodic_linear_interp(t_data: np.ndarray, y_data: np.ndarray, t_query):
    """Periodic linear interpolation.

    t_data: sorted in [0,24) (no 24 duplicate)
    y_data: same length
    t_query: scalar or array-like (hours; any real) -> wrapped to [0,24)
    """

    t = np.asarray(t_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    if t.size != y.size:
        raise ValueError("t_data and y_data must have same length.")
    if t.size < 2:
        raise ValueError("Need at least 2 points for interpolation.")

    tq = np.asarray(t_query, dtype=float) % 24.0

    t_ext = np.concatenate([t, [t[0] + 24.0]])
    y_ext = np.concatenate([y, [y[0]]])

    idx = np.searchsorted(t_ext, tq, side="right") - 1
    idx = np.clip(idx, 0, t_ext.size - 2)

    t0 = t_ext[idx]
    t1 = t_ext[idx + 1]
    y0 = y_ext[idx]
    y1 = y_ext[idx + 1]

    tq2 = tq.copy()
    tq2 = np.where(tq2 < t0, tq2 + 24.0, tq2)

    w = (tq2 - t0) / (t1 - t0)
    out = y0 + w * (y1 - y0)

    if np.isscalar(t_query):
        return float(np.asarray(out))
    return out


# -------------------------
# Diagnose times (robust: plateau-end t_min + persistent t_rise)
# -------------------------

def detect_key_times(
    lt_hour: np.ndarray,
    theta_anom: np.ndarray,
    smooth_window: int = 3,
    cool_fraction: float = 0.4,
) -> Dict[str, Any]:
    """Estimate t_min, t_rise, t_max, t_set from θ'(lt).

    Reproducibility-oriented:
    - t_min: if near-min is a plateau, use the *end* of the near-min plateau
    - t_rise: require persistent positive after neg->pos zero-cross
    """

    lt = np.asarray(lt_hour, dtype=float)
    th = np.asarray(theta_anom, dtype=float)

    order = np.argsort(lt)
    lt = lt[order]
    th = th[order]

    # drop duplicated 24h if present
    if lt[-1] > 23.9:
        lt = lt[:-1]
        th = th[:-1]

    M = lt.size
    if M < 5:
        raise ValueError("Too few points (need >=5).")

    dt_arr = np.diff(lt)
    dt = float(np.mean(dt_arr))
    if np.max(np.abs(dt_arr - dt)) > 0.05:
        print("[warn] Local time spacing is not perfectly uniform; results are approximate.")

    th_smooth = periodic_moving_average(th, window=smooth_window)

    # central difference dθ/dt (periodic)
    dth = np.empty_like(th_smooth)
    for i in range(M):
        ip = (i + 1) % M
        im = (i - 1) % M
        dth[i] = (th_smooth[ip] - th_smooth[im]) / (2.0 * dt)  # [K/hour]

    # robust epsK from residual (raw - smooth)
    resid = th - th_smooth
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma = 1.4826 * mad
    epsK = max(0.5, 2.0 * sigma)

    i_min0 = int(np.argmin(th_smooth))
    i_max = int(np.argmax(th_smooth))

    # t_min: end of near-min plateau
    near_min = th_smooth <= (float(th_smooth[i_min0]) + epsK)
    i_min = i_min0
    if near_min[i_min0]:
        start = i_min0
        steps = 0
        while near_min[(start - 1) % M] and steps < M:
            start = (start - 1) % M
            steps += 1
        end = start
        steps = 0
        while near_min[(end + 1) % M] and steps < M:
            end = (end + 1) % M
            steps += 1
        i_min = int(end)

    t_min = float(lt[i_min])
    t_max = float(lt[i_max])

    # t_rise: rotate by t_min then search until t_max, require persistence
    t_unwrap = lt - lt[i_min]
    t_unwrap[t_unwrap < 0.0] += 24.0
    shift = -i_min
    t_rot = np.roll(t_unwrap, shift)
    dth_rot = np.roll(dth, shift)
    t_max_rot = float(t_unwrap[i_max])

    POS_RUN = 2
    t_rise_rot = None
    for i in range(1, M):
        if t_rot[i] > t_max_rot + 1e-6:
            break
        if dth_rot[i - 1] < 0.0 <= dth_rot[i]:
            ok = True
            for j in range(POS_RUN):
                k = i + j
                if k >= M or t_rot[k] > t_max_rot + 1e-6:
                    ok = False
                    break
                if dth_rot[k] < 0.0:
                    ok = False
                    break
            if not ok:
                continue

            t0, t1 = float(t_rot[i - 1]), float(t_rot[i])
            y0, y1 = float(dth_rot[i - 1]), float(dth_rot[i])
            if abs(y1 - y0) < 1e-12:
                t_c = t0
            else:
                t_c = t0 + (0.0 - y0) * (t1 - t0) / (y1 - y0)
            t_rise_rot = float(t_c)
            break

    if t_rise_rot is None:
        t_rise_rot = 0.5 * t_max_rot

    t_rise = (t_min + t_rise_rot) % 24.0

    # t_set: rotate so i_max is 0h
    t_unwrap2 = lt - lt[i_max]
    t_unwrap2[t_unwrap2 < 0.0] += 24.0
    shift2 = -i_max
    t2 = np.roll(t_unwrap2, shift2)
    dth2 = np.roll(dth, shift2)
    cool = -dth2

    max_search_end = np.searchsorted(t2, min(18.0, float(t2[-1])), side="right")
    if max_search_end <= 1:
        idx_peak = 1
    else:
        idx_peak = 1 + int(np.argmax(cool[1:max_search_end]))

    cool_max = float(cool[idx_peak])
    target = float(cool_fraction) * cool_max

    t_set_rot = None
    for i in range(idx_peak + 1, M):
        if cool[i] <= target:
            t0, t1 = float(t2[i - 1]), float(t2[i])
            y0, y1 = float(cool[i - 1]), float(cool[i])
            if abs(y1 - y0) < 1e-12:
                t_c = t0
            else:
                t_c = t0 + (target - y0) * (t1 - t0) / (y1 - y0)
            t_set_rot = float(t_c)
            break

    if t_set_rot is None:
        t_set_rot = min(18.0, float(t2[-1]))

    t_set = (t_max + t_set_rot) % 24.0

    return {
        "t_min": t_min,
        "t_max": t_max,
        "t_rise": float(t_rise),
        "t_set": float(t_set),
        "lt_clean": lt,
        "theta_smooth": th_smooth,
        "dtheta_dt": dth,
        "epsK": float(epsK),
        "sigma_resid": float(sigma),
    }


# -------------------------
# Fourier fit (24h periodic)
# y(t) = a0 + Σ [a_n cos(nωt) + b_n sin(nωt)]
# -------------------------

def fit_fourier_24h_model(t_hour: np.ndarray, y: np.ndarray, n_harmonics: int = 2):
    t = np.asarray(t_hour, dtype=float)
    y = np.asarray(y, dtype=float)
    omega = 2.0 * np.pi / 24.0

    # enforce 1D
    t = t.ravel()
    y = y.ravel()

    # design matrix
    cols = [np.ones_like(t)]
    for n in range(1, int(n_harmonics) + 1):
        cols.append(np.cos(n * omega * t))
        cols.append(np.sin(n * omega * t))
    A = np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    def model(tq_hour):
        tq = np.asarray(tq_hour, dtype=float)
        res = coef[0] * np.ones_like(tq)
        idx = 1
        for n in range(1, int(n_harmonics) + 1):
            res = res + coef[idx] * np.cos(n * omega * tq)
            idx += 1
            res = res + coef[idx] * np.sin(n * omega * tq)
            idx += 1
        return res

    return model


# -------------------------
# Isotonic regression (PAVA)
# -------------------------

def isotonic_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Return y_fit that is non-decreasing and close to y (least squares, PAVA)."""

    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if w.size != n:
            raise ValueError("w must have same length as y")

    starts: List[int] = []
    ends: List[int] = []
    wsum: List[float] = []
    mean: List[float] = []

    for i in range(n):
        starts.append(i)
        ends.append(i)
        wsum.append(float(w[i]))
        mean.append(float(y[i]))

        while len(mean) >= 2 and mean[-2] > mean[-1]:
            w_new = wsum[-2] + wsum[-1]
            m_new = (wsum[-2] * mean[-2] + wsum[-1] * mean[-1]) / w_new

            ends[-2] = ends[-1]
            wsum[-2] = w_new
            mean[-2] = float(m_new)

            starts.pop()
            ends.pop()
            wsum.pop()
            mean.pop()

    y_fit = np.empty(n, dtype=float)
    for s, e, m in zip(starts, ends, mean):
        y_fit[s : e + 1] = m
    return y_fit


def isotonic_decreasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    return -isotonic_increasing(-np.asarray(y, dtype=float), w=w)


# -------------------------
# PCHIP (Fritsch-Carlson)
# -------------------------

def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of the same length")
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 points")

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")

    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    if n == 2:
        d[0] = delta[0]
        d[1] = delta[0]
        return d

    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3.0 * delta[0]):
        d0 = 3.0 * delta[0]
    d[0] = d0

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3.0 * delta[-1]):
        dn = 3.0 * delta[-1]
    d[-1] = dn

    return d


def pchip_eval(x: np.ndarray, y: np.ndarray, d: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    xq = np.asarray(xq, dtype=float)

    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    d0 = d[idx]
    d1 = d[idx + 1]

    h = x1 - x0
    s = (xq - x0) / h

    h00 = (2 * s**3 - 3 * s**2 + 1)
    h10 = (s**3 - 2 * s**2 + s)
    h01 = (-2 * s**3 + 3 * s**2)
    h11 = (s**3 - s**2)

    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1


def pchip_periodic(t_data: np.ndarray, y_data: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Evaluate PCHIP on a 24h-periodic curve defined by points in [0,24)."""

    t = np.asarray(t_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    if t.size < 2:
        raise ValueError("Need >=2 points")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t_data must be strictly increasing")

    t0 = float(t[0])
    tq = (np.asarray(t_query, dtype=float) - t0) % 24.0 + t0

    t_ext = np.concatenate([t, [t0 + 24.0]])
    y_ext = np.concatenate([y, [y[0]]])
    d_ext = pchip_slopes(t_ext, y_ext)
    return pchip_eval(t_ext, y_ext, d_ext, tq)


# -------------------------
# Night check
# -------------------------

def night_check(t_fine, forcing, t_set, t_min, tol=1e-9):
    t = np.asarray(t_fine, dtype=float)
    y = np.asarray(forcing, dtype=float)

    m1 = t >= t_set
    m2 = t <= t_min

    t1, y1 = t[m1], y[m1]
    t2, y2 = t[m2] + 24.0, y[m2]

    tn = np.concatenate([t1, t2])
    yn = np.concatenate([y1, y2])

    dy = np.diff(yn)
    imin = int(np.argmin(dy))
    imax = int(np.argmax(dy))

    print(
        f"[night check FIXED] min Δθ' = {dy[imin]:.6g} between {tn[imin]:.2f}h and {tn[imin+1]:.2f}h"
    )
    print(
        f"[night check FIXED] max Δθ' = {dy[imax]:.6g} between {tn[imax]:.2f}h and {tn[imax+1]:.2f}h"
    )
    print(
        f"[night check FIXED] night amplitude = {np.max(yn)-np.min(yn):.6g} K  (range {np.min(yn):.3f} .. {np.max(yn):.3f})"
    )

    ok = bool(np.all(dy <= tol))
    print(f"[night check FIXED] non-increasing (dy<=tol) ? {ok}")
    return ok, (tn, yn, dy)


def summarize_forcing_phase(t_fine: np.ndarray, forcing: np.ndarray) -> Dict[str, float]:
    t = np.asarray(t_fine, dtype=float)
    y = np.asarray(forcing, dtype=float)
    if t.size == 0:
        raise ValueError("t_fine is empty")
    if t.size != y.size:
        raise ValueError("t_fine and forcing must have same length")

    i_max = int(np.argmax(y))
    i_min = int(np.argmin(y))
    return {
        "t_peak": float(t[i_max]),
        "y_peak": float(y[i_max]),
        "t_trough": float(t[i_min]),
        "y_trough": float(y[i_min]),
    }


# -------------------------
# Forcing construction (ver4)
# -------------------------

def build_forcing_fourier_night_monotone_pchip(
    lt_clean: np.ndarray,
    y_smoothed: np.ndarray,
    t_set: float,
    t_min: float,
    t_fine: np.ndarray,
    n_harmonics: int = 2,
    enforce_morning_monotone: bool = False,
    t_max: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ver4 forcing builder.

    1) Fourier fit (low-order) to smoothed points.
    2) Evaluate fitted curve at the original hourly points (+ t_set, t_min).
    3) Enforce night monotone (non-increasing) on those points.
    4) PCHIP periodic through constrained points.

    Returns
    -------
    forcing : np.ndarray
    diag : dict
        knot_t, knot_y_fit, knot_y_after, night_adjust_max_abs
    """

    x0 = np.asarray(lt_clean, dtype=float)
    y0 = np.asarray(y_smoothed, dtype=float)

    order = np.argsort(x0)
    x0 = x0[order]
    y0 = y0[order]
    if x0[-1] > 23.9:
        x0 = x0[:-1]
        y0 = y0[:-1]

    # Fourier fit on smoothed points
    model = fit_fourier_24h_model(x0, y0, n_harmonics=int(n_harmonics))

    # Build point set to constrain/interpolate
    t_set = float(t_set) % 24.0
    t_min = float(t_min) % 24.0

    extra_t = np.array([t_set, t_min], dtype=float)
    x = np.concatenate([x0, extra_t])

    # unique times (keep last)
    uniq = {}
    for tx in x:
        uniq[float(tx)] = True
    x = np.array(sorted(uniq.keys()), dtype=float)

    y_fit = np.asarray(model(x), dtype=float)
    y_after = y_fit.copy()

    # --- night monotone (non-increasing) on fitted point sequence ---
    t_min_unwrap = t_min + 24.0 if t_min < t_set else t_min
    x_unwrap = np.where(x < t_set, x + 24.0, x)
    mask_night = (x_unwrap >= t_set - 1e-12) & (x_unwrap <= t_min_unwrap + 1e-12)
    night_idx = np.where(mask_night)[0]
    if night_idx.size >= 2:
        ord_n = np.argsort(x_unwrap[night_idx])
        ni = night_idx[ord_n]
        y_after[ni] = isotonic_decreasing(y_after[ni])

    # --- optional morning monotone (non-decreasing) on constrained points ---
    if enforce_morning_monotone:
        if t_max is None:
            raise ValueError("t_max must be provided when enforce_morning_monotone=True")
        t_max = float(t_max) % 24.0
        t_max_unwrap = t_max + 24.0 if t_max < t_min else t_max
        x_unwrap2 = np.where(x < t_min, x + 24.0, x)
        mask_morning = (x_unwrap2 >= t_min - 1e-12) & (x_unwrap2 <= t_max_unwrap + 1e-12)
        morn_idx = np.where(mask_morning)[0]
        if morn_idx.size >= 2:
            ord_m = np.argsort(x_unwrap2[morn_idx])
            mi = morn_idx[ord_m]
            y_after[mi] = isotonic_increasing(y_after[mi])

    forcing = pchip_periodic(x, y_after, t_fine)

    diag = {
        "knot_t": x,
        "knot_y_fit": y_fit,
        "knot_y_after": y_after,
        "night_adjust_max_abs": float(np.max(np.abs(y_after - y_fit))) if y_fit.size else 0.0,
    }
    return forcing, diag


def main() -> None:
    _maybe_use_agg_backend()

    ap = argparse.ArgumentParser()
    ap.add_argument("mcd_txt", help="e.g., MCDdata/Ls90_Lat-10_Lon240_Alt0.txt")
    ap.add_argument(
        "--half-window",
        type=int,
        default=1,
        help="smoothing half-window; window size = 2*half_window+1 (periodic)",
    )
    ap.add_argument(
        "--dt-hour",
        type=float,
        default=0.01,
        help="fine grid step [hour] for plotting / forcing evaluation",
    )
    ap.add_argument(
        "--set-frac",
        type=float,
        default=0.4,
        help="cool_fraction for t_set (fast->slow cooling threshold)",
    )
    ap.add_argument(
        "--fourier-harmonics",
        type=int,
        default=6,
        help="number of Fourier harmonics for low-frequency fit (default: 6)",
    )
    ap.add_argument(
        "--fit-to",
        choices=["raw", "smoothed"],
        default="raw",
        help="which points to fit with Fourier model (raw or smoothed). Key-time detection still uses smoothed.",
    )
    ap.add_argument(
        "--morning-monotone",
        action="store_true",
        help="(optional) enforce non-decreasing constraint on morning segment (t_min->t_max)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show plot window (still saves PNG).",
    )
    args = ap.parse_args()

    lt_in, val_in = load_lt_val_from_mcd_txt(args.mcd_txt)
    order = np.argsort(lt_in)
    lt_in = lt_in[order]
    val_in = val_in[order]

    # anomaly (treat as θ')
    y_raw_in = val_in - np.mean(val_in)

    smooth_window = 2 * int(args.half_window) + 1
    key = detect_key_times(
        lt_hour=lt_in,
        theta_anom=y_raw_in,
        smooth_window=smooth_window,
        cool_fraction=float(args.set_frac),
    )

    t_min = float(key["t_min"])
    t_rise = float(key["t_rise"])
    t_max = float(key["t_max"])
    t_set = float(key["t_set"])
    lt_clean = np.asarray(key["lt_clean"], dtype=float)
    y_sm = np.asarray(key["theta_smooth"], dtype=float)

    # Align raw series to lt_clean (detect_key_times may have dropped a 24h duplicate)
    y_raw_clean = y_raw_in.copy()
    if lt_in.size != lt_clean.size:
        # common case: input includes 24.0 point; lt_clean excludes it
        mask = lt_in < 24.0 - 1e-9
        lt_in2 = lt_in[mask]
        y_raw_in2 = y_raw_in[mask]
        if lt_in2.size == lt_clean.size and np.allclose(lt_in2, lt_clean, atol=1e-6, rtol=0.0):
            y_raw_clean = y_raw_in2
        else:
            # fallback: interpolate raw onto lt_clean
            y_raw_clean = periodic_linear_interp(lt_in % 24.0, y_raw_in, lt_clean)

    if args.fit_to == "raw":
        y_fit_points = y_raw_clean
    else:
        y_fit_points = y_sm

    print(
        "=== Diagnosed times (ver4: low-order Fourier fit + night monotone + PCHIP) ==="
    )
    print(f"t_min  : {t_min:6.2f} h")
    print(f"t_rise : {t_rise:6.2f} h  (cold -> warm)")
    print(f"t_max  : {t_max:6.2f} h")
    print(f"t_set  : {t_set:6.2f} h  (fast -> slow cooling)")
    print(
        f"[info] smooth_window = {smooth_window} points, cool_fraction(set-frac) = {args.set_frac}"
    )
    print(
        f"[info] epsK(min-plateau) = {key['epsK']:.3f} K  (sigma_resid={key['sigma_resid']:.3f} K)"
    )
    print(f"[info] fourier_harmonics = {int(args.fourier_harmonics)}")

    t_fine = np.arange(0.0, 24.0, float(args.dt_hour))

    forcing, diag = build_forcing_fourier_night_monotone_pchip(
        lt_clean=lt_clean,
        y_smoothed=y_fit_points,
        t_set=t_set,
        t_min=t_min,
        t_fine=t_fine,
        n_harmonics=int(args.fourier_harmonics),
        enforce_morning_monotone=bool(args.morning_monotone),
        t_max=t_max,
    )

    phase = summarize_forcing_phase(t_fine, forcing)
    print(
        f"[forcing phase] peak  at {phase['t_peak']:.2f} h (max {phase['y_peak']:.3f} K)"
    )
    print(
        f"[forcing phase] trough at {phase['t_trough']:.2f} h (min {phase['y_trough']:.3f} K)"
    )

    print(f"[night segment] t_set={t_set:.2f}h -> t_min={t_min:.2f}h (wrap, ordered)")
    night_check(t_fine, forcing, t_set, t_min, tol=1e-8)
    print(f"[night constraint] max |Δknot| = {diag['night_adjust_max_abs']:.6g} K")

    # plot
    plt.figure(figsize=(10, 5))

    # Always show raw points (QC baseline)
    plt.scatter(
        lt_clean,
        y_raw_clean,
        label="MCD θ'(raw)",
        alpha=0.7,
        s=28,
        color="k",
    )

    # Show the points actually used for the Fourier fit.
    # (If fit-to=smoothed, this also visualizes the smoothed series.)
    plt.scatter(
        lt_clean,
        y_fit_points,
        label=f"fit points: {args.fit_to}",
        alpha=0.9,
        s=28,
        color="tab:orange",
    )
    plt.plot(
        t_fine,
        forcing,
        label="forcing",
        linewidth=2,
    )

    # Key times on plot: show only those actually used in forcing construction.
    plt.axvline(
        t_set,
        linestyle="--",
        linewidth=1.2,
        color="C2",
        label=f"t_set={t_set:.2f}h",
    )
    plt.axvline(
        t_min,
        linestyle="--",
        linewidth=1.2,
        color="C3",
        label=f"t_min={t_min:.2f}h",
    )
    if args.morning_monotone:
        plt.axvline(
            t_max,
            linestyle="--",
            linewidth=1.2,
            color="C4",
            label=f"t_max={t_max:.2f}h",
        )

    # Grid + emphasize y=0 line
    plt.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.6)
    plt.axhline(0, color="k", linewidth=1.2, alpha=0.9)

    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 4))
    plt.xlabel("Local Time [hour]")
    plt.ylabel("θ' at surface [K]")
    plt.title(_format_title_from_filename(args.mcd_txt))
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Include key CLI options in filename to avoid accidental overwrite.
    suffix = (
        f"_hw{int(args.half_window)}"
        f"_dt{_float_token(float(args.dt_hour))}"
        f"_sf{_float_token(float(args.set_frac))}"
        f"_N{int(args.fourier_harmonics)}"
        f"_fit{args.fit_to}"
        f"_morn{int(bool(args.morning_monotone))}"
    )
    output_path = os.path.splitext(args.mcd_txt)[0] + suffix + ".png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Figure saved to: {output_path}")

    if args.show:
        plt.show()
    plt.close()


def _build_time_axis(sol_sec: float = 86400.0, dt: float = 0.1) -> np.ndarray:
    n = int(round(float(sol_sec) / float(dt)))
    if n <= 0:
        raise ValueError("sol_sec/dt must be >= 1")
    return np.arange(n, dtype=float) * float(dt)


def generate_surface_forcing_from_mcd_ver4(
    *,
    mcd_txt_path: str,
    output_nc_path: str,
    output_png_path: str,
    dt: float = 0.1,
    sol_sec: float = 86400.0,
    half_window: int = 1,
    set_frac: float = 0.4,
    fourier_harmonics: int = 6,
    fit_to: str = "raw",
    morning_monotone: bool = False,
    dt_hour_plot: float = 0.01,
) -> Dict[str, Any]:
    """Generate ver4 surface forcing NetCDF + diagnostic PNG.

    NetCDF format aims to be compatible with the existing K generator:
    - coord: time [s]
    - var:   theta_surf(time) [K] (anomaly)
    - attrs: interp_method="fourier" (compat) + interp_method_detail for ver4

    Returns
    -------
    diag : dict
        Contains key times and phase summary.
    """

    _maybe_use_agg_backend()

    if fit_to not in {"raw", "smoothed"}:
        raise ValueError("fit_to must be 'raw' or 'smoothed'")

    # Lazy import so that plot-only usage doesn't require xarray.
    import xarray as xr  # type: ignore

    lt_in, val_in = load_lt_val_from_mcd_txt(mcd_txt_path)
    order = np.argsort(lt_in)
    lt_in = lt_in[order]
    val_in = val_in[order]

    mean_abs = float(np.mean(val_in))
    y_raw_in = val_in - mean_abs

    smooth_window = 2 * int(half_window) + 1
    key = detect_key_times(
        lt_hour=lt_in,
        theta_anom=y_raw_in,
        smooth_window=smooth_window,
        cool_fraction=float(set_frac),
    )
    t_min = float(key["t_min"])
    t_rise = float(key["t_rise"])
    t_max = float(key["t_max"])
    t_set = float(key["t_set"])
    lt_clean = np.asarray(key["lt_clean"], dtype=float)
    y_sm = np.asarray(key["theta_smooth"], dtype=float)

    # Align raw series to lt_clean (detect_key_times may have dropped a 24h duplicate)
    y_raw_clean = y_raw_in.copy()
    if lt_in.size != lt_clean.size:
        mask = lt_in < 24.0 - 1e-9
        lt_in2 = lt_in[mask]
        y_raw_in2 = y_raw_in[mask]
        if lt_in2.size == lt_clean.size and np.allclose(lt_in2, lt_clean, atol=1e-6, rtol=0.0):
            y_raw_clean = y_raw_in2
        else:
            y_raw_clean = periodic_linear_interp(lt_in % 24.0, y_raw_in, lt_clean)

    y_fit_points = y_raw_clean if fit_to == "raw" else y_sm

    # Build knots (PCHIP periodic curve)
    t_fine = np.arange(0.0, 24.0, float(dt_hour_plot))
    _, diag_build = build_forcing_fourier_night_monotone_pchip(
        lt_clean=lt_clean,
        y_smoothed=y_fit_points,
        t_set=t_set,
        t_min=t_min,
        t_fine=t_fine,
        n_harmonics=int(fourier_harmonics),
        enforce_morning_monotone=bool(morning_monotone),
        t_max=t_max,
    )
    knot_t = np.asarray(diag_build["knot_t"], dtype=float)
    knot_y = np.asarray(diag_build["knot_y_after"], dtype=float)

    # Sample to time axis (seconds)
    time_sec = _build_time_axis(sol_sec=float(sol_sec), dt=float(dt))
    lt_target = time_sec / float(sol_sec) * 24.0
    theta_surf = pchip_periodic(knot_t, knot_y, lt_target).astype(np.float64)

    # Final forcing extrema on the output time grid (useful for validation)
    i_min_out = int(np.argmin(theta_surf))
    i_max_out = int(np.argmax(theta_surf))
    tsec_min_out = float(time_sec[i_min_out])
    tsec_max_out = float(time_sec[i_max_out])
    lt_min_out = float((tsec_min_out / float(sol_sec)) * 24.0)
    lt_max_out = float((tsec_max_out / float(sol_sec)) * 24.0)
    y_min_out = float(theta_surf[i_min_out])
    y_max_out = float(theta_surf[i_max_out])
    print(
        f"[forcing output extrema] min at LT={lt_min_out:.3f} h (t={tsec_min_out:.3f} s, {y_min_out:.3f} K); "
        f"max at LT={lt_max_out:.3f} h (t={tsec_max_out:.3f} s, {y_max_out:.3f} K)"
    )

    ds = xr.Dataset(
        coords={"time": ("time", time_sec.astype(np.float64))},
        data_vars={"theta_surf": ("time", theta_surf)},
        attrs={
            "theta_surf_mean": mean_abs,
            "theta0_surface_mean": mean_abs,
            "interp_method": "fourier",
            "interp_method_detail": "ver4_fourier-nightmono-pchip",
            "n_harmonics": int(fourier_harmonics),
            "dt": float(dt),
            "sol_sec": float(sol_sec),
            "fit_to": str(fit_to),
            "half_window": int(half_window),
            "set_frac": float(set_frac),
            # netCDF attributes do not support boolean dtype in netCDF4 backend
            "morning_monotone": int(bool(morning_monotone)),
            "t_min": float(t_min),
            "t_rise": float(t_rise),
            "t_max": float(t_max),
            "t_set": float(t_set),
            # extrema on the final output grid (seconds / local time)
            "tsec_min_output": float(tsec_min_out),
            "tsec_max_output": float(tsec_max_out),
            "lt_min_output": float(lt_min_out),
            "lt_max_output": float(lt_max_out),
            "theta_min_output": float(y_min_out),
            "theta_max_output": float(y_max_out),
            "description": (
                "surface potential temperature anomaly [K] from MCD (ver4); "
                "stored variable is deviation from daily mean; "
                "mean value is stored in global attributes."
            ),
        },
    )
    ds["theta_surf"].attrs["long_name"] = "surface potential temperature anomaly at surface"
    ds["theta_surf"].attrs["units"] = "K"

    out_nc = output_nc_path
    out_dir = os.path.dirname(out_nc)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    enc = {"theta_surf": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(out_nc, encoding=enc)
    print(f"[INFO] Saved surface forcing NetCDF (ver4) to: {out_nc}")

    # Diagnostic plot (same style as CLI)
    plt.figure(figsize=(10, 5))
    plt.scatter(
        lt_clean,
        y_raw_clean,
        label="MCD θ'(raw)",
        alpha=0.7,
        s=28,
        color="k",
    )
    plt.scatter(
        lt_clean,
        y_fit_points,
        label=f"fit points: {fit_to}",
        alpha=0.9,
        s=28,
        color="tab:orange",
    )
    plt.plot(
        lt_target % 24.0,
        theta_surf,
        label="forcing",
        linewidth=2,
    )
    plt.axvline(t_set, linestyle="--", linewidth=1.2, color="C2", label=f"t_set={t_set:.2f}h")
    plt.axvline(t_min, linestyle="--", linewidth=1.2, color="C3", label=f"t_min={t_min:.2f}h")
    if morning_monotone:
        plt.axvline(t_max, linestyle="--", linewidth=1.2, color="C4", label=f"t_max={t_max:.2f}h")
    plt.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.6)
    plt.axhline(0, color="k", linewidth=1.2, alpha=0.9)
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 4))
    plt.xlabel("Local Time [hour]")
    plt.ylabel("θ' at surface [K]")
    plt.title(_format_title_from_filename(mcd_txt_path))
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_png = output_png_path
    out_png_dir = os.path.dirname(out_png)
    if out_png_dir:
        os.makedirs(out_png_dir, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved surface forcing figure (ver4) to: {out_png}")

    phase = summarize_forcing_phase(t_fine, pchip_periodic(knot_t, knot_y, t_fine))
    return {
        "t_min": t_min,
        "t_rise": t_rise,
        "t_max": t_max,
        "t_set": t_set,
        "phase": phase,
        "night_adjust_max_abs": float(diag_build.get("night_adjust_max_abs", 0.0)),
        "lt_min_output": lt_min_out,
        "lt_max_output": lt_max_out,
    }


if __name__ == "__main__":
    main()
