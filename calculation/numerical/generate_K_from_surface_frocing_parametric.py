#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_K_distribution_from_surface_frocing_parametric.py

make_surface_forcing_parametric.py が出力した
surface_forcing_parametric/.../*.nc（theta_0: 地表温位偏差）を読み込み、
旧 generate_K_distribution.py の generate_K_1dim() に相当する処理を行って
位相同期した拡散係数 K(altitude, time) を生成して保存する。

【仕様】
- θ_0(t) は既にフーリエフィット済み（NetCDF に 0.1秒刻みで保存済み）とみなす。
- θ_0 -> K_base の変換は、旧 generate_K_1dim の Step 5 と同じ min-max マッピング。
- 「早朝の台地」は「基準系地方時 τ = (LT − Δt) mod 24 における K_base の最小時刻
  τ_min まで K_min が続く」とする。
- K の最大値が現れる τ_max から 22h（基準系）の間で cos で K_max → K_min に減衰。
- τ > 22h は K_min の台地。
- 高度方向の格子点数は NZ（同じ K(t) を高度方向に一様コピー）。
"""

import os
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse


# ======================================================================
# 1. 基本パラメータ
# ======================================================================

# 入力: surface_forcing_parametric の NetCDF ファイル
# 必要に応じてここを書き換える
INPUT_FORCING_NC = (
    "surface_forcing_parametric/lat-30.0_Ls90.0/210.5_40.0_0.0.nc"
)

# 出力: K を保存するルートディレクトリ
K_OUTPUT_ROOT = "K_parametric"

# 高度方向の格子点数（altitude 次元の長さ）
NZ = 260 + 1  # 0〜780 m を 1 m 刻みとみなす

# K のレンジ [m^2/s]
K_MIN = 3.0
K_MAX = 100.0

# 基準系（MSL Ls=90）での「夜の始まり」 [hour]
# ここでは「cos 減衰の終了時刻」と「深夜側台地の開始時刻」として使う。
NIGHT_LATE_START = 22.0 


# ======================================================================
# 2. θ_0 -> K_base の変換（min-max マッピング）
# ======================================================================

def compute_K_base(theta_0, k_min=K_MIN, k_max=K_MAX):
    """
    surface forcing の θ_0(t) から、連続な K_base(t) を作る。

    旧 generate_K_1dim() の「Step 5: 正規化してKを作成」に対応。
    theta_min -> K_min, theta_max -> K_max となる線形マッピング。

    Parameters
    ----------
    theta_0 : 1D ndarray
        surface forcing の地表温位偏差 [K]
    k_min, k_max : float
        K の最小・最大値 [m^2/s]

    Returns
    -------
    K_base : 1D ndarray (float32)
        θ_0 と同じ形・位相を持つ連続な K_base(t)
    """
    theta_0 = np.asarray(theta_0, dtype=float)
    th_min = float(np.min(theta_0))
    th_max = float(np.max(theta_0))
    dtheta = th_max - th_min

    if dtheta < 1.0e-6:
        # ほぼ一定なら K_min でフラット
        return np.full_like(theta_0, k_min, dtype=np.float32)

    norm = (theta_0 - th_min) / dtheta  # 0〜1 に正規化
    norm = np.clip(norm, 0.0, 1.0)

    K_base = k_min + norm * (k_max - k_min)
    return K_base.astype(np.float32)


# ======================================================================
# 3. 旧 generate_K_1dim() をベースにした K(t) の生成
#    - ただし「夜の台地の終了時刻」は τ_min（K_base の最小時刻）に変更
# ======================================================================

def compute_K_from_surface_forcing(forcing_ds,
                                   k_min=K_MIN,
                                   k_max=K_MAX):
    """
    surface_forcing_parametric の Dataset から、
    旧 generate_K_1dim() とほぼ同等の処理を行って K(t) を作る。

    ・θ_0 -> K_base は min-max マッピング。
    ・基準系地方時 τ = (LT − Δt) mod 24 を定義し、
      - まず K_base の最小値をとる時刻 τ_min まで K_min 台地とする。
      - その後、K の最大値時刻 τ_max から 22h まで cos で K_max → K_min に減衰。
      - τ > 22h は K_min 台地。

    Parameters
    ----------
    forcing_ds : xarray.Dataset
        make_surface_forcing_parametric.py の出力 NetCDF を open したもの
    k_min, k_max : float
        K の最小・最大値 [m^2/s]

    Returns
    -------
    time_sec : 1D ndarray
        時刻 [s]
    K_1d : 1D ndarray (float32)
        K(t) [m^2/s]
    """
    if "theta_0" not in forcing_ds:
        raise KeyError("Input Dataset does not contain 'theta_0'.")

    theta_0 = forcing_ds["theta_0"].values
    time_sec = forcing_ds["time"].values.astype(float)
    t_hour = time_sec / 3600.0

    # 位相シフト Δt [hour] （なければ 0 とみなす）
    phase_shift_hours = float(forcing_ds.attrs.get("phase_shift_hours", 0.0))

    # --- 1) θ_0 -> K_base ---
    K_base = compute_K_base(theta_0, k_min=k_min, k_max=k_max)

    # ここでの K_min_val は「変換後の最小値」（夜の K 台地の値）
    K_min_val = float(np.min(K_base))

    # --- 2) 基準系地方時 τ = (LT − Δt) mod 24 を計算 ---
    tau = (t_hour - phase_shift_hours) % 24.0

    # K を K_base からスタートさせる
    K = K_base.copy()

    # --- 3) 早朝の K_min 台地: τ_min まで K_min ---
    # K_base が最小となる τ_min（もっとも寒い時刻）を求める
    idx_min = int(np.argmin(K_base))
    tau_min = float(tau[idx_min])

    # τ <= τ_min を K_min 台地とする
    mask_early_plateau = tau <= tau_min
    K[mask_early_plateau] = K_min_val

    # --- 4) 最大値の要素番号と時刻を取得（τ 空間） ---
    idx_max = int(np.argmax(K))
    tau_max = float(tau[idx_max])

    # --- 5) cos 減衰区間: τ_max 〜 22h ---
    if tau_max < NIGHT_LATE_START:
        mask_target = (tau >= tau_max) & (tau <= NIGHT_LATE_START)
        tau_target = tau[mask_target]

        if tau_target.size > 0 and (NIGHT_LATE_START - tau_max) > 1.0e-6:
            # x=0 -> 最大, x=1 -> 最小(22時)
            x = (tau_target - tau_max) / (NIGHT_LATE_START - tau_max)

            K_max_val = float(K[idx_max])
            K_range = K_max_val - K_min_val

            K_replacement = (
                K_min_val + K_range * 0.5 * (1.0 + np.cos(np.pi * x))
            ).astype(np.float32)

            K[mask_target] = K_replacement

    # --- 6) 深夜台地: τ > 22h を K_min で固定 ---
    mask_night_late = tau > NIGHT_LATE_START
    K[mask_night_late] = K_min_val

    return time_sec, K.astype(np.float32)


# ======================================================================
# 4. NetCDF Dataset を組み立てる
# ======================================================================

def make_K_dataset(forcing_ds,
                   time_sec,
                   K_1d,
                   source_path,
                   nz=NZ,
                   k_min=K_MIN,
                   k_max=K_MAX):
    """
    surface_forcing の Dataset と K(t) から K(altitude, time) の Dataset を作る。

    Parameters
    ----------
    forcing_ds : xarray.Dataset
        surface_forcing_parametric の Dataset
    time_sec : 1D ndarray
        時刻 [s]
    K_1d : 1D ndarray
        時間依存の K(t) [m^2/s]
    source_path : str
        入力 surface forcing NetCDF のパス
    nz : int
        altitude 次元の長さ
    k_min, k_max : float
        K の最小・最大値 [m^2/s]

    Returns
    -------
    ds_K : xarray.Dataset
        K(altitude, time) の Dataset
    """
    time_sec = np.asarray(time_sec, dtype=float)
    K_1d = np.asarray(K_1d, dtype=np.float32)

    if K_1d.shape[0] != time_sec.shape[0]:
        raise ValueError(
            "K_1d and time must have same length, got %d and %d"
            % (K_1d.shape[0], time_sec.shape[0])
        )

    altitude = np.arange(nz, dtype=float)  # [m]
    K_2d = np.tile(K_1d, (nz, 1))

    # coords
    coords = {
        "altitude": (
            "altitude",
            altitude,
            {
                "unit": "meters",
                "description": "height above ground (1 m spacing assumed)",
            },
        ),
        "time": (
            "time",
            time_sec,
            getattr(forcing_ds["time"], "attrs", {
                "unit": "seconds",
                "description": "time from local midnight",
            }),
        ),
    }

    # data_vars
    data_vars = {
        "K": (
            ("altitude", "time"),
            K_2d,
            {
                "unit": "m^2/s",
                "description": "turbulent diffusivity, uniform in altitude",
            },
        )
    }

    # attrs: surface_forcing の属性をコピーしつつ K 用の情報を追加
    attrs = dict(forcing_ds.attrs)
    attrs.update(
        {
            "history": "Created on %s from surface forcing file"
            % datetime.now().isoformat(),
            "source_surface_forcing_file": os.path.abspath(source_path),
            "K_min_param": float(k_min),
            "K_max_param": float(k_max),
            "NZ": int(nz),
            "altitude_points": int(nz),
            "night_late_start_hour_base": float(NIGHT_LATE_START),
            "night_rule_description": (
                "Early-night plateau extends up to the hour at which "
                "K_base (mapped from theta_0) attains its minimum in "
                "tau=(LT-phase_shift_hours) mod 24. "
                "From tau_max (hour of maximum K) to 22h, K is smoothly "
                "reduced from K_max to K_min with a cosine; "
                "for tau>22h, K is set to K_min."
            ),
        }
    )

    ds_K = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return ds_K


# ======================================================================
# 5. NetCDF と PNG を保存
# ======================================================================

def save_K_netcdf_and_png(ds_K, output_root, input_nc_path):
    """
    K Dataset を NetCDF と PNG として保存する。

    - 出力 NetCDF: {output_root}/{case_name}/{元のファイル名}_NZ{NZ}.nc
    - 出力 PNG  : 同じディレクトリ・同じベース名の .png

    Parameters
    ----------
    ds_K : xarray.Dataset
        K の Dataset
    output_root : str
        出力ルートディレクトリ
    input_nc_path : str
        入力 surface forcing NetCDF のパス

    Returns
    -------
    output_nc : str
        保存した NetCDF のパス
    output_png : str
        保存した PNG のパス
    """
    case_name = ds_K.attrs.get("case_name", "case_unknown")

    base_name = os.path.basename(input_nc_path)       # 215.0_40.0_0.0.nc
    base_no_ext, ext = os.path.splitext(base_name)    # 215.0_40.0_0.0, .nc
    base_with_nz = f"{base_no_ext}_NZ{NZ}"

    out_dir = os.path.join(output_root, case_name)
    os.makedirs(out_dir, exist_ok=True)

    output_nc = os.path.join(out_dir, base_with_nz + ext)
    ds_K.to_netcdf(output_nc)

    output_png = os.path.join(out_dir, base_with_nz + ".png")

    # K(t) をプロット（altitude=0 を代表として使用）
    t_sec = ds_K["time"].values.astype(float)
    t_hour = t_sec / 3600.0
    K_1d = ds_K["K"].values[0, :]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_hour, K_1d)
    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Turbulent diffusivity K [m^2/s]")
    ax.set_title(
        "K(t) from surface forcing case: %s"
        % ds_K.attrs.get("case_name", "unknown")
    )

    ax.set_xticks(np.arange(0.0, 25.0, 6.0))
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    return output_nc, output_png


# ======================================================================
# 6. メイン処理
# ======================================================================

def main(debug=False):
    forcing_ds = xr.open_dataset(INPUT_FORCING_NC)

    time_sec, K_1d = compute_K_from_surface_forcing(
        forcing_ds,
        k_min=K_MIN,
        k_max=K_MAX,
    )

    ds_K = make_K_dataset(
        forcing_ds,
        time_sec,
        K_1d,
        source_path=INPUT_FORCING_NC,
        nz=NZ,
        k_min=K_MIN,
        k_max=K_MAX,
    )

    output_nc, output_png = save_K_netcdf_and_png(
        ds_K,
        output_root=K_OUTPUT_ROOT,
        input_nc_path=INPUT_FORCING_NC,
    )

    if debug:
        print("=== DEBUG: generate K-distribution from surface forcing ===")
        print("  INPUT_FORCING_NC :", os.path.abspath(INPUT_FORCING_NC))
        print("  K_OUTPUT_ROOT    :", os.path.abspath(K_OUTPUT_ROOT))
        print("  CASE_NAME        :", ds_K.attrs.get("case_name", "None"))
        print("  LAT [deg]        :", ds_K.attrs.get("latitude_deg", "None"))
        print("  Ls  [deg]        :", ds_K.attrs.get("solar_longitude_deg", "None"))
        print("  theta0_mean      :", ds_K.attrs.get("theta0_surface_mean", "None"))
        print("  theta0_offset    :", ds_K.attrs.get("theta0_offset", "None"))
        print("  dtheta_amp       :", ds_K.attrs.get("dtheta_amp", "None"))
        print("  phase_shift_hours:",
              ds_K.attrs.get("phase_shift_hours", "None"))
        print("  NZ               :", NZ)
        print("  K_MIN, K_MAX     :", K_MIN, K_MAX)
        print("  NIGHT_LATE_START :", NIGHT_LATE_START)
        print("  output_nc        :", output_nc)
        print("  output_png       :", output_png)
        print("  K(t) stats       : min=%.3f, max=%.3f"
              % (float(np.min(K_1d)), float(np.max(K_1d))))
        print("  time steps       :", time_sec.shape[0])


# ======================================================================
# 7. エントリーポイント（--debug のみ）
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate time-dependent diffusivity K(t,z) in phase with "
            "surface forcing (theta_0) from a NetCDF file, "
            "using (almost) the same logic as original generate_K_1dim(), "
            "with early-night plateau extending up to the coldest time."
        )
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print parameters and output paths / K stats.",
    )
    args = parser.parse_args()
    main(debug=args.debug)
