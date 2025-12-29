#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_surface_forcing_parametric.py

斜面流モデルに与える「地表温位強制 θ'(t)（平均0の偏差）」を生成するスクリプト。

仕様（要約）
-----------
- 地表温位は θ(z=0,t) = θ0 + θ'(t) と分解する。
- θ0 は高度方向に一定の基本場で、ケースごとに
      θ0 = BASE_MEAN + THETA0_OFFSET
  で決める。
  - BASE_MEAN : MSL Ls≈90 の基準カーブ GroundTempModel() の時間平均
  - THETA0_OFFSET : 季節・緯度による平均温位のオフセット（手で設定）
- θ'(t) は
      θ'(t) = Δθ · f_unit(t; Δt)
  で与える。
  - Δθ = DTHETA_AMP : 日変化振幅スケール（手で設定）
  - Δt = PHASE_SHIFT_HOURS : 位相シフト（手で設定）
  - f_unit : MSL Ls≈90 の日変化カーブをフーリエ級数（N=5）でフィットし、
             平均0・ピークtoピーク≈1に正規化した形状関数。
- GroundTempModel() の形は全ケース共通（昼夜比などは固定）。
- 季節・緯度ごとの差は θ0, Δθ, Δt にのみ反映する。

保存形式
--------
- NetCDF: {OUTPUT_ROOT}/{CASE_NAME}/{theta0}_{DTHETA_AMP}_{PHASE_SHIFT_HOURS}.nc
  例: surface_forcing_parametric/lat-30.0_Ls90.0/215.0_40.0_0.0.nc
- 変数:
  - theta_0(time): 地表温位偏差（mean=0）
- 属性 (global attrs):
  - theta0_surface_mean : θ0
  - theta0_offset       : THETA0_OFFSET
  - dtheta_amp          : DTHETA_AMP
  - phase_shift_hours   : PHASE_SHIFT_HOURS (Δt)
  - base_mean_from_GroundTempModel : BASE_MEAN
  - latitude_deg        : LAT
  - solar_longitude_deg : LS
  - case_name           : CASE_NAME
  - time_resolution_sec : TIME_RES

付加機能
--------
- NetCDF 保存後に、同じ階層・同じベース名で
  地表温位偏差 θ'(t) の PNG 図を保存する。
  - 横軸: 時間 [hour]（6時間ごとに目盛）
  - 縦軸: 温位偏差 [K]
  - ファイル名: {同じベース名}.png

コマンドライン引数
--------------------
- --debug : パラメータと出力先・Dataset概要を print する。
"""

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt


# ======================================================================
# 1. ユーザーが編集するパラメータ（ケースごとに書き換える）
# ======================================================================

# 緯度 [deg], 太陽黄経 Ls [deg]（メタデータ用）
LAT = -30.0
LS = 90.0

# ケース名（出力ディレクトリ名にも使用）
CASE_NAME = f"lat{LAT}_Ls{LS}"

# θ0 のオフセット [K]
# θ0 = BASE_MEAN + THETA0_OFFSET として計算される。
THETA0_OFFSET = 0.0

# 日変化の振幅スケール Δθ [K]
# 基準カーブの peak-to-peak ≈ 100 K 程度を想定し、その半分 ~50 K が目安。
DTHETA_AMP = 40.0

# 位相シフト Δt [hour]
# 正の値 → 温度カーブ全体が地方時の大きい側へ平行移動。
PHASE_SHIFT_HOURS = 0.0

# 出力ルートディレクトリ
OUTPUT_ROOT = "surface_forcing_parametric"

# 時間分解能 [s]
TIME_RES = 0.1

# 1 sol [s]
DAY_SEC = 24.0 * 3600.0


# ======================================================================
# 2. 基準となる地表温度の日変化（MSL Ls≈90）
#    別のローバーデータを使う場合は、この関数を書き換える。
# ======================================================================

def GroundTempModel():
    """
    1時間ごとの地表温度 [K]。
    これまで使っていた MSL Ls≈90 の橙色カーブをそのまま使用。

    必要に応じて置き換え。
    """
    T = np.array([
        184.0, 183.0, 180.0, 178.5, 177.0, 174.0,
        161.1, 192.0, 219.0, 234.0, 248.0, 259.0,
        260.0, 261.0, 258.5, 252.0, 240.0, 225.0,
        210.0, 198.0, 193.0, 190.0, 189.0, 186.0
    ], dtype=float)
    return T


# ======================================================================
# 3. フーリエ級数（N=5）で日変化の「形」を表現
# ======================================================================

def fourier_series_N5(t_sec, *a):
    """
    フーリエ級数 N=5 で日変化の形を表現する関数。

    Parameters
    ----------
    t_sec : array-like
        時刻 [s], 0〜DAY_SEC
    a : sequence of float
        フーリエ係数 (a0, a1c, a1s, ..., a5c, a5s) 計11個

    Returns
    -------
    array-like
        t_sec に対応する関数値
    """
    result = a[0]
    for n in range(1, 6):
        result += (
            a[2 * n - 1] * np.cos(2.0 * np.pi * n * t_sec / DAY_SEC) +
            a[2 * n]     * np.sin(2.0 * np.pi * n * t_sec / DAY_SEC)
        )
    return result


def fit_base_shape(T_hourly=None):
    """
    基準の日変化（24点）からフーリエ係数 params と、
    peak-to-peak の基準振幅 amp0 を求める。

    Parameters
    ----------
    T_hourly : array-like, shape (24,), optional
        1時間ごとの地表温度 [K]。
        None の場合は GroundTempModel() が使われる。

    Returns
    -------
    params : ndarray, shape (11,)
        フーリエ係数
    amp0 : float
        高分解能上での peak-to-peak 振幅
    """
    if T_hourly is None:
        T_hourly = GroundTempModel()

    T_hourly = np.asarray(T_hourly, dtype=float)
    if T_hourly.shape != (24,):
        raise ValueError("T_hourly must have shape (24,), got %r" % (T_hourly.shape,))

    # 1時間ごとの時刻 [s]
    t_hourly = np.arange(24.0, dtype=float) * 3600.0

    # 平均を引いた偏差（ここでは T≈θ とみなす）
    theta_anom = T_hourly - np.mean(T_hourly)

    # a0 + 5次まで（cos/sin ペアで10個） => 計11個
    initial_guess = [0.0] + [1.0] * 10
    params, _ = curve_fit(fourier_series_N5, t_hourly, theta_anom, p0=initial_guess)

    # 高分解能で peak-to-peak を評価
    t_dense = np.linspace(0.0, DAY_SEC, int(DAY_SEC / 0.5), endpoint=False)
    shape_dense = fourier_series_N5(t_dense, *params)
    amp0 = float(np.max(shape_dense) - np.min(shape_dense))

    return params, amp0


def eval_shape_unit(t_sec, params, amp0, phase_shift_hours=0.0):
    """
    正規化された形 f_unit(t) を返す。
    - 平均 0
    - peak-to-peak ≈ 1（正確には amp0 で割った値）

    Parameters
    ----------
    t_sec : array-like
        時刻 [s]
    params : ndarray
        フーリエ係数
    amp0 : float
        基準 peak-to-peak 振幅
    phase_shift_hours : float
        位相シフト [hour]。正の値で曲線全体が遅いLT側へずれる。

    Returns
    -------
    f_unit : ndarray
        正規化された形状関数
    """
    t_sec = np.asarray(t_sec, dtype=float)
    t_shifted = (t_sec - phase_shift_hours * 3600.0) % DAY_SEC
    theta_anom = fourier_series_N5(t_shifted, *params)
    f_unit = theta_anom / amp0
    return f_unit


# ======================================================================
# 4. NetCDF Dataset を組み立て
# ======================================================================

def makeDataset(
    t_sec,
    theta_anom,
    theta0=None,
    lat=None,
    Ls=None,
    case_name=None,
    time_res=None,
    theta0_offset=None,
    dtheta_amp=None,
    phase_shift_hours=None,
    base_mean=None,
):
    """
    NetCDF 用の xarray.Dataset を作成する。

    Parameters
    ----------
    t_sec : ndarray
        時刻 [s]
    theta_anom : ndarray
        地表温位偏差 θ'(t)（平均0）
    theta0 : float, optional
        基本場の地表平均温位 θ0
    lat : float, optional
        緯度 [deg]
    Ls : float, optional
        太陽黄経 [deg]
    case_name : str, optional
        ケース名
    time_res : float, optional
        時間分解能 [s]
    theta0_offset : float, optional
        θ0 のオフセット（THETA0_OFFSET）
    dtheta_amp : float, optional
        日変化振幅 Δθ（DTHETA_AMP）
    phase_shift_hours : float, optional
        位相シフト Δt（PHASE_SHIFT_HOURS）
    base_mean : float, optional
        GroundTempModel の時間平均 BASE_MEAN

    Returns
    -------
    ds : xarray.Dataset
    """
    t_sec = np.asarray(t_sec, dtype=float)
    theta_anom = np.asarray(theta_anom, dtype=float)

    if t_sec.shape != theta_anom.shape:
        raise ValueError(
            "t_sec and theta_anom must have same shape, got %r and %r"
            % (t_sec.shape, theta_anom.shape)
        )

    data_vars = {
        "theta_0": (
            "time",
            theta_anom,
            {
                "unit": "kelvin",
                "description": "surface potential temperature anomaly (mean=0)",
            },
        )
    }

    coords = {
        "time": (
            "time",
            t_sec,
            {
                "unit": "seconds",
                "description": "time from local midnight",
            },
        )
    }

    attrs = {
        "history": "Created on %s" % datetime.now().isoformat(),
        "reference": "Martinez et al. (2017)",
    }

    if theta0 is not None:
        attrs["theta0_surface_mean"] = float(theta0)
    if theta0_offset is not None:
        attrs["theta0_offset"] = float(theta0_offset)
    if dtheta_amp is not None:
        attrs["dtheta_amp"] = float(dtheta_amp)
    if phase_shift_hours is not None:
        attrs["phase_shift_hours"] = float(phase_shift_hours)
    if base_mean is not None:
        attrs["base_mean_from_GroundTempModel"] = float(base_mean)
    if lat is not None:
        attrs["latitude_deg"] = float(lat)
    if Ls is not None:
        attrs["solar_longitude_deg"] = float(Ls)
    if case_name is not None:
        attrs["case_name"] = str(case_name)
    if time_res is not None:
        attrs["time_resolution_sec"] = float(time_res)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return ds


# ======================================================================
# 5. 出力パス生成：{OUTPUT_ROOT}/{CASE_NAME}/{theta0}_{DTHETA_AMP}_{PHASE_SHIFT_HOURS}.nc
# ======================================================================

def build_output_path(output_root, case_name,
                      theta0, dtheta_amp, phase_shift_hours):
    """
    出力ファイルパスを生成する。

    Parameters
    ----------
    output_root : str
        ルートディレクトリ
    case_name : str
        ケース名（サブディレクトリ名として使用）
    theta0 : float
        基本場の地表平均温位
    dtheta_amp : float
        日変化振幅 Δθ
    phase_shift_hours : float
        位相シフト Δt

    Returns
    -------
    output_nc : str
        出力 NetCDF ファイルパス
    """
    dirname = os.path.join(output_root, case_name)
    os.makedirs(dirname, exist_ok=True)

    filename = "%0.1f_%0.1f_%0.1f.nc" % (theta0, dtheta_amp, phase_shift_hours)
    output_nc = os.path.join(dirname, filename)
    return output_nc


# ======================================================================
# 6. メインの強制生成関数
# ======================================================================

def make_surface_forcing(dtheta_amp, phase_shift_hours,
                         lat=None, Ls=None,
                         case_name=None, time_res=0.1,
                         output_root="surface_forcing_parametric",
                         theta0_offset=0.0):
    """
    θ̄(0,t) = θ0 + θ'(t) のうち、θ'(t) を生成し NetCDF と PNG を保存する。

    Parameters
    ----------
    dtheta_amp : float
        日変化振幅 Δθ [K]
    phase_shift_hours : float
        位相シフト Δt [hour]
    lat : float, optional
        緯度 [deg]
    Ls : float, optional
        太陽黄経 [deg]
    case_name : str, optional
        ケース名
    time_res : float, optional
        時間分解能 [s]
    output_root : str, optional
        出力ルートディレクトリ
    theta0_offset : float, optional
        θ0 のオフセット [K]。θ0 = BASE_MEAN + theta0_offset。

    Returns
    -------
    ds : xarray.Dataset
        生成された Dataset
    output_nc : str
        出力 NetCDF ファイルパス
    output_png : str
        出力 PNG ファイルパス
    theta0 : float
        このケースで用いた θ0
    base_mean : float
        基準カーブ GroundTempModel の時間平均 BASE_MEAN
    """
    # 基準24点から BASE_MEAN を計算
    T_hourly = GroundTempModel()
    base_mean = float(np.mean(T_hourly))
    theta0 = base_mean + float(theta0_offset)

    # 基準 shape のフィット（同じ T_hourly を使用）
    params, amp0 = fit_base_shape(T_hourly=T_hourly)

    # 時間軸（endpoint=False で 0〜24h を time_res 刻み）
    nsteps = int(DAY_SEC / time_res)
    t_sec = np.arange(0, nsteps, dtype=float) * time_res

    # 正規化 shape → 振幅を dtheta_amp にスケーリング
    f_unit = eval_shape_unit(
        t_sec, params, amp0, phase_shift_hours=phase_shift_hours
    )
    theta_anom = dtheta_amp * f_unit  # 平均0の偏差

    # Dataset 作成
    if case_name is None:
        case_name_local = "default"
    else:
        case_name_local = case_name

    ds = makeDataset(
        t_sec,
        theta_anom,
        theta0=theta0,
        lat=lat,
        Ls=Ls,
        case_name=case_name_local,
        time_res=time_res,
        theta0_offset=theta0_offset,
        dtheta_amp=dtheta_amp,
        phase_shift_hours=phase_shift_hours,
        base_mean=base_mean,
    )

    # 出力パス生成 & NetCDF 保存
    output_nc = build_output_path(
        output_root,
        case_name_local,
        theta0,
        dtheta_amp,
        phase_shift_hours,
    )
    ds.to_netcdf(output_nc)

    # ==============================
    # 図の作成と保存（同じ階層・同じベース名）
    # ==============================
    base, _ = os.path.splitext(output_nc)
    output_png = base + ".png"

    # 横軸: 時間[hour], 縦軸: 温位偏差[K]
    t_hour = t_sec / 3600.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_hour, theta_anom)
    ax.set_xlabel("Local time [hour]")
    ax.set_ylabel("Surface potential temperature anomaly [K]")
    ax.set_title("Surface forcing anomaly: %s" % case_name_local)

    # 6時間ごとに目盛り
    ax.set_xticks(np.arange(0.0, 25.0, 6.0))
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return ds, output_nc, output_png, theta0, base_mean


# ======================================================================
# 7. main 関数（--debug 指定時にパラメータなどを print）
# ======================================================================

def main(debug=False):
    ds, output_nc, output_png, theta0, base_mean = make_surface_forcing(
        dtheta_amp=DTHETA_AMP,
        phase_shift_hours=PHASE_SHIFT_HOURS,
        lat=LAT,
        Ls=LS,
        case_name=CASE_NAME,
        time_res=TIME_RES,
        output_root=OUTPUT_ROOT,
        theta0_offset=THETA0_OFFSET,
    )

    if debug:
        print("=== DEBUG: surface forcing parameters ===")
        print("  CASE_NAME          : %s" % CASE_NAME)
        print("  LAT [deg]          : %s" % str(LAT))
        print("  Ls  [deg]          : %s" % str(LS))
        print("  BASE_MEAN [K]      : %.3f" % base_mean)
        print("  THETA0_OFFSET [K]  : %.3f" % THETA0_OFFSET)
        print("  theta0 [K]         : %.3f" % theta0)
        print("  DTHETA_AMP [K]     : %.3f" % DTHETA_AMP)
        print("  PHASE_SHIFT_HOURS  : %.3f" % PHASE_SHIFT_HOURS)
        print("  TIME_RES [s]       : %.3f" % TIME_RES)
        print("  OUTPUT_ROOT        : %s" % OUTPUT_ROOT)
        print("  output_nc          : %s" % output_nc)
        print("  output_png         : %s" % output_png)
        print("--- Dataset summary ---")
        print(ds)


# ======================================================================
# 8. エントリーポイント：コマンドライン引数は --debug のみ
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate parametric surface forcing (theta_0(t)) "
                    "for a slope-wind model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print parameters and output path / dataset summary."
    )
    args = parser.parse_args()
    main(debug=args.debug)
