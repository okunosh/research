"""
make_forcing_and_K_parametric.py

1. 地表温位強制 θ'(t)（theta_0(t)）を生成して NetCDF / PNG を保存
2. その NetCDF を入力として、位相同期した K(t,z) を生成して NetCDF / PNG を保存

までを **1回の実行で行うラッパースクリプト**。

ケースごとのパラメータは、このファイル先頭の定数を編集して切り替える想定。
make_surface_forcing_parametric.py / generate_K_from_surface_frocing_parametric.py
側は編集不要。
"""

import os
import argparse

import xarray as xr

import make_surface_forcing_parametric as sf_mod
import generate_K_from_surface_frocing_parametric as k_mod


# ======================================================================
# 1. ケースごとのパラメータ設定（ここだけ書き換えればよい）
# ======================================================================

# --- 地表温位強制（theta_0）のパラメータ ---

LAT = 0.0          # [deg]  緯度
LS = 90.0            # [deg]  太陽黄経
CASE_NAME = f"lat{LAT:+.1f}_Ls{LS:.1f}"

# θ0 = BASE_MEAN + THETA0_OFFSET で決める
THETA0_OFFSET = 0.       # [K]   季節・緯度による平均温位のオフセット
DTHETA_AMP = 97.4         # [K]   日変化振幅 Δθ
PHASE_SHIFT_HOURS = 0.   # [hour] 位相シフト Δt
TIME_RES = 0.1            # [s]   出力 NetCDF の時間分解能
FORCING_OUTPUT_ROOT = "surface_forcing_parametric"

# --- K のパラメータ ---

NZ = 260 + 1              # 高度方向の格子点数（0〜780 m を 1 m 刻みなど）
K_MIN = 3.0               # [m^2/s]
K_MAX = 100.0             # [m^2/s]
K_OUTPUT_ROOT = "K_parametric"


# ======================================================================
# 2. メインの処理
# ======================================================================

def run_all(debug: bool = False):
    """
    1. surface forcing θ'(t)（theta_0(t)）を生成
    2. その NetCDF を用いて K(t,z) を生成

    Returns
    -------
    ds_sf : xarray.Dataset
        surface forcing Dataset
    sf_nc : str
        surface forcing NetCDF ファイルパス
    sf_png : str
        surface forcing PNG ファイルパス
    theta0 : float
        ケースの θ0
    base_mean : float
        GroundTempModel の時間平均
    ds_K : xarray.Dataset
        K(t,z) の Dataset
    K_nc : str
        K NetCDF ファイルパス
    K_png : str
        K PNG ファイルパス
    """
    # --------------------------------------------------
    # 1. surface forcing の作成
    # --------------------------------------------------
    ds_sf, sf_nc, sf_png, theta0, base_mean = sf_mod.make_surface_forcing(
        dtheta_amp=DTHETA_AMP,
        phase_shift_hours=PHASE_SHIFT_HOURS,
        lat=LAT,
        Ls=LS,
        case_name=CASE_NAME,
        time_res=TIME_RES,
        output_root=FORCING_OUTPUT_ROOT,
        theta0_offset=THETA0_OFFSET,
    )

    # --------------------------------------------------
    # 2. K の作成（強制 NetCDF を入力）
    #    generate_K_from_surface_frocing_parametric.py の
    #    下位関数を使って作成する。
    # --------------------------------------------------
    forcing_ds = xr.open_dataset(sf_nc)

    # θ0(t) → K_base(t) → 時刻依存 K(t) を作成
    time_sec, K_1d = k_mod.compute_K_from_surface_forcing(
        forcing_ds,
        k_min=K_MIN,
        k_max=K_MAX,
    )

    # K(t) を高度方向に NZ コピーして Dataset 化
    ds_K = k_mod.make_K_dataset(
        forcing_ds,
        time_sec,
        K_1d,
        source_path=sf_nc,
        nz=NZ,
        k_min=K_MIN,
        k_max=K_MAX,
    )

    # NetCDF & PNG 保存（ファイル名末尾に _NZ{NZ} を付ける仕様）
    # ※ generate_K_from_surface_frocing_parametric.py 側の
    #    save_K_netcdf_and_png() を利用
    K_nc, K_png = k_mod.save_K_netcdf_and_png(
        ds_K,
        output_root=K_OUTPUT_ROOT,
        input_nc_path=sf_nc,
    )

    if debug:
        print("=== Surface forcing (theta_0) ===")
        print(f"  CASE_NAME          : {CASE_NAME}")
        print(f"  LAT [deg]          : {LAT}")
        print(f"  Ls  [deg]          : {LS}")
        print(f"  BASE_MEAN [K]      : {base_mean:.3f}")
        print(f"  THETA0_OFFSET [K]  : {THETA0_OFFSET:.3f}")
        print(f"  theta0 [K]         : {theta0:.3f}")
        print(f"  DTHETA_AMP [K]     : {DTHETA_AMP:.3f}")
        print(f"  PHASE_SHIFT_HOURS  : {PHASE_SHIFT_HOURS:.3f}")
        print(f"  TIME_RES [s]       : {TIME_RES:.3f}")
        print(f"  OUTPUT_ROOT        : {FORCING_OUTPUT_ROOT}")
        print(f"  surface forcing nc : {sf_nc}")
        print(f"  surface forcing png: {sf_png}")
        print()
        print("  --- Dataset summary (theta_0) ---")
        print(ds_sf)
        print()

        print("=== K(t,z) ===")
        print(f"  NZ                 : {NZ}")
        print(f"  K_MIN, K_MAX       : {K_MIN}, {K_MAX}")
        print(f"  K_OUTPUT_ROOT      : {K_OUTPUT_ROOT}")
        print(f"  K nc               : {K_nc}")
        print(f"  K png              : {K_png}")
        print()
        print("  --- Dataset summary (K) ---")
        print(ds_K)

    return ds_sf, sf_nc, sf_png, theta0, base_mean, ds_K, K_nc, K_png


# ======================================================================
# 3. エントリーポイント
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parametric surface forcing θ'(t) と "
            "位相同期した K(t,z) を一度に生成するスクリプト"
        )
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="パラメータと出力パス Dataset 概要を print する",
    )
    args = parser.parse_args()

    run_all(debug=args.debug)


if __name__ == "__main__":
    main()
