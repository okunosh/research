"""
make_K_from_surface_forcing_MCD.py

MCD から作成した surface forcing (theta_surf) の NetCDF を入力として，
位相同期した乱流拡散係数 K(z, t) を作成するスクリプト／モジュール。

- 入力:  surface_forcing_from_MCD/ にある theta_surf(time) を含む NetCDF
- 出力:  K_MCD/ 以下に K(z, t) を保存した NetCDF
         併せて θ̄(0,t) と地表面 K の日変化を比較する図を保存

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse


# ============================================================
# コア計算部分
# ============================================================


def create_K_dataset_from_surface_forcing(
    ds_theta: xr.Dataset,
    nz: int,
    K_min: float,
    K_max: float,
    shape_power: float = 1.0,
    *,
    theta_var_name: str = "theta_surf",
    K_var_name: str = "K",
    sol_sec: float = 86400.0,
) -> xr.Dataset:
    """
    surface forcing Dataset (theta_surf(time)) から K(z, t) の Dataset を作成する。

    ds_theta[theta_var_name] には「日平均からの偏差（anomaly）」または
    絶対温度が入っていることを想定するが、どちらの場合でも
    min-max 正規化のみを行うので結果は同一になる。

    Parameters
    ----------
    ds_theta : xr.Dataset
        theta_surf(time) を含む Dataset。
    nz : int
        高度方向の格子点数。K をこの数だけ高さ方向にコピーする。
    K_min, K_max : float
        K(t) の最小値・最大値 [m^2/s]。
    shape_power : float, default 1.0
        theta_surf の正規化値を K に写像する際のべき指数。
        1.0 なら線形，>1 で昼間を強調，<1 で昼夜の差を弱める。
    theta_var_name : str, default "theta_surf"
        surface forcing の変数名。
    K_var_name : str, default "K"
        出力 Dataset での K の変数名。
    sol_sec : float, default 86400.0
        1 sol の秒数（ここでは主に属性用）。
    """
    if theta_var_name not in ds_theta:
        raise KeyError(f"入力 Dataset に変数 '{theta_var_name}' が見つかりません。")
    if "time" not in ds_theta.coords:
        raise KeyError("入力 Dataset に 'time' 座標が見つかりません。")

    if nz <= 0:
        raise ValueError("nz は 1 以上の整数である必要があります。")

    if shape_power <= 0.0:
        raise ValueError("shape_power は 0 より大きい値である必要があります。")

    time_sec = ds_theta["time"].to_numpy()
    theta = ds_theta[theta_var_name].to_numpy().astype(float)

    # ここでは mean は使わず、min-max 正規化のみを行う
    th_min = float(theta.min())
    th_max = float(theta.max())

    if th_max - th_min <= 0.0:
        # 変化が無い場合は一律 K_min
        s = np.zeros_like(theta)
    else:
        s = (theta - th_min) / (th_max - th_min)

    # 形状調整
    if shape_power != 1.0:
        s = np.power(s, shape_power)

    # K(t) の生成
    K_t = K_min + s * (K_max - K_min)

    # 高度方向にコピーして K(z, t) を作成
    z = np.arange(nz, dtype=float)
    K_zt = np.repeat(K_t[np.newaxis, :], nz, axis=0)

    ds_K = xr.Dataset(
        data_vars={
            K_var_name: (("z", "time"), K_zt),
        },
        coords={
            "z": z,
            "time": time_sec,
        },
        attrs={
            "K_min": float(K_min),
            "K_max": float(K_max),
            "shape_power": float(shape_power),
            "theta_source_var": theta_var_name,
            "sol_sec": float(sol_sec),
        },
    )

    return ds_K


def save_K_dataset(
    ds_K: xr.Dataset,
    output_nc_path: Union[str, Path],
) -> None:
    """K(z, t) の Dataset を NetCDF として保存する。"""
    out_path = Path(output_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds_K.data_vars}
    ds_K.to_netcdf(out_path, encoding=enc)
    print(f"[INFO] Saved K NetCDF to: {out_path}")


def plot_K_and_theta(
    ds_theta: xr.Dataset,
    ds_K: xr.Dataset,
    sol_sec: float,
    output_png_path: Union[str, Path],
    *,
    theta_var_name: str = "theta_surf",
    K_var_name: str = "K",
) -> None:
    """
    θ̄(0,t)（偏差）と地表面 K の日変化を同じ図に描画して保存する。

    - x 軸: Local time [hour] (0〜24, 6 h ごとの目盛り)
    - 左 y 軸: θ̄(0,t) anomaly [K]（赤破線）
    - 右 y 軸: K_surface [m^2/s]（黒実線）
    """
    if theta_var_name not in ds_theta:
        raise KeyError(f"入力 Dataset に変数 '{theta_var_name}' が見つかりません。")
    if K_var_name not in ds_K:
        raise KeyError(f"入力 Dataset に変数 '{K_var_name}' が見つかりません。")

    time_sec = ds_theta["time"].to_numpy()
    theta = ds_theta[theta_var_name].to_numpy().astype(float)

    # ここからは「theta は既に偏差（anomaly）」という前提でそのまま描画する
    theta_anom = theta

    # 地表面の K (z=0 を代表として使用)
    K_surface = ds_K[K_var_name].isel(z=0).to_numpy().astype(float)

    # 地方時 [hour]
    lt_hour = time_sec / sol_sec * 24.0

    fig, ax_theta = plt.subplots(figsize=(6, 4))

    # θ̄(0,t)（左軸）
    line_theta, = ax_theta.plot(
        lt_hour, theta_anom, linestyle="--", color="red", label=r"$\bar{\theta}(0,t)$"
    )
    ax_theta.set_xlim(0.0, 24.0)
    ax_theta.set_xticks(np.arange(0, 25, 6))
    ax_theta.set_xlabel("Local time (hour)")
    ax_theta.set_ylabel(r"$\bar{\theta}(0,t)$ anomaly [K]", color="red")
    ax_theta.tick_params(axis="y", labelcolor="red")
    ax_theta.grid(True, linestyle="--", alpha=0.5)

    # K（右軸）
    ax_K = ax_theta.twinx()
    line_K, = ax_K.plot(
        lt_hour, K_surface, color="black", label="K", alpha=0.6
    )
    ax_K.set_ylabel("K [m$^2$/s]", color="black")
    ax_K.tick_params(axis="y", labelcolor="black")

    # タイトルと凡例
    ax_theta.set_title("Diurnal distribution of $\\bar{\\theta}(0,t)$ and K")
    lines = [line_theta, line_K]
    labels = [line.get_label() for line in lines]
    ax_theta.legend(lines, labels, loc="upper right")

    out_png = Path(output_png_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved K & theta plot to: {out_png}")


# ============================================================
# CLI インターフェース
# ============================================================


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MCD 由来の surface forcing (theta_surf) から "
            "位相同期した拡散係数 K(z,t) を作成するスクリプト"
        )
    )

    parser.add_argument(
        "theta_nc",
        help="入力 surface forcing NetCDF のパス "
        "(例: surface_forcing_from_MCD/Ls90_Lat0_Long260_Alt0_fourierN6.nc)",
    )
    parser.add_argument(
        "--nz",
        type=int,
        required=True,
        help="高度方向の格子点数 nz（K を高さ方向にコピーする回数）",
    )
    parser.add_argument(
        "--K-min",
        type=float,
        required=True,
        help="K(t) の最小値 [m^2/s]",
    )
    parser.add_argument(
        "--K-max",
        type=float,
        required=True,
        help="K(t) の最大値 [m^2/s]",
    )
    parser.add_argument(
        "--shape-power",
        type=float,
        default=1.0,
        help="theta_surf 正規化値から K への写像に用いるべき指数 p [default: 1.0]",
    )
    parser.add_argument(
        "--sol-sec",
        type=float,
        default=86400.0,
        help="1 sol の秒数 [default: 86400.0]",
    )
    parser.add_argument(
        "--out-nc",
        type=str,
        default=None,
        help="出力 K NetCDF のパス（省略時は入力ファイル名に基づき自動決定）",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=None,
        help="出力 図 (png) のパス（省略時は入力ファイル名に基づき自動決定）",
    )

    return parser.parse_args(argv)


def generate_K_from_surface_forcing_MCD(
    surface_nc_path: Union[str, Path],
    output_nc_path: Union[str, Path],
    output_png_path: Union[str, Path],
    nz: int,
    K_min: float,
    K_max: float,
    shape_power: float = 1.0,
    *,
    theta_var_name: str = "theta_surf",
    K_var_name: str = "K",
    sol_sec: float = 86400.0,
) -> None:
    """
    surface forcing NetCDF から K(z, t) を作成し，
    NetCDF + 図 を保存する高レベル関数。
    """
    surface_nc_path = Path(surface_nc_path)
    if not surface_nc_path.is_file():
        raise FileNotFoundError(f"surface forcing NetCDF が見つかりません: {surface_nc_path}")

    # 入力 Dataset を開く
    ds_theta = xr.open_dataset(surface_nc_path)

    try:
        ds_K = create_K_dataset_from_surface_forcing(
            ds_theta=ds_theta,
            nz=nz,
            K_min=K_min,
            K_max=K_max,
            shape_power=shape_power,
            theta_var_name=theta_var_name,
            K_var_name=K_var_name,
            sol_sec=sol_sec,
        )

        save_K_dataset(ds_K, output_nc_path=output_nc_path)
        plot_K_and_theta(
            ds_theta=ds_theta,
            ds_K=ds_K,
            sol_sec=sol_sec,
            output_png_path=output_png_path,
            theta_var_name=theta_var_name,
            K_var_name=K_var_name,
        )
    finally:
        ds_theta.close()


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    theta_nc_path = Path(args.theta_nc)
    if not theta_nc_path.is_file():
        raise FileNotFoundError(f"入力 NetCDF が見つかりません: {theta_nc_path}")

    stem = theta_nc_path.stem  # 例: "Ls90_Lat0_Long260_Alt0_fourierN6"
    base_dir = theta_nc_path.parent.parent  # surface_forcing_from_MCD/
    K_dir = base_dir.parent / "K_MCD"
    K_dir.mkdir(parents=True, exist_ok=True)

    if args.out_nc is None:
        output_nc_path = K_dir / f"{stem}.nc"
    else:
        output_nc_path = Path(args.out_nc)

    if args.out_fig is None:
        output_png_path = K_dir / f"{stem}_K_theta.png"
    else:
        output_png_path = Path(args.out_fig)

    generate_K_from_surface_forcing_MCD(
        surface_nc_path=theta_nc_path,
        output_nc_path=output_nc_path,
        output_png_path=output_png_path,
        nz=args.nz,
        K_min=args.K_min,
        K_max=args.K_max,
        shape_power=args.shape_power,
        theta_var_name="theta_surf",
        K_var_name="K",
        sol_sec=args.sol_sec,
    )


if __name__ == "__main__":
    main()
