"""make_surface_forcing_ver4_and_K_from_MCD.py

MCD ASCII テキストを入力として、ver4 の surface forcing（theta_surf）の NetCDF + 図 と、
それと位相同期した拡散係数 K(z,t) の NetCDF + 図 を一括で生成する。

既存の make_surface_forcing_and_K_from_MCD.py は（従来版）を呼ぶ用途として残し、
このスクリプトは make_surface_forcing_from_MCD_ver4.py を利用する。

依存:
- make_surface_forcing_from_MCD_ver4.py
    - generate_surface_forcing_from_mcd_ver4(...)
- make_K_from_surface_forcing_MCD.py
    - generate_K_from_surface_forcing_MCD(...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

from make_K_from_surface_forcing_MCD import generate_K_from_surface_forcing_MCD
from make_surface_forcing_from_MCD_ver4 import generate_surface_forcing_from_mcd_ver4


def _float_token(x: float, *, digits: int = 6) -> str:
    s = f"{float(x):.{int(digits)}g}"
    return s.replace("-", "m").replace(".", "p")


def _build_ver4_suffix(
    *,
    half_window: int,
    dt_hour_plot: float,
    set_frac: float,
    n_harmonics: int,
    fit_to: str,
    morning_monotone: bool,
    dt: float,
) -> str:
    """ver4 用のファイル名サフィックス。

    forcing と K の対応が崩れないよう、同一 suffix を必ず共有する。
    """

    return (
        "_ver4"
        f"_hw{int(half_window)}"
        f"_dth{_float_token(float(dt_hour_plot))}"
        f"_sf{_float_token(float(set_frac))}"
        f"_N{int(n_harmonics)}"
        f"_fit{fit_to}"
        f"_morn{int(bool(morning_monotone))}"
        f"_dt{_float_token(float(dt))}"
    )


def generate_surface_forcing_ver4_and_K_from_MCD(
    mcd_txt_path: Union[str, Path],
    *,
    # ver4 forcing parameters
    half_window: int = 1,
    dt_hour_plot: float = 0.01,
    set_frac: float = 0.4,
    fourier_harmonics: int = 6,
    fit_to: str = "raw",
    morning_monotone: bool = False,
    # NetCDF sampling parameters
    dt: float = 0.1,
    sol_sec: float = 86400.0,
    # K parameters
    nz: int = 0,
    K_min: float = 0.0,
    K_max: float = 0.0,
    shape_power: float = 1.0,
    # output
    surface_dir: Union[str, Path] = "surface_forcing_from_MCD",
    K_dir: Union[str, Path] = "K_MCD",
) -> None:
    mcd_txt_path = Path(mcd_txt_path)
    if not mcd_txt_path.is_file():
        raise FileNotFoundError(f"MCD テキストファイルが見つかりません: {mcd_txt_path}")

    if fit_to not in {"raw", "smoothed"}:
        raise ValueError("fit_to must be 'raw' or 'smoothed'")

    if nz <= 0:
        raise ValueError("nz は 1 以上の整数である必要があります。")

    if shape_power <= 0.0:
        raise ValueError("shape_power は 0 より大きい値である必要があります。")

    stem = mcd_txt_path.stem
    suffix = _build_ver4_suffix(
        half_window=half_window,
        dt_hour_plot=dt_hour_plot,
        set_frac=set_frac,
        n_harmonics=fourier_harmonics,
        fit_to=fit_to,
        morning_monotone=morning_monotone,
        dt=dt,
    )

    surface_dir = Path(surface_dir)
    K_dir = Path(K_dir)

    surface_nc_path = surface_dir / f"{stem}{suffix}.nc"
    surface_fig_path = surface_dir / f"{stem}{suffix}_theta_anomaly.png"

    # 1) ver4 forcing
    generate_surface_forcing_from_mcd_ver4(
        mcd_txt_path=str(mcd_txt_path),
        output_nc_path=str(surface_nc_path),
        output_png_path=str(surface_fig_path),
        dt=float(dt),
        sol_sec=float(sol_sec),
        half_window=int(half_window),
        set_frac=float(set_frac),
        fourier_harmonics=int(fourier_harmonics),
        fit_to=str(fit_to),
        morning_monotone=bool(morning_monotone),
        dt_hour_plot=float(dt_hour_plot),
    )

    # 2) K (phase-synced)
    K_nc_path = K_dir / f"{stem}{suffix}.nc"
    K_fig_path = K_dir / f"{stem}{suffix}_K_theta.png"

    generate_K_from_surface_forcing_MCD(
        surface_nc_path=surface_nc_path,
        output_nc_path=K_nc_path,
        output_png_path=K_fig_path,
        nz=int(nz),
        K_min=float(K_min),
        K_max=float(K_max),
        shape_power=float(shape_power),
        sol_sec=float(sol_sec),
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MCD ASCII テキストから ver4 surface forcing (theta_surf) と "
            "位相同期した K(z,t) をまとめて作成するスクリプト"
        )
    )

    p.add_argument(
        "mcd_txt",
        help="入力となる MCD ASCII テキストファイルのパス (例: MCDdata/Ls90_Lat0_Lon260_Alt0.txt)",
    )

    # ver4 forcing
    p.add_argument("--half-window", type=int, default=1, help="smoothing half-window (window=2*hw+1)")
    p.add_argument("--dt-hour-plot", type=float, default=0.01, help="ver4 内部の評価/図用の dt [hour]")
    p.add_argument("--set-frac", type=float, default=0.4, help="cool_fraction for t_set")
    p.add_argument("--fourier-harmonics", type=int, default=6, help="Fourier harmonics N")
    p.add_argument("--fit-to", choices=["raw", "smoothed"], default="raw", help="points to fit")
    p.add_argument("--morning-monotone", action="store_true", help="enforce morning non-decreasing")

    # NetCDF sampling
    p.add_argument("--dt", type=float, default=0.1, help="forcing NetCDF の time 刻み [s]")
    p.add_argument("--sol-sec", type=float, default=86400.0, help="1 sol の秒数")

    # K
    p.add_argument("--nz", type=int, required=True, help="K(z,t) の高度方向格子点数 nz")
    p.add_argument("--K-min", type=float, required=True, help="拡散係数 K の最小値 [m^2/s]")
    p.add_argument("--K-max", type=float, required=True, help="拡散係数 K の最大値 [m^2/s]")
    p.add_argument("--shape-power", type=float, default=1.0, help="K 形状の指数 p")

    # output dirs
    p.add_argument("--surface-dir", type=str, default="surface_forcing_from_MCD", help="forcing 出力ディレクトリ")
    p.add_argument("--K-dir", type=str, default="K_MCD", help="K 出力ディレクトリ")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    generate_surface_forcing_ver4_and_K_from_MCD(
        mcd_txt_path=args.mcd_txt,
        half_window=args.half_window,
        dt_hour_plot=args.dt_hour_plot,
        set_frac=args.set_frac,
        fourier_harmonics=args.fourier_harmonics,
        fit_to=args.fit_to,
        morning_monotone=bool(args.morning_monotone),
        dt=args.dt,
        sol_sec=args.sol_sec,
        nz=args.nz,
        K_min=args.K_min,
        K_max=args.K_max,
        shape_power=args.shape_power,
        surface_dir=args.surface_dir,
        K_dir=args.K_dir,
    )


if __name__ == "__main__":
    main()
