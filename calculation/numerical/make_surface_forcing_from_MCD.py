"""
make_surface_forcing_from_MCD.py

- MCD の ASCII テキスト (local time [h], surface temperature [K]) を読む
- 0.1 s 刻みの theta_surf(time) を作る
  - デフォルトでは、1日周期フーリエ級数で平滑化してからサンプリング
- NetCDF として surface_forcing_from_MCD/ に保存
  - ファイル名: <MCDテキストのベース名>_fourierN6.nc など
- 平均からの偏差（温位偏差）の図を保存
  - ファイル名: <MCDテキストのベース名>_fourierN6_theta_anomaly.png

【重要】
  - NetCDF に保存する変数 theta_surf(time) は
    「日平均からの偏差（anomaly）」を保存する。
  - global attributes に日平均値 theta_surf_mean / theta0_surface_mean を保存することで、
    元の絶対温度（潜在温位）も再構成できる。

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# MCD ASCII 読み込み
# ============================================================


def load_lt_T_from_mcd_txt(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    MCD の ASCII テキスト
    (local time [h], surface temperature [K]) を読み込む。

    返り値
    ------
    lt_hour : np.ndarray
        Local time [hour] (0〜24)
    Tsurf : np.ndarray
        地表温度 [K]
    """
    path = Path(path)
    lt_list: list[float] = []
    T_list: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 2:
                continue

            try:
                lt = float(parts[0])
                T = float(parts[1])
            except ValueError as e:
                raise ValueError(f"数値に変換できない行があります: {line!r}") from e

            lt_list.append(lt)
            T_list.append(T)

    if not lt_list:
        raise ValueError(f"MCD テキスト {path} に有効なデータ行がありません。")

    lt_hour = np.asarray(lt_list, dtype=float)
    Tsurf = np.asarray(T_list, dtype=float)

    return lt_hour, Tsurf


# ============================================================
# フーリエ級数フィット
# ============================================================


def fit_fourier_series(
    lt_hour: np.ndarray,
    Tsurf: np.ndarray,
    n_harmonics: int = 6,
):
    """
    1日周期のフーリエ級数で T(lt) をフィットする簡単な関数。

    T(lt) ≒ a0 + Σ [ak cos(2πk lt/24) + bk sin(2πk lt/24)]

    を最小二乗で求め，model(lt) として返す。
    """
    lt_hour = np.asarray(lt_hour, dtype=float)
    Tsurf = np.asarray(Tsurf, dtype=float)

    # デザイン行列 A を作る
    lt_rad = 2.0 * np.pi * lt_hour / 24.0  # [rad]
    cols = [np.ones_like(lt_rad)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * lt_rad))
        cols.append(np.sin(k * lt_rad))

    A = np.vstack(cols).T  # shape: (N, 1 + 2*n_harmonics)

    # 最小二乗
    coef, *_ = np.linalg.lstsq(A, Tsurf, rcond=None)

    def model(t_hour: Union[np.ndarray, float]):
        t_hour_arr = np.asarray(t_hour, dtype=float)
        t_rad = 2.0 * np.pi * t_hour_arr / 24.0
        cols_m = [np.ones_like(t_rad)]
        for k in range(1, n_harmonics + 1):
            cols_m.append(np.cos(k * t_rad))
            cols_m.append(np.sin(k * t_rad))
        A_m = np.vstack(cols_m).T
        return A_m @ coef

    return model


# ============================================================
# 補間 & Dataset 作成
# ============================================================


def build_time_axis(sol_sec: float = 86400.0, dt: float = 0.1) -> np.ndarray:
    """0〜sol_sec の 0.1 s 刻み time 軸を作る。"""
    nstep = int(sol_sec / dt)
    return np.linspace(0.0, sol_sec, nstep, endpoint=False)


def interpolate_diurnal_cycle(
    lt_hour: np.ndarray,
    Tsurf: np.ndarray,
    sol_sec: float = 86400.0,
    dt: float = 0.1,
    method: str = "fourier",
    n_harmonics: int = 6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    MCD の (lt_hour, Tsurf) → 0.1 s 刻みの日周期データへ補間する。

    返り値
    ------
    t_target : np.ndarray
        time [s] 軸 (0〜sol_sec, dt 刻み)
    theta_anom : np.ndarray
        「日平均からの偏差」 θ̄(0,t) [K]
    theta_mean : float
        日平均値 [K] （元の Tsurf の平均）
    """
    lt_hour = np.asarray(lt_hour, dtype=float)
    Tsurf = np.asarray(Tsurf, dtype=float)
    t_target = build_time_axis(sol_sec=sol_sec, dt=dt)

    theta_mean = float(Tsurf.mean())

    if method == "linear":
        # LT[h] → time[s]
        t_src = lt_hour / 24.0 * sol_sec
        t_src_ext = np.concatenate([t_src, t_src + sol_sec])
        T_ext = np.concatenate([Tsurf, Tsurf])
        T_interp = np.interp(t_target, t_src_ext, T_ext)
    elif method == "fourier":
        model = fit_fourier_series(lt_hour, Tsurf, n_harmonics=n_harmonics)
        lt_target = t_target / sol_sec * 24.0
        T_interp = model(lt_target)
    else:
        raise ValueError(f"未知の補間方法です: {method!r} (fourier or linear)")

    theta_anom = T_interp - theta_mean
    return t_target, theta_anom, theta_mean


def create_surface_forcing_dataset_from_mcd(
    lt_hour: np.ndarray,
    Tsurf: np.ndarray,
    sol_sec: float = 86400.0,
    dt: float = 0.1,
    var_name: str = "theta_surf",
    interp_method: str = "fourier",
    n_harmonics: int = 6,
) -> Tuple[xr.Dataset, np.ndarray, float]:
    """
    MCD データから surface forcing Dataset (theta_surf anomaly) を作成する。

    戻り値
    ------
    ds : xr.Dataset
        theta_surf(time) を含む Dataset。theta_surf は「平均からの偏差」。
    t_target : np.ndarray
        time [s] 軸
    theta_mean : float
        日平均の絶対温度 [K]
    """
    t_target, theta_anom, theta_mean = interpolate_diurnal_cycle(
        lt_hour=lt_hour,
        Tsurf=Tsurf,
        sol_sec=sol_sec,
        dt=dt,
        method=interp_method,
        n_harmonics=n_harmonics,
    )

    time_coord = t_target

    ds = xr.Dataset(
        data_vars={
            var_name: ("time", theta_anom.astype(float)),
        },
        coords={"time": time_coord},
        attrs={
            f"{var_name}_mean": float(theta_mean),
            "theta0_surface_mean": float(theta_mean),
            "description": (
                "theta_surf は MCD の地表温度から日平均を引いた偏差（anomaly）。"
            ),
            "interp_method": interp_method,
            "n_harmonics": int(n_harmonics),
            "dt": float(dt),
            "sol_sec": float(sol_sec),
        },
    )

    return ds, t_target, theta_mean


# ============================================================
# 保存 & プロット
# ============================================================


def save_surface_forcing_dataset(
    ds: xr.Dataset,
    output_nc_path: Union[str, Path],
) -> None:
    """surface_forcing Dataset を NetCDF として保存する。"""
    out_path = Path(output_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=enc)
    print(f"[INFO] Saved surface forcing NetCDF to: {out_path}")


def plot_surface_forcing_with_raw(
    lt_hour: np.ndarray,
    Tsurf: np.ndarray,
    t_target: np.ndarray,
    theta_anom_interp: np.ndarray,
    theta_mean: float,
    sol_sec: float,
    output_png_path: Union[str, Path],
) -> None:
    """
    MCD の元データ（raw）と、補間後の時系列を
    どちらも「日平均からの偏差」として描画する。

    - 横軸: Local time [hour]
    - 縦軸: θ̄(0,t) [K] （平均からの偏差）
    - 線: 補間結果 "Interpolated"
    - 点: MCD 生データ "MCD (raw)"
    """
    # raw データの偏差
    theta_raw_anom = Tsurf - theta_mean

    # 時間 → 地方時 [hour]
    lt_interp = t_target / sol_sec * 24.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lt_interp, theta_anom_interp, label="Interpolated")
    ax.plot(lt_hour, theta_raw_anom, "o", label="MCD (raw)")

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xlabel("Local time (hour)")
    ax.set_ylabel(r"$\bar{\theta}(0,t)$ [K]")
    ax.set_title("Surface forcing")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    out_png = Path(output_png_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved forcing plot to: {out_png}")


# ============================================================
# 一括実行用（他ファイルから import して呼んでもOK）
# ============================================================


def generate_surface_forcing_from_mcd(
    mcd_txt_path: Union[str, Path],
    output_nc_path: Union[str, Path],
    output_png_path: Union[str, Path],
    sol_sec: float = 86400.0,
    dt: float = 0.1,
    var_name: str = "theta_surf",
    interp_method: str = "fourier",
    n_harmonics: int = 6,
) -> None:
    """
    1) MCD テキスト → surface_forcing Dataset 作成（theta_surf = anomaly）
    2) NetCDF 保存
    3) 補間＋MCD raw の偏差の図を保存

    という一連の処理をまとめた関数。

    他のスクリプト（make_surface_forcing_and_K_from_MCD.py など）から
    import して使うことを想定している。
    """
    # まず MCD テキストを読み込む
    lt_hour, Tsurf = load_lt_T_from_mcd_txt(mcd_txt_path)

    # Dataset を作成（内部で補間も実施）
    ds, t_target, theta_mean = create_surface_forcing_dataset_from_mcd(
        lt_hour=lt_hour,
        Tsurf=Tsurf,
        sol_sec=sol_sec,
        dt=dt,
        var_name=var_name,
        interp_method=interp_method,
        n_harmonics=n_harmonics,
    )

    # NetCDF として保存
    save_surface_forcing_dataset(ds, output_nc_path=output_nc_path)

    # 図を作成（補間 + MCD raw の偏差）
    theta_anom_interp = ds[var_name].to_numpy().astype(float)
    plot_surface_forcing_with_raw(
        lt_hour=lt_hour,
        Tsurf=Tsurf,
        t_target=t_target,
        theta_anom_interp=theta_anom_interp,
        theta_mean=theta_mean,
        sol_sec=sol_sec,
        output_png_path=output_png_path,
    )
