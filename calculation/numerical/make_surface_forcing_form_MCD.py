#!/usr/bin/env python3
"""
make_surface_forcing_from_MCD.py

- MCD の ASCII テキスト (local time [h], surface temperature [K]) を読む
- 0.1 s 刻みの theta_0(time) を作る
  - デフォルトでは、1日周期フーリエ級数で平滑化してからサンプリング
- NetCDF として surface_forcing_from_MCD/ に保存
  - ファイル名: <MCDテキストのベース名>.nc
- 平均からの偏差（温位偏差）の図を保存
  - ファイル名: <MCDテキストのベース名>_theta_anomaly.png

【重要】
  NetCDF に保存する変数 theta_0(time) は
  「日平均からの偏差（anomaly）」を保存する。
  一方で、global attributes に日平均値 theta0_surface_mean などを保存することで、
  元の絶対温度（潜在温位）も再構成できるようにしている。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# 基本ユーティリティ
# ============================================================

def build_time_axis(sol_sec: float = 86400.0, dt: float = 0.1) -> np.ndarray:
    """1 sol=sol_sec [s], 時間刻み dt [s] の time 配列を作る。"""
    n = int(sol_sec / dt)
    t = np.arange(n, dtype=float) * dt  # [s] 0 〜 sol_sec - dt
    return t


# ============================================================
# MCD テキスト読み込み
# ============================================================

def load_lt_T_from_mcd_txt(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    MCD の ASCII テキスト（ヘッダが # で始まり、その後 2 列の数値）の
    ファイルから local time [h], Temperature [K] を読み込む。

    データ部の例:
        0.00000e+00    1.60371e+02
        1.00000e+00    1.60349e+02
        ...
        2.40000e+01    1.60371e+02
    """
    path = Path(path)
    lt_list = []
    T_list = []

    with path.open("r") as f:
        for line in f:
            s = line.strip()
            # 空行・コメント行をスキップ
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
        raise ValueError(f"有効なデータ行が見つかりませんでした: {path}")

    lt_hour = np.asarray(lt_list, dtype=float)   # [hour]
    Tsurf = np.asarray(T_list, dtype=float)      # [K]
    return lt_hour, Tsurf


# ============================================================
# フーリエ級数での平滑化 + 補間
# ============================================================

def fit_fourier_series(
    lt_hour: np.ndarray,
    Tsurf: np.ndarray,
    n_harmonics: int = 3,
):
    """
    1日周期(24h)のフーリエ級数で T(lt) をフィットする。

    T(lt) ≈ a0
            + Σ_{n=1..N} [ a_n cos(2πn lt/24) + b_n sin(2πn lt/24) ]

    Returns
    -------
    model : callable
        model(lt_hour_array) -> T(lt) を返す関数
    """
    lt_hour = np.asarray(lt_hour, dtype=float)
    Tsurf = np.asarray(Tsurf, dtype=float)
    omega = 2.0 * np.pi / 24.0
    M = len(lt_hour)

    cols = [np.ones(M)]
    for n in range(1, n_harmonics + 1):
        cols.append(np.cos(n * omega * lt_hour))
        cols.append(np.sin(n * omega * lt_hour))
    A = np.column_stack(cols)  # shape: (M, 1+2*N)

    coeff, *_ = np.linalg.lstsq(A, Tsurf, rcond=None)

    def model(t_hour):
        t_hour = np.asarray(t_hour, dtype=float)
        res = coeff[0] * np.ones_like(t_hour)
        idx = 1
        for n in range(1, n_harmonics + 1):
            res += coeff[idx] * np.cos(n * omega * t_hour)
            idx += 1
            res += coeff[idx] * np.sin(n * omega * t_hour)
            idx += 1
        return res

    return model


def interp_diurnal_to_01s(
    lt_hour: np.ndarray,
    Tsurf: np.ndarray,
    sol_sec: float = 86400.0,
    dt: float = 0.1,
    method: str = "fourier",
    n_harmonics: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MCD の (lt_hour, Tsurf_K) から 0.1 s 刻みの Tsurf(t) に補間する。

    method:
        "fourier" : 1日周期フーリエ級数で平滑化してからサンプリング（デフォルト）
        "linear"  : 単純な線形補間（カクカクだが元値に忠実）
    """
    t_target = build_time_axis(sol_sec=sol_sec, dt=dt)

    if method == "linear":
        # LT[h] → 時間[s] (0〜sol_sec)
        t_src = lt_hour / 24.0 * sol_sec
        # 周期境界を滑らかにするため 1 周期分を 2 回並べる
        t_src_ext = np.concatenate([t_src, t_src + sol_sec])
        T_ext = np.concatenate([Tsurf, Tsurf])
        T_interp = np.interp(t_target, t_src_ext, T_ext)

    elif method == "fourier":
        # 先に 1日周期フーリエ級数で滑らかな関数 T(lt) を作る
        model = fit_fourier_series(lt_hour, Tsurf, n_harmonics=n_harmonics)
        # time [s] → local time [hour]
        lt_target = t_target / sol_sec * 24.0
        T_interp = model(lt_target)

    else:
        raise ValueError(f"未知の method: {method}")

    return t_target, T_interp


# ============================================================
# NetCDF 作成（テンプレ依存なし）
# ============================================================

def create_surface_forcing_dataset_from_mcd(
    mcd_txt_path: str | Path,
    sol_sec: float = 86400.0,
    dt: float = 0.1,
    var_name: str = "theta_0",
    interp_method: str = "fourier",
    n_harmonics: int = 3,
) -> xr.Dataset:
    """
    MCD テキストから直接 surface_forcing 用 Dataset を作る。

    - time [s]: 0 〜 sol_sec-dt
    - var_name(time): 「日平均からの偏差（anomaly）」を保存する
      （元の絶対温度は global attr の日平均値を使えば復元できる）
    """
    lt_hour, Tsurf = load_lt_T_from_mcd_txt(mcd_txt_path)
    t_target, T_interp_abs = interp_diurnal_to_01s(
        lt_hour,
        Tsurf,
        sol_sec=sol_sec,
        dt=dt,
        method=interp_method,
        n_harmonics=n_harmonics,
    )

    # 絶対値の平均（[K]）
    theta_mean = float(T_interp_abs.mean())
    # NetCDF に保存するのは「偏差」
    theta_anom = T_interp_abs - theta_mean

    # 互換性のため、いくつかの名前で平均値を attrs に入れておく
    mean_key = f"{var_name}_mean"

    ds = xr.Dataset(
        coords={"time": ("time", t_target)},
        data_vars={
            var_name: ("time", theta_anom.astype(np.float64)),
        },
        attrs={
            mean_key: theta_mean,                 # 一般形: 例) "theta_0_mean"
            "theta0_surface_mean": theta_mean,    # 旧コード互換
            "theta_surf_mean": theta_mean,        # 新コード互換用（必要なら）
            "description": (
                "surface potential temperature anomaly [K] from MCD "
                f"(interp={interp_method}, n_harmonics={n_harmonics}); "
                "variable stores deviation from daily mean, "
                "mean value is stored in global attributes."
            ),
        },
    )
    ds[var_name].attrs["long_name"] = "surface potential temperature anomaly at surface"
    ds[var_name].attrs["units"] = "K"

    return ds


def save_surface_forcing_dataset(ds: xr.Dataset, output_nc_path: str) -> None:
    """surface_forcing Dataset を NetCDF として保存する。"""
    out_path = Path(output_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    # 情報表示用に平均値を探す
    theta_mean = None
    # 代表的なキーを順に探す
    for key in ("theta_0_mean", "theta_surf_mean", "theta0_surface_mean"):
        if key in ds.attrs:
            theta_mean = float(ds.attrs[key])
            break

    print(f"[INFO] Wrote new surface forcing to: {out_path}")
    if theta_mean is not None:
        print(f"[INFO] mean surface potential temperature = {theta_mean:.3f} K")


# ============================================================
# 図の作成（温位偏差）
# ============================================================

def plot_surface_forcing_anomaly(
    ds: xr.Dataset,
    var_name: str,
    sol_sec: float,
    output_png_path: str,
) -> None:
    """
    surface_forcing NetCDF から温位偏差を取り出し，
    Local time vs anomaly の図を保存する。

    - 横軸: 地方時 [hour] (0〜24, 6h ごとの目盛り)
    - 縦軸: 温位偏差 [K]

    注意:
        このスクリプトでは ds[var_name] 自体が「日平均からの偏差」
        （= anomaly）になっている前提で描画する。
    """
    if "time" not in ds.coords:
        raise KeyError("Dataset に time 座標がありません。")

    if var_name not in ds.variables:
        raise KeyError(f"Dataset に変数 {var_name!r} がありません。")

    time_sec = ds["time"].to_numpy()
    theta_anom = ds[var_name].to_numpy().astype(float)  # すでに anomaly のはず

    # 時間 → 地方時 [hour]
    lt_hour = time_sec / sol_sec * 24.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lt_hour, theta_anom)
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xlabel("Local time (hour)")
    ax.set_ylabel(r"$\bar{\theta}(0,t)$ anomaly [K]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Surface forcing from MCD (theta_0 anomaly)")

    out_png = Path(output_png_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved anomaly plot to: {out_png}")


# ============================================================
# 一括実行用（他ファイルから import して呼んでもOK）
# ============================================================

def generate_surface_forcing_from_mcd(
    mcd_txt_path: str | Path,
    output_nc_path: str | Path,
    output_png_path: str | Path,
    sol_sec: float = 86400.0,
    dt: float = 0.1,
    var_name: str = "theta_0",
    interp_method: str = "fourier",
    n_harmonics: int = 3,
) -> None:
    """
    1) MCD テキスト → surface_forcing Dataset 作成
    2) NetCDF 保存
    3) 温位偏差の図を保存

    という一連の処理をまとめた関数。
    """
    ds = create_surface_forcing_dataset_from_mcd(
        mcd_txt_path=mcd_txt_path,
        sol_sec=sol_sec,
        dt=dt,
        var_name=var_name,
        interp_method=interp_method,
        n_harmonics=n_harmonics,
    )
    save_surface_forcing_dataset(ds, output_nc_path=output_nc_path)
    plot_surface_forcing_anomaly(
        ds,
        var_name=var_name,
        sol_sec=sol_sec,
        output_png_path=output_png_path,
    )


if __name__ == "__main__":
    # 入力 MCD データ: MCDdata/ 以下に配置
    mcd_txt_path = Path("MCDdata") / "Ls90_Lat0_Long260_Alt0.txt"

    # 出力先: surface_forcing_from_MCD/ 以下
    base_dir = Path("surface_forcing_from_MCD")
    stem = mcd_txt_path.stem  # 例: "Ls90_Lat0_Long260_Alt0"

    output_nc_path = base_dir / f"{stem}.nc"
    output_png_path = base_dir / f"{stem}_theta_anomaly.png"

    generate_surface_forcing_from_mcd(
        mcd_txt_path=mcd_txt_path,
        output_nc_path=output_nc_path,
        output_png_path=output_png_path,
        sol_sec=86400.0,
        dt=0.1,
        var_name="theta_0",
        interp_method="fourier",   # "linear" にすると元データに忠実なカクカク版
        n_harmonics=6,
    )
