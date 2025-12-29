#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_peak_summary_for_run.py

指定した run_id ディレクトリ配下の NetCDF 出力から、
最終 24 時間の u / theta の極値情報をケースごとに集約し、
1 本の CSV にまとめるスクリプト。

- 単体実行:
    python make_peak_summary_for_run.py mcd_ls90_lat-40_Lon260

  ※ 上記の場合、カレントディレクトリから
     output/mcd_ls90_lat-40_Lon260/{planet}/{各ケース}/
     を探索します。

- 他のスクリプトからインポート:
    from make_peak_summary_for_run import make_peak_summary_for_run
"""

import argparse
import os
import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


_T_REGEX = re.compile(r"_t(\d+)\.nc$")
_LS_REGEX = re.compile(r"[lL]s(-?\d+(?:\.\d+)?)")  # run_id から Ls を取る用
_FLOW_REGEX = re.compile(r"(SuperCritical|SubCritical|Critical)")  # flow regime 用
_ALP_TOKEN_REGEX = re.compile(r"(?:^|_)alp(?P<val>-?\d+(?:-\d+)?)(?:_|$)")


def _parse_decimal(text: str) -> Optional[Decimal]:
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _extract_alpha_decimal_from_case_name(case_name: str) -> Optional[Decimal]:
    """ケース名中の `alp0-48` のようなトークンから alpha を Decimal として抽出する。"""
    m = _ALP_TOKEN_REGEX.search(case_name)
    if m is None:
        return None
    token = m.group("val")

    sign = ""
    remainder = token
    if remainder.startswith("-"):
        sign = "-"
        remainder = remainder[1:]

    if "-" in remainder:
        int_part, frac_part = remainder.split("-", 1)
        alpha_text = f"{sign}{int_part}.{frac_part}"
    else:
        alpha_text = f"{sign}{remainder}"

    return _parse_decimal(alpha_text)


def _alpha_matches_exact(alpha_value: Optional[float], case_name: str, target: Decimal) -> bool:
    """alpha を target と完全一致(数値として等しい)で判定する。

    優先順位:
    1) NetCDFの alpha attrs/var を Decimal(str(alpha)) へ
    2) 無ければ case_name の `alp0-48` トークンから推定
    """
    if alpha_value is not None:
        try:
            alpha_dec = Decimal(str(alpha_value))
        except (InvalidOperation, ValueError, TypeError):
            alpha_dec = None
        if alpha_dec is not None:
            return alpha_dec == target

    alpha_from_name = _extract_alpha_decimal_from_case_name(case_name)
    if alpha_from_name is None:
        return False
    return alpha_from_name == target


def _print_u_min_candidates_topn(
    *,
    case_name: str,
    alpha_value: Optional[float],
    sol_sec: float,
    candidates: List[Tuple[float, float, float]],
    topn: int,
    to_lt_hour,
) -> None:
    """u_min 候補(スナップショットごとの鉛直min)を小さい順に上位N件表示する。"""
    if not candidates:
        print(f"[DEBUG] alpha={alpha_value} case={case_name}: u_min candidates: none")
        return

    if topn <= 0:
        return

    sorted_cands = sorted(candidates, key=lambda x: x[0])
    best_u = float(sorted_cands[0][0])
    n_show = min(int(topn), len(sorted_cands))

    alpha_from_name = _extract_alpha_decimal_from_case_name(case_name)
    alpha_from_name_text = str(alpha_from_name) if alpha_from_name is not None else "(none)"

    print(
        "[DEBUG] u_min candidates top-N (per-snapshot vertical min)\n"
        f"        case      : {case_name}\n"
        f"        alpha(nc) : {alpha_value}\n"
        f"        alpha(name): {alpha_from_name_text}\n"
        f"        sol_sec   : {sol_sec}\n"
        f"        total candidates: {len(sorted_cands)}\n"
        f"        showing   : {n_show}"
    )
    for rank, (u_min, t_sec, z_at_min) in enumerate(sorted_cands[:n_show], start=1):
        lt_hour = to_lt_hour(t_sec)
        delta = float(u_min) - best_u
        print(
            f"[DEBUG] #{rank:02d} u_min={u_min:+.6g}  Δ={delta:.3g}  "
            f"t_sec={t_sec:.0f}  LT={lt_hour:.6g} h  z={z_at_min:.6g}"
        )


def _extract_t_sec_from_filename(filename: str) -> Optional[int]:
    """
    ファイル名末尾の `_t******.nc` から経過秒数 t_sec を取得する。
    見つからなければ None を返す。
    """
    m = _T_REGEX.search(os.path.basename(filename))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_ls_from_run_id(run_id: str) -> Optional[float]:
    """
    run_id 文字列から Ls を取り出す。
    例: "mcd_ls180_lat-10_Lon260" -> 180.0
        "mcd_Ls90.0_lat0_Lon260"  -> 90.0
    見つからなければ None。
    """
    m = _LS_REGEX.search(run_id)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_flow_regime(case_name: str) -> Optional[str]:
    """
    ケースディレクトリ名から flow regime を取り出す。

    例:
        "20251127T181710_SubCritical_num261_..." -> "SubCritical"
        "20251127T181710_SuperCritical_num261_..." -> "SuperCritical"

    SuperCritical / SubCritical / Critical 以外は None。
    """
    m = _FLOW_REGEX.search(case_name)
    if m is None:
        return None
    regime = m.group(1)
    if regime in {"SuperCritical", "SubCritical", "Critical"}:
        return regime
    return None


def _load_1d_profile(ds: xr.Dataset, var_name: str, z_size: int) -> np.ndarray:
    """
    Dataset から指定変数の鉛直プロファイルを 1 次元配列として取り出す。

    想定パターン:
    - shape == (z,)                  -> そのまま
    - shape == (z, time) with time=1 -> [:, 0]
    - shape == (time, z) with time=1 -> [0, :]
    """
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    arr = ds[var_name].values

    if arr.ndim == 1:
        if arr.shape[0] != z_size:
            raise ValueError(
                f"Variable '{var_name}' is 1D but length {arr.shape[0]} "
                f"!= z_size {z_size}."
            )
        return arr.astype(float)

    if arr.ndim == 2:
        if arr.shape[0] == z_size and arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        if arr.shape[1] == z_size and arr.shape[0] == 1:
            return arr[0, :].astype(float)
        raise ValueError(
            f"Variable '{var_name}' has shape {arr.shape}, which does not "
            f"match expected (z, 1) or (1, z) with z_size={z_size}."
        )

    raise ValueError(
        f"Variable '{var_name}' has unsupported ndim={arr.ndim}. "
        "Expected 1D or 2D with a singleton time dimension."
    )


def _get_scalar_from_attrs_or_var(
    ds: xr.Dataset, attr_key: str, var_name: Optional[str] = None
) -> Optional[float]:
    """
    まず global attrs の attr_key を探し、見つからなければ
    var_name 変数 (デフォルトは attr_key と同じ名前) から平均値を取る。
    どちらも無ければ None。
    """
    if attr_key in ds.attrs:
        try:
            return float(ds.attrs[attr_key])
        except Exception:
            pass

    if var_name is None:
        var_name = attr_key

    if var_name in ds.variables:
        try:
            arr = ds[var_name].values
            return float(np.nanmean(arr))
        except Exception:
            return None

    return None


def make_peak_summary_for_run(
    run_dir: str,
    planet: str = "Mars",
    sol_sec: float = 86400.0,
    output_csv: Optional[str] = None,
    u_var_name: str = "u_bar",
    theta_var_name: str = "theta_bar",
    z_var_name: str = "altitude",   # altitude を高度変数として読む
    verbose: bool = True,
    debug_alpha: Optional[str] = None,
    debug_topn: int = 10,
) -> pd.DataFrame:
    """
    指定した run_id ディレクトリ配下の {planet} サブディレクトリを走査し、
    各ケース({date}_{params} ディレクトリ)について最終 24 時間の
    u / theta の極値情報を 1 行にまとめた DataFrame を返す。

    さらに output_csv が指定されていれば CSV に保存する。
    """

    # 引数 run_dir は「そのままパス」と「run_id（output配下）」の両方に対応する
    run_dir_abs = os.path.abspath(run_dir)
    candidate1_planet_dir = os.path.join(run_dir_abs, planet)

    # カレントディレクトリから output/{run_dir}/{planet} も候補にする
    output_base = os.path.abspath(os.path.join(os.getcwd(), "output"))
    run_dir_output = os.path.join(output_base, run_dir)
    candidate2_planet_dir = os.path.join(run_dir_output, planet)

    if os.path.isdir(candidate1_planet_dir):
        planet_dir = candidate1_planet_dir
        resolved_run_dir = os.path.dirname(planet_dir)
    elif os.path.isdir(candidate2_planet_dir):
        planet_dir = candidate2_planet_dir
        resolved_run_dir = os.path.dirname(planet_dir)
    else:
        raise FileNotFoundError(
            "Planet directory not found.\n"
            f"  Tried: {candidate1_planet_dir}\n"
            f"     or: {candidate2_planet_dir}"
        )

    run_dir = resolved_run_dir
    run_id = os.path.basename(os.path.normpath(run_dir))

    # run_id から Ls を決める
    Ls_value = _extract_ls_from_run_id(run_id)

    if output_csv is None:
        output_csv = os.path.join(planet_dir, f"{run_id}_peak_summary.csv")

    if verbose:
        print(f"[INFO] resolved run_dir : {run_dir}")
        print(f"[INFO] run_id          : {run_id}")
        print(f"[INFO] planet_dir      : {planet_dir}")
        print(f"[INFO] output_csv      : {output_csv}")
        print(f"[INFO] sol_sec         : {sol_sec}")
        print(f"[INFO] u variable      : {u_var_name}")
        print(f"[INFO] theta var       : {theta_var_name}")
        print(f"[INFO] altitude var    : {z_var_name}")
        print(f"[INFO] Ls (from run_id): {Ls_value}")

    debug_alpha_dec = _parse_decimal(debug_alpha) if debug_alpha is not None else None
    if debug_alpha is not None and debug_alpha_dec is None:
        raise ValueError(f"Invalid --debug-alpha value: {debug_alpha}")

    debug_mode = debug_alpha_dec is not None
    found_debug_case = False

    # planet_dir 配下の {date}_{params} ディレクトリを列挙
    case_dirs: List[str] = []
    for name in sorted(os.listdir(planet_dir)):
        path = os.path.join(planet_dir, name)
        if os.path.isdir(path):
            case_dirs.append(path)

    if verbose:
        print(f"[INFO] Found {len(case_dirs)} case directories under {planet_dir}")

    rows = []

    for case_path in case_dirs:
        case_name = os.path.basename(case_path)
        rel_directory = os.path.join(planet, case_name)

        # flow regime を case_name から抽出
        flow_regime = _extract_flow_regime(case_name)

        nc_files = [
            os.path.join(case_path, f)
            for f in os.listdir(case_path)
            if f.endswith(".nc")
        ]
        if not nc_files:
            if verbose:
                print(f"[WARN] No .nc files in {case_path}, skipping.")
            continue

        # 各ファイルから t_sec を抽出
        file_infos: List[Tuple[int, str]] = []
        for fpath in nc_files:
            t_sec = _extract_t_sec_from_filename(fpath)
            if t_sec is None:
                if verbose:
                    print(f"[WARN] Could not parse t_* from {os.path.basename(fpath)}")
                continue
            file_infos.append((t_sec, fpath))

        if not file_infos:
            if verbose:
                print(f"[WARN] No valid t_* pattern in {case_path}, skipping.")
            continue

        # 最終 24 時間の範囲を決める
        t_values = np.array([t for t, _ in file_infos], dtype=float)
        t_max = float(t_values.max())
        t_min = t_max - float(sol_sec)

        # 最終 24 時間に入るファイルだけを抽出
        selected = [(t, f) for (t, f) in file_infos if t >= t_min]
        if not selected:
            if verbose:
                print(f"[WARN] No files in last {sol_sec} s in {case_path}, skipping.")
            continue

        selected.sort(key=lambda tf: tf[0])

        if verbose:
            print(f"[INFO] Case {rel_directory}: {len(selected)} files in last 24h window.")

        # グローバル情報と極値用の変数を初期化
        alpha = None
        gamma = None
        theta_0 = None

        debug_this_case = False
        u_min_candidates: List[Tuple[float, float, float]] = []  # (u_min, t_sec, z_at_min)

        z = None  # type: Optional[np.ndarray]

        u_max_value = None
        u_min_value = None
        theta_max_value = None
        theta_min_value = None

        t_at_u_max_value = None
        t_at_u_min_value = None
        t_at_theta_max_value = None
        t_at_theta_min_value = None

        z_at_u_max_value = None
        z_at_u_min_value = None
        z_at_theta_max_value = None
        z_at_theta_min_value = None

        # スナップショットごとのループ（最終24時間の各ファイル）
        for t_sec, fpath in selected:
            with xr.open_dataset(fpath) as ds:
                # 最初のファイルで altitude とパラメータ系を取得
                if z is None:
                    if z_var_name not in ds:
                        raise KeyError(
                            f"Variable '{z_var_name}' not found in dataset: {fpath}"
                        )
                    z_arr = ds[z_var_name].values
                    if z_arr.ndim != 1:
                        raise ValueError(
                            f"altitude variable '{z_var_name}' must be 1D, "
                            f"but got shape {z_arr.shape} in {fpath}"
                        )
                    z = z_arr.astype(float)
                    z_size = z.size

                    # パラメータ類 (global attrs or variables)
                    alpha = _get_scalar_from_attrs_or_var(ds, "alpha")
                    gamma = _get_scalar_from_attrs_or_var(ds, "gamma")
                    theta_0 = _get_scalar_from_attrs_or_var(ds, "theta_0")

                    if debug_alpha_dec is not None:
                        debug_this_case = _alpha_matches_exact(alpha, case_name, debug_alpha_dec)

                # u, theta の鉛直プロファイル
                assert z is not None
                z_size = z.size

                try:
                    u_profile = _load_1d_profile(ds, u_var_name, z_size)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load '{u_var_name}' from {fpath}: {e}"
                    )

                try:
                    theta_profile = _load_1d_profile(ds, theta_var_name, z_size)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load '{theta_var_name}' from {fpath}: {e}"
                    )

                # NaN だらけのスナップショットはスキップ
                if np.all(np.isnan(u_profile)) and np.all(np.isnan(theta_profile)):
                    continue

                # u の極値
                if not np.all(np.isnan(u_profile)):
                    u_max_snap = np.nanmax(u_profile)
                    u_min_snap = np.nanmin(u_profile)

                    # 上昇風 (正の最大値)
                    if u_max_value is None or u_max_snap > u_max_value:
                        u_max_value = float(u_max_snap)
                        idx_max = int(np.nanargmax(u_profile))
                        z_at_u_max_value = float(z[idx_max])
                        t_at_u_max_value = float(t_sec)

                    # 下降風 (負方向の極値)
                    if u_min_value is None or u_min_snap < u_min_value:
                        u_min_value = float(u_min_snap)
                        idx_min = int(np.nanargmin(u_profile))
                        z_at_u_min_value = float(z[idx_min])
                        t_at_u_min_value = float(t_sec)

                    if debug_this_case:
                        idx_min_dbg = int(np.nanargmin(u_profile))
                        u_min_candidates.append(
                            (float(u_min_snap), float(t_sec), float(z[idx_min_dbg]))
                        )

                # theta の極値
                if not np.all(np.isnan(theta_profile)):
                    theta_max_snap = np.nanmax(theta_profile)
                    theta_min_snap = np.nanmin(theta_profile)

                    if theta_max_value is None or theta_max_snap > theta_max_value:
                        theta_max_value = float(theta_max_snap)
                        idx_tmax = int(np.nanargmax(theta_profile))
                        z_at_theta_max_value = float(z[idx_tmax])
                        t_at_theta_max_value = float(t_sec)

                    if theta_min_value is None or theta_min_snap < theta_min_value:
                        theta_min_value = float(theta_min_snap)
                        idx_tmin = int(np.nanargmin(theta_profile))
                        z_at_theta_min_value = float(z[idx_tmin])
                        t_at_theta_min_value = float(t_sec)

        # 1 ケース分の結果を整形
        def _to_lt_hour(t_opt: Optional[float]) -> Optional[float]:
            if t_opt is None:
                return None
            return float((t_opt % float(sol_sec)) / float(sol_sec) * 24.0)

        if debug_this_case:
            found_debug_case = True
            _print_u_min_candidates_topn(
                case_name=case_name,
                alpha_value=alpha,
                sol_sec=sol_sec,
                candidates=u_min_candidates,
                topn=debug_topn,
                to_lt_hour=_to_lt_hour,
            )
            # デバッグ用途では CSV は保存しない。指定αのケースを出したら終了。
            break

        if debug_mode:
            # debug-alpha 指定時は該当ケース以外の集計を行わない
            continue

        row = {
            "run_id": run_id,
            "directory": rel_directory,
            "Ls": Ls_value,                 # run_id 由来で固定
            "flow_regime": flow_regime,     # 追加: Super/Sub/Critical or None
            "alpha": alpha,                 # NetCDF の alpha
            "gamma": gamma,
            "theta_0": theta_0,
            "u_max_value": u_max_value,
            "u_t_at_max": _to_lt_hour(t_at_u_max_value),
            "u_altitude_at_max": z_at_u_max_value,
            "u_min_value": u_min_value,
            "u_t_at_min": _to_lt_hour(t_at_u_min_value),
            "u_altitude_at_min": z_at_u_min_value,
            "theta_max_value": theta_max_value,
            "theta_t_at_max": _to_lt_hour(t_at_theta_max_value),
            "theta_altitude_at_max": z_at_theta_max_value,
            "theta_min_value": theta_min_value,
            "theta_t_at_min": _to_lt_hour(t_at_theta_min_value),
            "theta_altitude_at_min": z_at_theta_min_value,
        }

        rows.append(row)

    # debug-alpha 指定時はここで終了（CSVは保存しない）
    if debug_mode:
        if not found_debug_case:
            print(f"[DEBUG] No matching case found for alpha={str(debug_alpha_dec)}")
        print(
            "[INFO] --debug-alpha is set: skipping CSV output and exiting. "
            "To save CSV, rerun without --debug-alpha."
        )
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)

    # CSV 出力
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"[INFO] Wrote peak summary with {len(df)} rows to: {output_csv}")

    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize peak upslope/downslope wind and theta extremes for the "
            "last 24 hours of each case under a given run_id."
        )
    )
    parser.add_argument(
        "run_dir",
        help=(
            "run_id または run_id へのパス "
            "(例: mcd_ls90_lat-40_Lon260 または output/mcd_ls90_lat-40_Lon260)"
        ),
    )
    parser.add_argument(
        "--planet",
        default="Mars",
        help="planet サブディレクトリ名 (デフォルト: Mars)",
    )
    parser.add_argument(
        "--sol-sec",
        type=float,
        default=86400.0,
        help="1 日(1 sol)の秒数 (デフォルト: 86400)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "出力 CSV パス。指定しない場合は "
            "{resolved_run_dir}/{planet}/{run_id}_peak_summary.csv"
        ),
    )
    parser.add_argument(
        "--u-var-name",
        default="u_bar",
        help="斜面方向風速の変数名 (デフォルト: u_bar)",
    )
    parser.add_argument(
        "--theta-var-name",
        default="theta_bar",
        help="温位偏差(θ')の変数名 (デフォルト: theta_bar)",
    )
    parser.add_argument(
        "--z-var-name",
        default="altitude",
        help="高度座標の変数名 (デフォルト: altitude)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="進捗メッセージを出さない",
    )
    parser.add_argument(
        "--debug-alpha",
        default=None,
        help=(
            "指定した alpha のケースだけ、u_min候補(スナップショットごとの鉛直min) "
            "上位N件を標準出力に表示する。完全一致(数値一致)。"
            "このオプションを指定した場合、CSVは保存せずに終了します（CSVが欲しい場合は --debug-alpha を外して実行）。"
        ),
    )
    parser.add_argument(
        "--debug-topn",
        type=int,
        default=10,
        help="--debug-alpha 指定時に表示する上位N件 (デフォルト: 10)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    make_peak_summary_for_run(
        run_dir=args.run_dir,
        planet=args.planet,
        sol_sec=args.sol_sec,
        output_csv=args.output_csv,
        u_var_name=args.u_var_name,
        theta_var_name=args.theta_var_name,
        z_var_name=args.z_var_name,
        verbose=not args.quiet,
        debug_alpha=args.debug_alpha,
        debug_topn=args.debug_topn,
    )
