#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alpha だけ異なるパラメータファイルをまとめて生成するスクリプト。

- ベースとなる 1 つのパラメータファイル
  （例: params/CASE-B_alpha_TEMPLATE.py）を読み取り、
  そこから複数の alpha 用ファイルを作成する。
- ベースファイルは
      ALPHA_DEG = __ALPHA_DEG__
  のようなプレースホルダを含んでいる想定。
- デフォルトでは、同じ CASE に対して臨界斜面角 alpha_crit を
  自動で計算し、その値に対応するファイルも 1 つ生成する。
  （--no-critical を指定するとこの自動追加を無効化）

想定ファイル名
--------------
- テンプレ: CASE-B_alpha_TEMPLATE.py
- 出力   : CASE-B_alpha0-20.py, CASE-B_alpha0-62.py, ...

使い方（例）
------------
python numerical/params/generate_alpha_params.py \\
    numerical/params/CASE-B_alpha_TEMPLATE.py \\
    --alphas 0.20 0.62 1.00

"""

import argparse
import re
import runpy
from pathlib import Path
from typing import Iterable, List

import xarray as xr

from params.base_cases import CASES
from params.calc_critical_alpha import calc_critical_alpha


def alpha_to_tag(alpha: float) -> str:
    """
    ファイル名に使うための alpha のタグ文字列を作る。
    例: 0.2  -> "0-20"
        0.62 -> "0-62"
        1.0  -> "1-00"
    """
    return f"{alpha:.2f}".replace(".", "-")


def load_param_metadata(param_path: Path):
    """
    param ファイルを実行して CASE_NAME / PARAMS / OVERRIDES / surface_temp を取り出す。

    ALPHA_DEG = __ALPHA_DEG__ のようなプレースホルダがあっても、
    __ALPHA_DEG__ にダミー値 (0.0) を注入してから実行することで NameError を防ぐ。
    """
    init_globals = {"__ALPHA_DEG__": 0.0}
    ns = runpy.run_path(str(param_path), init_globals=init_globals)

    case_name = ns.get("CASE_NAME")
    if case_name is None:
        raise RuntimeError(f"{param_path} に CASE_NAME が定義されていません。")

    params = ns.get("PARAMS", {})
    overrides = ns.get("OVERRIDES", {})

    # surface_temp は OVERRIDES を優先し、なければ PARAMS から取る
    surface_temp = overrides.get("surface_temp") or params.get("surface_temp")
    if surface_temp is None:
        raise RuntimeError(
            f"{param_path} に surface_temp が見つかりません。"
            "OVERRIDES または PARAMS に 'surface_temp' を定義してください。"
        )

    return case_name, params, overrides, surface_temp


def read_theta0_from_netcdf(root_dir: Path, surface_temp_rel: str) -> float:
    """
    surface_temp NetCDF の global attribute 'theta_surf_mean' から theta_0 を読む。
    """
    nc_path = (root_dir / surface_temp_rel).resolve()
    if not nc_path.exists():
        raise FileNotFoundError(f"surface_temp ファイルが見つかりません: {nc_path}")

    ds = xr.open_dataset(nc_path)
    try:
        if "theta_surf_mean" not in ds.attrs:
            raise KeyError(
                f"{nc_path} の global attrs に 'theta_surf_mean' がありません。"
            )
        theta0 = float(ds.attrs["theta_surf_mean"])
    finally:
        ds.close()

    return theta0


def ensure_critical_alpha(
    alphas: Iterable[float],
    case_name: str,
    theta0: float,
) -> List[float]:
    """
    alpha のリストに、その CASE に対応する critical alpha を追加する。
    すでに近い値が含まれている場合は追加しない。
    """
    case = CASES[case_name]
    gamma = float(case["gamma"])
    g = float(case["g"])
    omega = float(case["omega"])  # [rad/s]

    print(
        f"[INFO] CASE_NAME={case_name}, gamma={gamma:.3e} [K/m], "
        f"g={g:.2f} [m/s^2], omega={omega:.3e} [rad/s]"
    )
    print(f"[INFO] theta_0 (from theta_surf_mean) = {theta0:.2f} K")

    alpha_crit = calc_critical_alpha(gamma=gamma, g=g, theta_0=theta0, omega=omega)
    print(f"[INFO] computed critical alpha = {alpha_crit:.3f} deg")

    alphas_list = list(alphas)

    def already_has(val: float, arr: Iterable[float], tol: float = 1e-3) -> bool:
        return any(abs(val - a) < tol for a in arr)

    if already_has(alpha_crit, alphas_list):
        print("[INFO] critical alpha は --alphas に既に含まれているため追加しません。")
    else:
        alphas_list.append(alpha_crit)
        print(f"[INFO] critical alpha を追加しました。alphas = {alphas_list}")

    return alphas_list


def build_output_name(base_path: Path, alpha: float) -> Path:
    """
    ベースファイル名から、指定された alpha 用のファイル名を作る。

    想定ケース:
      CASE-B_alpha_TEMPLATE.py  -> CASE-B_alpha0-20.py
      CASE-B_alpha0-20.py      -> CASE-B_alpha0-35.py  （数字だけ差し替え）
      foo.py                   -> foo_alpha0-20.py
    """
    base_name = base_path.name
    tag = alpha_to_tag(alpha)

    # 1) _alpha_TEMPLATE を置き換えるパターン
    if "_alpha_TEMPLATE" in base_name:
        new_name = base_name.replace("_alpha_TEMPLATE", f"_alpha{tag}")
        return base_path.with_name(new_name)

    # 2) 既に alpha 数字タグがある場合は、その数字部分だけ差し替える
    m = re.search(r"_alpha([0-9\-]+)", base_name)
    if m:
        new_name = f"{base_name[: m.start(1)]}{tag}{base_name[m.end(1) :]}"
        return base_path.with_name(new_name)

    # 3) それ以外は末尾に付け足す
    stem, suffix = base_path.stem, base_path.suffix
    new_name = f"{stem}_alpha{tag}{suffix}"
    return base_path.with_name(new_name)


def generate_files(param_path: Path, alphas: Iterable[float]) -> None:
    """
    実際に param ファイルを生成するメイン処理。
    テンプレート内には必ず '__ALPHA_DEG__' が含まれている前提。
    """
    template_text = param_path.read_text(encoding="utf-8")

    if "__ALPHA_DEG__" not in template_text:
        raise RuntimeError(
            f"{param_path} 内に '__ALPHA_DEG__' が見つかりません。\n"
            "テンプレートは 'ALPHA_DEG = __ALPHA_DEG__' の形式にしてください。"
        )

    for alpha in alphas:
        tag = alpha_to_tag(alpha)
        out_path = build_output_name(param_path, alpha)

        new_text = template_text.replace("__ALPHA_DEG__", f"{alpha:.2f}")

        if out_path.exists():
            print(f"[WARN] 上書き: {out_path}")
        else:
            print(f"[INFO] 生成: {out_path}")

        out_path.write_text(new_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="alpha だけ異なる param ファイルをまとめて生成するツール",
    )
    parser.add_argument(
        "param_file",
        help="ベースとなる param ファイルへのパス（例: params/CASE-B_alpha_TEMPLATE.py）",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        required=True,
        help="生成したい alpha[deg] をスペース区切りで指定（例: --alphas 0.2 0.62 1.0）",
    )
    parser.add_argument(
        "--no-critical",
        action="store_true",
        help="critical alpha を自動で追加しない",
    )

    args = parser.parse_args()

    param_path = Path(args.param_file).resolve()
    if not param_path.exists():
        raise FileNotFoundError(f"param ファイルが見つかりません: {param_path}")

    print(f"[INFO] ベース param ファイル: {param_path}")
    print(f"[INFO] 初期 alphas: {args.alphas}")

    # param ファイルから CASE_NAME と surface_temp パスを取得
    case_name, params, overrides, surface_temp_rel = load_param_metadata(param_path)

    # critical alpha を追加
    alphas = list(args.alphas)
    project_root = param_path.parent.parent  # ../ （params/ の 1 つ上）想定

    if not args.no_critical:
        theta0 = read_theta0_from_netcdf(project_root, surface_temp_rel)
        alphas = ensure_critical_alpha(alphas, case_name, theta0)
    else:
        print("[INFO] --no-critical が指定されたため critical alpha の自動追加は行いません。")

    # 並びを揃えておくと後から見やすい
    alphas_sorted = sorted(alphas)
    print(f"[INFO] 最終的な alpha リスト: {alphas_sorted}")

    generate_files(param_path, alphas_sorted)


if __name__ == "__main__":
    main()
