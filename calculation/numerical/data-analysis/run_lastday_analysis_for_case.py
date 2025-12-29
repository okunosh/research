from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from visualization.common.io_utils import (
    load_all_data,
    stack_by_variable,
    convert_to_standard_shapes,
    read_global_attr_values,
)
from visualization.pressure_t_alt_map import (
    calc_pressure,
    calc_temperature,
    calc_density,
)
from visualization.plot_Ri import compute_vertical_gradients_and_ri
from visualization.Ri_analysis_summary import (
    plot_ri_gradients_summary,
    _fallback_g,
)
from visualization.plot_momentumMixing import (
    run_last_day_mixing_pipeline,
    prepare_last_day_inputs,
    plot_near_surface_wind_timeseries,
    plot_mixed_u_like_reference,
)


def run_lastday_analysis_for_case(directory: str) -> None:
    """
    1ケース分について以下を一括実行するスクリプト：

      (B) 最後の1日の Ri・∂u/∂z・∂θ/∂z を 1行3列の図にまとめて保存
      (C) Ri<0.25 に基づく運動量混合を最後の1日に適用し，
          ・混合結果の可視化（plot_momentumMixing 側に任せる）
          ・近地表風速の時系列
          ・混合前後の u プロファイル参照図
        を保存する。
    """
    out_dir = Path(directory)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {out_dir}")

    # ---------- データ読み込み ----------
    varnames = ["u_bar", "theta_bar", "altitude", "time", "K", "theta_0", "gamma"]
    attrs    = ["g", "planet"]

    data_list = load_all_data(str(out_dir), varnames)
    if not data_list:
        raise RuntimeError(f"No NetCDF files found under: {out_dir}")

    stacked = stack_by_variable(data_list, varnames)
    rs = convert_to_standard_shapes(stacked)  # reshaped_stacked と同じもの

    attrs_dic = read_global_attr_values(str(out_dir), attrs)

    # ---------- g の決定（Ri_analysis_summary と同じ fallback ロジック） ----------
    p_surf = 610.0  # [Pa] 火星用の代表値（必要なら将来拡張）
    g = _fallback_g(attrs_dic, p_surf=p_surf)
    print(f"[info] using g = {g} m s^-2")

    # ---------- 物理量計算（p, T, ρ, ρu） ----------
    p   = calc_pressure(rs, p_surf, g)
    T   = calc_temperature(rs, p, p_surf)
    rho = calc_density(rs, p, T)
    fields = {
        "p": p,
        "T": T,
        "rho": rho,
        "u_rho": rho * rs["u_bar"],
    }

    # ========== (B) Ri・∂u/∂z・∂θ/∂z の 1×3 サマリー図 ==========
    # 既存の Ri_analysis_summary をそのまま呼ぶ
    plot_ri_gradients_summary(str(out_dir))

    # （ついでに混合用に Ri も計算しておく：ここで計算したものを C でも再利用）
    ri_results = compute_vertical_gradients_and_ri(rs, g=g)

    # デバッグ出力：Ri の min/max と 0 < Ri < 0.25 の割合
    du_dz, dtheta_dz, Ri, z_mid, t_mid, theta_mid = ri_results
    with np.errstate(all="ignore"):
        if Ri.size > 0:
            ri_min = float(np.nanmin(Ri))
            ri_max = float(np.nanmax(Ri))
            mask = (Ri > 0.0) & (Ri < 0.25)
            frac = float(np.count_nonzero(mask)) / float(Ri.size)
        else:
            ri_min = ri_max = frac = np.nan
    print(f"[Ri debug] min = {ri_min:.3g}, max = {ri_max:.3g}")
    print(f"[Ri debug] fraction(0 < Ri < 0.25) = {frac:.3f}")

    # ========== (C) Ri<0.25 に基づく運動量混合パイプライン ==========
    # ここは plot_momentumMixing.py の __main__ に近い処理を統合
    mom_mixed, u_mixed, applied_idx, top_idx_list = run_last_day_mixing_pipeline(
        rs,
        fields,
        rho=rho,
        ri_results=ri_results,   # ここで計算した Ri を再利用
        g=g,
        ri_upper=0.25,
        require_positive=False,
        out_dir=str(out_dir),    # 結果図はこのディレクトリ配下に保存
        profile_max_plots=12,
    )

    # 近地表（最下層から1つ上）の風速時系列
    prepared = prepare_last_day_inputs(rs, rho=rho, ri_results=ri_results, g=g)
    plot_near_surface_wind_timeseries(
        rs,
        prepared,
        u_mixed,
        level_index=1,  # 「最下層から1つ上」を想定
        out_path=str(out_dir / "near_surface_wind_timeseries_lv1.png"),
    )

    # K 右軸＋極値★の混合後 u 参照図
    plot_mixed_u_like_reference(
        rs,
        u_mixed,
        save_path=str(out_dir / "mixed_u_reference.png"),
        cmap="bwr",
    )

    print(f"[done] Outputs saved under: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Case-wise last-day analysis:\n"
            "  (B) Ri & gradients 1x3 summary\n"
            "  (C) Ri<0.25-based momentum mixing & near-surface wind"
        )
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing NetCDF files for a single case "
             "(e.g. output/.../Mars/2025xxxx_.../)",
    )
    args = parser.parse_args()
    run_lastday_analysis_for_case(args.directory)


if __name__ == "__main__":
    main()
