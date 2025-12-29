# numerical/scripts/data_analysis.py

from __future__ import annotations
import argparse
from pathlib import Path

# ---- IO 共通 ----
from visualization.common.io_utils import (
    load_all_data, stack_by_variable, convert_to_standard_shapes, read_global_attr_values
)

# ---- 可視化 / 計算（visualization ディレクトリの既存関数を利用）----
from visualization.plot_time_alt_map import plot_ubar_thetabar
from visualization.pressure_t_alt_map import (
    calc_pressure, calc_temperature, calc_density, plot_last_day_generic
)
from visualization.plot_Ri import (
    compute_vertical_gradients_and_ri, plot_last_day_gradients_and_ri
)
from visualization.plot_momentumMixing import (
    run_last_day_mixing_pipeline,
    prepare_last_day_inputs,
    plot_near_surface_wind_timeseries,
    plot_mixed_u_like_reference,
)


def run_all(directory: str) -> None:
    """
    数値計算出力ディレクトリ（NetCDF 群）を入力に、
    - ū/θ̄（または b）の時間×高度図
    - p, T, ρ, ρu（最後の1日）
    - du/dz, dθ/dz, Ri（最後の1日）
    - Ri<0.25 に基づく運動量混合と各種比較図
    を一括で作成する。
    """
    out_dir = Path(directory)

    # ---------- データ読み込み ----------
    varnames = ["u_bar", "theta_bar", "altitude", "time", "K", "theta_0", "gamma"]
    attrs    = ["g"]

    data_list = load_all_data(str(out_dir), varnames)
    stacked   = stack_by_variable(data_list, varnames)
    rs        = convert_to_standard_shapes(stacked)
    attrs_dic = read_global_attr_values(str(out_dir), attrs)
    g = float(attrs_dic["g"])

    # ---------- 物理量計算（p, T, ρ, ρu） ----------
    p_surf = 610.0  # [Pa] 火星の代表値。必要に応じて差し替え。
    p   = calc_pressure(rs, p_surf, g)
    T   = calc_temperature(rs, p, p_surf)
    rho = calc_density(rs, p, T)
    fields = {
        "p": p,
        "T": T,
        "rho": rho,
        "u_rho": rho * rs["u_bar"],
    }

    # ---------- (1) ū & θ̄/b の時間×高度図 ----------
    plot_ubar_thetabar(
        t_array=rs["time"],
        altitude=rs["altitude"],
        u_bar=rs["u_bar"],
        theta_bar=rs["theta_bar"],
        theta_0=rs["theta_0"],
        g=g,
        K=rs.get("K", None),
        period=24 * 3600,
        cmap="bwr",
        colorbar="each",
        method="pcolormesh",
        save_path=str(out_dir / "time_alt_map_pcolormesh.png"),
    )

    # ---------- (2) p, T, ρ, ρu の最後1日マップ ----------
    plot_last_day_generic(rs, "p",     fields, save_path=str(out_dir / "map_pressure.png"))
    plot_last_day_generic(rs, "T",     fields, save_path=str(out_dir / "map_temperature.png"))
    plot_last_day_generic(rs, "rho",   fields, save_path=str(out_dir / "map_density.png"))
    plot_last_day_generic(rs, "u_rho", fields, save_path=str(out_dir / "map_momentum_density.png"))

    # ---------- (3) du/dz, dθ/dz, Ri（最後の1日） ----------
    ri_results = compute_vertical_gradients_and_ri(rs, g=g)
    plot_last_day_gradients_and_ri(
        rs, ri_results,
        out_dir=str(out_dir),
        vlim_grad=0.05,          # 必要に応じて変更可
        ri_range=(-4, 0.25, 4),  # center=0.25
        show_ri_mask=False
    )

    # ---------- (4) Ri<0.25 に基づく運動量混合と比較図 ----------
    mom_mixed, u_mixed, applied_idx, top_idx = run_last_day_mixing_pipeline(
        rs, fields,
        rho=rho,
        g=g,
        ri_upper=0.25,
        require_positive=False,
        out_dir=str(out_dir),
        profile_max_plots=12,
    )

    # （補助図）近地表 u の時系列（最下層から1つ上のレベル）
    prepared = prepare_last_day_inputs(rs, rho=rho, g=g)
    plot_near_surface_wind_timeseries(
        rs, prepared, u_mixed,
        level_index=1,
        out_path=str(out_dir / "near_surface_wind_timeseries_lv1.png"),
    )

    # （補助図）K右軸＋極値★の混合後 u 参照図
    plot_mixed_u_like_reference(
        rs, u_mixed,
        save_path=str(out_dir / "mixed_u_reference.png"),
        cmap="bwr",
    )

    print(f"[done] Outputs saved under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidated data analysis pipeline for numerical simulation outputs."
    )
    parser.add_argument("directory", type=str, help="Directory containing NetCDF results")
    args = parser.parse_args()
    run_all(args.directory)
