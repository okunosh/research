#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peak_summary CSV から

1. |u| vs alpha
2. peak height vs alpha
3. LT of peaks vs alpha (+ LT of theta max/min)

の 3 つのパネル図を描画するスクリプト。

使い方例
--------
# 画像を保存（デフォルト名: <csv名>_alpha_peaks.png）
python plot_alpha_peaks.py output/.../mcd_ls180_lat-10_Lon260_peak_summary.csv

# クリティカル alpha に縦線を引く（flow_regime == "Critical"）
python plot_alpha_peaks.py peak_summary.csv --critical_alpha

# alpha 全範囲＋対数軸
python plot_alpha_peaks.py peak_summary.csv --alpha-all

# 画像は保存せずポップアップで表示したい場合
python plot_alpha_peaks.py peak_summary.csv --show

# クリティカル線 + alpha 全範囲 + 保存先を指定
python plot_alpha_peaks.py peak_summary.csv --critical_alpha --alpha-all -o my_figure.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figure(
    csv_path: Path,
    critical_alpha: bool = False,
    alpha_all: bool = False,
    gamma_val: float = None,
    theta0_val: float = None,
    u_ylim: tuple = None,
):
    """CSV を読み込んで Figure を返す。
    
    Parameters
    ----------
    gamma_val : float, optional
        フィルタリング対象の gamma 値。None の場合は全て使用。
    theta0_val : float, optional
        フィルタリング対象の theta_0 値。None の場合は全て使用。
    u_ylim : tuple, optional
        風速図(|u|)の縦軸範囲 (ymin, ymax)。None の場合は自動設定。
    """

    df = pd.read_csv(csv_path)

    # 必要な列だけ残して NaN を落とす
    cols_needed = [
        "alpha",
        "u_max_value",
        "u_t_at_max",
        "u_min_value",
        "u_t_at_min",
        "theta_t_at_max",
        "theta_t_at_min",
        "u_altitude_at_max",
        "u_altitude_at_min",
        "gamma",
        "theta_0",
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in CSV: {missing}")

    df_plot = df.dropna(subset=cols_needed).copy()

    # gamma, theta_0 でのフィルタリング
    if gamma_val is not None:
        df_plot = df_plot[df_plot["gamma"] == gamma_val]
    if theta0_val is not None:
        df_plot = df_plot[df_plot["theta_0"] == theta0_val]

    # alpha のフィルタリング
    if alpha_all:
        # alpha が存在する行は全て使う（NaN は除外）
        df_plot = df_plot[df_plot["alpha"].notna()]
    else:
        # デフォルト: 0〜2度の範囲に絞る（以前の設定）
        df_plot = df_plot[
            df_plot["alpha"].notna()
            & (df_plot["alpha"] >= 0.0)
            & (df_plot["alpha"] <= 2.0)
        ]

    if df_plot.empty:
        raise ValueError("条件に合う alpha のデータがありません。")

    # θ ピーク時刻のユニーク値
    theta_t_max_uniques = np.unique(df_plot["theta_t_at_max"].round(6))
    theta_t_min_uniques = np.unique(df_plot["theta_t_at_min"].round(6))

    print("theta_t_at_max unique (hour):", theta_t_max_uniques)
    print("theta_t_at_min unique (hour):", theta_t_min_uniques)

    # gamma, theta_0 をタイトル用に取得
    gamma_vals = df_plot["gamma"].dropna().unique()
    theta0_vals = df_plot["theta_0"].dropna().unique()

    gamma_val = gamma_vals[0] if len(gamma_vals) > 0 else None
    theta0_val = theta0_vals[0] if len(theta0_vals) > 0 else None

    if gamma_val is not None:
        gamma_scaled = gamma_val * 1e3  # ×10^-3
        gamma_text = rf"$\gamma={gamma_scaled:.2f}\times10^{{-3}}$"
    else:
        gamma_text = r"$\gamma=\mathrm{N/A}$"

    if theta0_val is not None:
        theta0_text = rf"$\theta_0={theta0_val:.3g}$"
    else:
        theta0_text = r"$\theta_0=\mathrm{N/A}$"

    # === Figure 本体 ===
    fig, (ax_u_abs, ax_alt, ax_lt) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # --- 上：alpha vs |u| (upslope / downslope) ---
    ax_u_abs.scatter(
        df_plot["alpha"],
        df_plot["u_max_value"].abs(),
        label="|upslope|",
        color="red",
    )
    ax_u_abs.scatter(
        df_plot["alpha"],
        df_plot["u_min_value"].abs(),
        marker="x",
        label="|downslope|",
        color="blue",
    )

    if u_ylim is not None:
        ymin, ymax = u_ylim
    else:
        ymax_abs = max(
            df_plot["u_max_value"].abs().max(),
            df_plot["u_min_value"].abs().max(),
        )
        ymin = 10.0
        ymax = np.ceil(ymax_abs / 10.0) * 10.0
    
    ax_u_abs.set_ylim(ymin, ymax)
    ax_u_abs.set_yticks(np.arange(ymin, ymax + 1e-6, 10.0))

    ax_u_abs.set_ylabel("|u| [m/s]")
    ax_u_abs.legend(loc="best")

    title_base = "alpha vs |u|, height and LT of peaks (last 24h)"
    title_params = f"({gamma_text}, {theta0_text})"
    fig.suptitle(f"{title_base} {title_params}", y=0.98)

    # --- 中：alpha vs peak height ---
    ax_alt.scatter(
        df_plot["alpha"],
        df_plot["u_altitude_at_max"],
        label="upslope peak",
        color="red",
    )
    ax_alt.scatter(
        df_plot["alpha"],
        df_plot["u_altitude_at_min"],
        marker="x",
        label="downslope peak",
        color="blue",
    )

    ax_alt.set_ylabel("Height at peaks [m]")
    ax_alt.legend(loc="best")

    # --- 下：alpha vs LT (+ theta max/min) ---
    ax_lt.scatter(
        df_plot["alpha"],
        df_plot["u_t_at_max"],
        label="upslope peak",
        color="red",
    )
    ax_lt.scatter(
        df_plot["alpha"],
        df_plot["u_t_at_min"],
        marker="x",
        label="downslope peak",
        color="blue",
    )

    for i, lt in enumerate(theta_t_max_uniques):
        ax_lt.axhline(
            lt,
            linestyle="--",
            linewidth=1,
            color="red",
            label="theta max" if i == 0 else None,
        )

    for i, lt in enumerate(theta_t_min_uniques):
        ax_lt.axhline(
            lt,
            linestyle="--",
            linewidth=1,
            color="blue",
            label="theta min" if i == 0 else None,
        )

    ax_lt.set_ylim(0, 24)
    ax_lt.set_yticks(np.arange(0, 25, 3))

    ax_lt.set_ylabel("Local time [hour]")
    ax_lt.set_xlabel("Slope angle alpha [deg]")
    ax_lt.legend(loc="best")

    # --- クリティカル alpha に縦線を引くオプション ---
    if critical_alpha:
        if "flow_regime" not in df_plot.columns:
            print(
                "[WARN] flow_regime 列が見つからなかったため、"
                "Critical alpha の縦線は描画しません。"
            )
        else:
            crit_alphas = (
                df_plot.loc[df_plot["flow_regime"] == "Critical", "alpha"]
                .dropna()
                .unique()
            )
            crit_alphas = np.sort(crit_alphas)
            if crit_alphas.size == 0:
                print("[INFO] flow_regime == 'Critical' のケースが無かったため、縦線はありません。")
            else:
                print("[INFO] Critical alpha:", crit_alphas)
                first = True
                for a in crit_alphas:
                    for ax in (ax_u_abs, ax_alt, ax_lt):
                        ax.axvline(
                            a,
                            color="gray",
                            linestyle="--",
                            linewidth=1,
                            alpha=0.4,
                            label="Critical alpha" if first and ax is ax_u_abs else None,
                        )
                    first = False
                # 凡例を更新（上段だけ）
                handles, labels = ax_u_abs.get_legend_handles_labels()
                ax_u_abs.legend(handles, labels, loc="best")

    # --- X 軸スケール（alpha_all のとき log） ---
    alphas = df_plot["alpha"].values
    if alphas.size > 0 and alpha_all:
        a_min = float(alphas.min())
        a_max = float(alphas.max())
        if a_min <= 0.0:
            print(
                "[WARN] alpha に 0 以下の値が含まれているため、"
                " --alpha-all ですが x 軸は線形スケールのままにします。"
            )
        else:
            for ax in (ax_u_abs, ax_alt, ax_lt):
                ax.set_xscale("log")
            x_min = a_min * 0.9
            x_max = a_max * 1.1
            for ax in (ax_u_abs, ax_alt, ax_lt):
                ax.set_xlim(x_min, x_max)

    # グリッド
    for ax in (ax_u_abs, ax_alt, ax_lt):
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot |u|, peak height, and LT vs alpha from peak_summary CSV."
    )
    parser.add_argument(
        "csv_path",
        help="Path to *_peak_summary.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path (PNG etc.). Ignored when --show is given.",
        default=None,
    )
    parser.add_argument(
        "--critical_alpha",
        action="store_true",
        help="Draw vertical dashed lines at alpha where flow_regime == 'Critical'.",
    )
    parser.add_argument(
        "--alpha-all",
        action="store_true",
        help="Use all alpha values (no 02 deg filter) and set x-axis to log scale.",
    )
    parser.add_argument(
        "--u-ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        help="Y-axis range for wind speed plot (e.g., --u-ylim 0 50).",
        default=None,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure instead of saving it.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)

    # gamma, theta_0 のグループを抽出して表示
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.dropna(subset=["gamma", "theta_0"])
    
    groups = df_raw[["gamma", "theta_0"]].drop_duplicates().sort_values(
        by=["gamma", "theta_0"]
    )
    
    print("\n" + "=" * 70)
    print("Found gamma, theta_0 groups:")
    print("=" * 70)
    for idx, (gamma_val, theta0_val) in enumerate(groups.values, 1):
        count = len(
            df_raw[(df_raw["gamma"] == gamma_val) & (df_raw["theta_0"] == theta0_val)]
        )
        print(f"  Group {idx}: gamma={gamma_val:.6f}, theta_0={theta0_val:.3g} "
              f"(n={count} rows)")
    print("=" * 70 + "\n")

    # 各グループごとに図を生成
    for gamma_val, theta0_val in groups.values:
        fig = make_figure(
            csv_path=csv_path,
            critical_alpha=args.critical_alpha,
            alpha_all=args.alpha_all,
            gamma_val=gamma_val,
            theta0_val=theta0_val,
            u_ylim=tuple(args.u_ylim) if args.u_ylim else None,
        )

        if args.show:
            plt.show()
        else:
            if args.output is not None:
                out_path = Path(args.output)
            else:
                # 自動ファイル名に gamma, theta_0 を付加
                group_str = f"_gamma{gamma_val:.6f}_theta0_{theta0_val:.3g}".replace(
                    ".", "_"
                )
                stem = csv_path.stem + "_alpha_peaks" + group_str
                
                if args.critical_alpha:
                    stem = stem + "_critical_alpha"
                if args.alpha_all:
                    stem = stem + "_alpha_all"
                
                out_path = csv_path.with_name(stem + ".png")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Saved figure to: {out_path}")

    if args.show:
        print(
            "\n[INFO] 図はファイルには保存していません。\n"
            "      画像として保存したい場合は --show を付けずに実行してください。"
        )


if __name__ == "__main__":
    main()
