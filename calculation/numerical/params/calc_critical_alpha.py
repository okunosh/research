# params/calc_critical_alpha.py
"""
指定した静的安定度 gamma から、Prandtl 型斜面風モデルの
「臨界斜面角 (critical slope angle)」を求めるユーティリティ。

alpha_crit は
    N_alpha = N * sin(alpha)
が日周期の強制角周波数 omega と一致する alpha を
    N * sin(alpha_crit) = omega
として求めたもの。

Parameters
----------
gamma : float
    背景の鉛直温位勾配 [K/m]
g : float
    重力加速度 [m/s^2]
theta_0 : float
    基本場の温位 [K] （例: 地表付近の代表値）
omega : float
    強制の角周波数 [rad/s] （例: 2π / 88775）

Returns
-------
alpha_deg : float
    臨界斜面角 [degree]
"""

from __future__ import annotations

import numpy as np


def calc_critical_alpha(gamma: float, g: float, theta_0: float, omega: float) -> float:
    """critical alpha [deg] を返す."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    # BruntVäisälä frequency
    N = np.sqrt(gamma * g / theta_0)

    ratio = float(omega / N)
    if ratio >= 1.0:
        raise ValueError(
            f"omega / N = {ratio:.3f} >= 1. "
            "このパラメータでは実数解としての critical alpha は存在しません。"
        )

    alpha_rad = np.arcsin(ratio)
    alpha_deg = np.degrees(alpha_rad)
    return float(alpha_deg)


if __name__ == "__main__":
    # 単体テスト用：直に実行したときだけ動く
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("gamma", type=float, help="鉛直温位勾配 gamma [K/m]")
    p.add_argument("--g", type=float, default=3.72, help="重力加速度 g [m/s^2] (default: 3.72)")
    p.add_argument("--theta0", type=float, default=213.19, help="基本場温位 theta_0 [K]")
    p.add_argument(
        "--period",
        type=float,
        default=88775.0,
        help="強制の周期 [s] (default: 88775 ≒ 1 sol)",
    )
    args = p.parse_args()

    omega = 2 * np.pi / args.period
    alpha = calc_critical_alpha(args.gamma, args.g, args.theta0, omega)
    print(f"gamma = {args.gamma:.3e} [K/m]")
    print(f"g      = {args.g:.3f} [m/s^2]")
    print(f"theta0 = {args.theta0:.2f} [K]")
    print(f"period = {args.period:.1f} [s]  ->  omega = {omega:.3e} [rad/s]")
    print(f"==> critical alpha = {alpha:.3f} deg")
