from typing import Dict, Tuple
import numpy as np


def compute_vertical_gradients_and_ri(
    reshaped_stacked: Dict[str, np.ndarray],
    *,
    g: float,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    最後の 1 日分について du/dz, dθ/dz, 勾配リチャードソン数 Ri を計算する。

    Parameters
    ----------
    reshaped_stacked : dict
        convert_to_standard_shapes の出力を想定。
        少なくとも以下のキーを持つことを想定する::

            time        : (nt,)                 [s]
            altitude    : (nt, nz) or (nz,)     [m]
            u_bar       : (nt, nz)              [m s-1]
            theta_bar   : (nt, nz)              [K]  （偏差）
            gamma       : (nt, nz), (nz,), (nt,), (nt_last,), or scalar
            theta_0     : (1,) or scalar        [K]  （背景場の基準温位）

    g : float
        重力加速度 [m s-2]
    eps : float, optional
        0 除算を避けるための小さな値。

    Returns
    -------
    du_dz : (nt_last, nz-1)
        風速の鉛直勾配 [s-1]
    dtheta_dz : (nt_last, nz-1)
        絶対温位の鉛直勾配 [K m-1]
    Ri : (nt_last, nz-1)
        勾配リチャードソン数
    z_mid : (nt_last, nz-1)
        セル間の高度 [m]
    t_mid : (nt_last, nz-1)
        セル間の時刻 [s] （time の中央値）
    theta_mid : (nt_last, nz-1)
        セル間の絶対温位 [K]
    """

    # --- 入力を numpy 配列に変換 ---
    t_full = np.asarray(reshaped_stacked["time"], dtype=float)          # (nt_full,)
    u_full = np.asarray(reshaped_stacked["u_bar"], dtype=float)        # (nt_full, nz)
    thb_full = np.asarray(reshaped_stacked["theta_bar"], dtype=float)  # (nt_full, nz)
    z_raw = np.asarray(reshaped_stacked["altitude"], dtype=float)      # (nt_full, nz) or (nz,)
    gam_raw = np.asarray(reshaped_stacked["gamma"], dtype=float)       # いろいろな形状を許容
    th0_arr = np.asarray(reshaped_stacked["theta_0"], dtype=float)
    th0 = float(th0_arr.ravel()[0])  # scalar baseline θ0

    # --- 最後の 1 日分だけを取り出すマスク ---
    period = 86400.0  # [s]
    t_end = float(t_full[-1])
    mask = t_full > (t_end - period)    # bool, shape: (nt_full,)

    t = t_full[mask]                    # (nt_last,)
    u = u_full[mask, :]                 # (nt_last, nz)
    thb = thb_full[mask, :]             # (nt_last, nz)

    nt_last, nz = u.shape

    # ------------------------------------------------------------------
    #  任意の配列 arr を (nt_last, nz) 形状にそろえるためのヘルパー
    # ------------------------------------------------------------------
    def _time_align_to_nt_nz(arr: np.ndarray) -> np.ndarray:
        """
        想定パターン:
          - 2D: (nt_full, nz)  … 全期間×高度
          - 2D: (nt_last, nz)  … 最後の1日×高度
          - 2D: (1, nz)        … 時間一定プロファイル（1×nz）
          - 1D: (nz,)          … 高度のみ（時間一定）
          - 1D: (nt_full,)     … 時間のみ（全期間、鉛直一様）
          - 1D: (nt_last,)     … 時間のみ（最後の1日、鉛直一様）
          - 0D: スカラー       … 時間×高度とも一定
        """
        arr = np.asarray(arr, dtype=float)

        # 2D: (nt, nz)
        if arr.ndim == 2:
            # 全期間×高度 → 最後の1日だけにマスク
            if arr.shape == (t_full.size, nz):
                return arr[mask, :]
            # すでに最後の1日分
            if arr.shape == (nt_last, nz):
                return arr
            # 時間一定プロファイル
            if arr.shape == (1, nz):
                return np.broadcast_to(arr, (nt_last, nz))
            raise ValueError(f"形状不整合: {arr.shape} -> (nt_last={nt_last}, nz={nz})")

        # 1D
        if arr.ndim == 1:
            # 高度のみ（時間一定）
            if arr.size == nz:
                return np.broadcast_to(arr[None, :], (nt_last, nz))
            # 全期間の時間のみ（鉛直一様） → マスクして最後の1日分を複製
            if arr.size == t_full.size:
                arr_last = arr[mask]  # (nt_last,)
                return np.broadcast_to(arr_last[:, None], (nt_last, nz))
            # すでに最後の1日分の時間のみ（鉛直一様）
            if arr.size == nt_last:
                return np.broadcast_to(arr[:, None], (nt_last, nz))

        # 0D: スカラー
        if arr.ndim == 0:
            return np.full((nt_last, nz), float(arr))

        raise ValueError(f"扱えない形状: {arr.shape}")

    # z, gamma を (nt_last, nz) に揃える
    z = _time_align_to_nt_nz(z_raw)      # (nt_last, nz)
    gam = _time_align_to_nt_nz(gam_raw)  # (nt_last, nz)

    # --- 絶対温位 θ = θ0 + γ z + θ̄ ---
    theta = th0 + gam * z + thb          # (nt_last, nz)

    # --- 鉛直差分（上-下）/ Δz：時間ごとに z 方向差分 ---
    # ここでは np.gradient は使わず、単純な差分 np.diff を用いることで
    # 「2D に np.gradient をかけた結果が list になる」問題を避ける
    dz = np.diff(z, axis=1)              # (nt_last, nz-1)
    du = np.diff(u, axis=1)              # (nt_last, nz-1)
    dthb = np.diff(thb, axis=1)          # (nt_last, nz-1)

    du_dz = du / (dz + eps)  # (nt_last, nz-1)

    # γ はセル中心値なので、セル間では平均をとる
    gam_mid = 0.5 * (gam[:, 1:] + gam[:, :-1])  # (nt_last, nz-1)
    dtheta_dz = dthb / (dz + eps) + gam_mid     # (nt_last, nz-1)

    # --- セル間中央値 ---
    z_mid = 0.5 * (z[:, 1:] + z[:, :-1])                   # (nt_last, nz-1)
    t_mid = np.broadcast_to(t[:, None], z_mid.shape)       # (nt_last, nz-1)
    theta_mid = 0.5 * (theta[:, 1:] + theta[:, :-1])       # (nt_last, nz-1)

    # --- 勾配リチャードソン数 ---
    Ri = (g / (theta_mid + eps)) * dtheta_dz / (du_dz ** 2 + eps)

    return du_dz, dtheta_dz, Ri, z_mid, t_mid, theta_mid
