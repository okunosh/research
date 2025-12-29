import pandas as pd
import os
from datetime import datetime


def save_max_min_info_to_csv(output_csv_path, dir_path, u_info, theta_info, save=True):
    """
    最大・最小情報を CSV に保存する関数（追記モード）

    Parameters
    ----------
    output_csv_path : str
        書き込み先 CSV ファイルのパス
    dir_path : str
        処理対象データのディレクトリパス（キー）
        → run_id は原則としてこのディレクトリ名（basename）とみなす
    u_info : dict
        "u" の最大・最小情報（get_max_min_info() の返り値）
    theta_info : dict
        "theta" の最大・最小情報（同上）
    save : bool
        True のときのみ保存を実行する
    """
    if save is True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # run_id は dir_path の末端ディレクトリ名から決める
        # 例: dir_path="output/mars_Ls90_alpha0-62" → run_id="mars_Ls90_alpha0-62"
        run_id = os.path.basename(os.path.normpath(dir_path))

        row = {
            "executed_at": current_time,
            "run_id": run_id,
            "directory": dir_path,
            "u_max_value": u_info["max_value"],
            "u_t_at_max": u_info["t_at_max"],
            "u_altitude_at_max": u_info["altitude_at_max"],
            "u_min_value": u_info["min_value"],
            "u_t_at_min": u_info["t_at_min"],
            "u_altitude_at_min": u_info["altitude_at_min"],
            "theta_max_value": theta_info["max_value"],
            "theta_t_at_max": theta_info["t_at_max"],
            "theta_altitude_at_max": theta_info["altitude_at_max"],
            "theta_min_value": theta_info["min_value"],
            "theta_t_at_min": theta_info["t_at_min"],
            "theta_altitude_at_min": theta_info["altitude_at_min"],
        }

        # DataFrame 化して 1 行として保存（追記）
        df_new = pd.DataFrame([row])

        if os.path.exists(output_csv_path):
            df_existing = pd.read_csv(output_csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(output_csv_path, index=False)
        else:
            df_new.to_csv(output_csv_path, index=False)
    else:
        print(
            "If you want to save the calculation results such as wind speed, you need ->  save==True"
        )
