import pandas as pd
import os
import signal
import sys
from datetime import datetime


def write_conditions_to_file(filename, conditions, completed):
    """
    計算条件を CSV に 1 行追記する。

    Parameters
    ----------
    filename : str
        書き込み先 CSV ファイルのパス
    conditions : dict
        計算条件（numerical_sim_exec.py から渡す params を想定）
        ※ 'run_id' を必須とする
    completed : int or bool
        計算が完了したかどうかを表すフラグ（0/1 など）
    """
    data = conditions.copy()

    # run_id が無いのに記録しても後で追えないので、ここで明示的にエラーにする
    if "run_id" not in data:
        raise KeyError("conditions に 'run_id' が含まれていません。")

    # 付加情報
    data["Completed"] = completed
    data["today"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 列の並びを明示的に決める
    fixed_cols = ["today", "run_id", "Completed"]
    other_cols = [col for col in data.keys() if col not in fixed_cols]
    columns_order = fixed_cols + other_cols

    df = pd.DataFrame([data], columns=columns_order)

    if os.path.isfile(filename):
        # 既存ファイルがあるときはヘッダなしで追記
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        # 無ければ新規作成してヘッダ付きで書き出し
        df.to_csv(filename, mode="w", header=True, index=False)


def log_error(error_message):
    """
    エラーメッセージをテキストファイルに追記するだけの関数。
    """
    with open("error_log.txt", mode="a") as file:
        file.write(error_message + "\n")


def handle_error(conditions, error_message):
    """
    エラー時に呼び出されるハンドラ。
    v2 の CSV（calculation_conditions_v2.csv）に記録する。
    """
    log_error(error_message)
    write_conditions_to_file("../calculation_conditions_v2.csv", conditions, 0)


def signal_handler(sig, frame, conditions):
    """
    Ctrl+C が押されたときにエラーハンドリングを実行。
    """
    handle_error(conditions, "Process interrupted by user")
    sys.exit(0)


def setup_signal_handler(conditions):
    """
    SIGINT シグナル（Ctrl+C）をキャッチするためのハンドラを設定。
    """
    signal.signal(
        signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, conditions)
    )
