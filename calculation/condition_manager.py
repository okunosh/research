import pandas as pd
import os
import signal
import sys
from datetime import datetime

def write_conditions_to_file(filename, conditions, completed):
    data = conditions.copy()
    data['Completed'] = completed
    data['now'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def log_error(error_message):
    with open('error_log.txt', mode='a') as file:
        file.write(error_message + '\n')

def handle_error(conditions, error_message):
    log_error(error_message)
    write_conditions_to_file('../calculation_conditions.csv', conditions, 0)

def signal_handler(sig, frame, conditions):
    # Ctrl+Cが押されたときにエラーハンドリングを実行
    handle_error(conditions, "Process interrupted by user")
    sys.exit(0)

# SIGINTシグナル（Ctrl+C）をキャッチするためのハンドラを設定
def setup_signal_handler(conditions):
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, conditions))
