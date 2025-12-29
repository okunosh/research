# params/024-High_alpha_TEMPLATE.py
from pathlib import Path
from params.base_cases import CASES

# ケース名だけ固定。Ls / Lon / Lat は下の CASE_B_DIC でまとめて管理
CASE_NAME = "024-High"

CASE_B_DIC = {
    "LS": 174.535,    # degree
    "LON": 256.0,  # degrees_north  -> ファイル名の Lat 部分に使う
    "LAT": -12.3,  # degrees_east   -> ファイル名の Lon 部分に使う
}

#---------------
#以下は書き換え不要
#---------------

# 基本はファイル名を run_id にするが、必要ならここを書き換えればよい
DEFAULT_RUN_ID = (
    f"{CASE_NAME}_mcd_ls{CASE_B_DIC['LS']}"
    f"_lat{CASE_B_DIC['LON']}_Lon{CASE_B_DIC['LAT']}"
)
RUN_ID = DEFAULT_RUN_ID

BASE = CASES[CASE_NAME].copy()

# ← ここにプレースホルダ（数値ではなくトークン）を置く
ALPHA_DEG = __ALPHA_DEG__  # 自動生成スクリプトで数値を埋める

# MCD 由来ファイル名（ディレクトリを除いた共通部分）
MCD_BASENAME = (
    f"Ls{CASE_B_DIC['LS']}_Lat{CASE_B_DIC['LON']}"
    f"_Lon{CASE_B_DIC['LAT']}_Alt0_fourierN6.nc"
)

OVERRIDES = {
    "alpha_deg": ALPHA_DEG,
    # run_id ごとに surface_temp, K_file を変える想定 not use!!
    #"surface_temp": f"surface_forcing_from_MCD/{MCD_BASENAME}",
    #"K_file": f"K_MCD/{MCD_BASENAME}",
}

PARAMS = {**BASE, **OVERRIDES}
