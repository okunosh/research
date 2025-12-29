import os
import re
import numpy as np
import xarray as xr

"""
def extract_time_from_filename(filename):
    ファイル名からt******部分を抽出（例: _t0345600.nc → 345600）
    m = re.search(r'_t(\d{6})\.nc$', filename)
    return int(m.group(1)) if m else None
"""

def extract_time_from_filename(filename):
    """ファイル名からt******部分を抽出（例: A_***_t0342000.nc → 342000）"""
    m = re.search(r't(\d{6})', filename)
    return int(m.group(1)) if m else None

def list_netcdf_files_sorted(directory):
    """ディレクトリ内のNetCDFファイルをt******順にソートして返す"""
    files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    files_with_time = []
    for f in files:
        t_val = extract_time_from_filename(f)
        if t_val is not None:
            files_with_time.append((t_val, f))
    files_sorted = [f for t, f in sorted(files_with_time)]
    return files_sorted

def load_all_data(directory, varnames):
    """
    指定ディレクトリ内のnetcdfファイルを時系列順に全て読み、
    指定した変数(varnames)を欠損時はnp.nanで埋めて辞書のリストで返す。
    """
    files_sorted = list_netcdf_files_sorted(directory)
    #print(files_sorted)
    
    data_list = []
    for fname in files_sorted:
        ds = xr.open_dataset(os.path.join(directory, fname))
        data = {}
        for v in varnames:
            if v in ds:
                data[v] = ds[v].values
            else:
                # 欠損時は最小限のnp.nan（shape決めが難しい場合は0次元で）
                data[v] = np.nan
        data_list.append(data)
        ds.close()
    return data_list

def read_global_attr_values(directory, attr_names):
    f = list_netcdf_files_sorted(directory)[0]
    
    ds = xr.open_dataset(os.path.join(directory, f))

    data = {}
    for v in attr_names:
        data[v] = ds.attrs.get(v, np.nan)
    ds.close()
    return data

def stack_by_variable(data_list, varnames):
    """
    data_list（ファイルごと辞書のリスト）を
    変数ごとの2次元配列（[ファイル数, ...shape]）に変換して返すdict。
    欠損（np.nan）が混じる場合は型やshapeに注意（object型で返る場合も）。
    """
    #print(data_list)
    #input("stop")
    result = {}
    for v in varnames:
        # ファイルごとに該当変数をリストアップ
        vals = [d[v] if not isinstance(d[v], float) or not np.isnan(d[v]) else None for d in data_list]
        # shape推定（すべて欠損の場合はスキップ）
        if all(x is None for x in vals):
            result[v] = np.array([np.nan]*len(vals))
        else:
            # 欠損値が混じる場合は、最初に見つかった非欠損shapeに揃えてnp.nanで埋める
            valid = [x for x in vals if x is not None]
            target_shape = valid[0].shape if valid else ()
            arr = []
            for x in vals:
                if x is None:
                    arr.append(np.full(target_shape, np.nan))
                else:
                    arr.append(x)
            result[v] = np.stack(arr)
    return result

def convert_to_standard_shapes(stacked, var_time="time", var_alt="altitude"):
    """
    stacked: stack_by_variableの出力dict
    各変数を1次元or2次元配列に揃えて返す
    """
    result = {}
    # 時間
    t_array = stacked[var_time]
    if t_array.ndim == 2:
        t_array = np.squeeze(t_array)  # (nt,)
    result["time"] = t_array

    # 高度（常に2D: (nt, nz) に統一）
    altitude = stacked[var_alt]
    altitude = np.squeeze(altitude)
    nt = t_array.shape[0]

    if altitude.ndim == 2:
        if altitude.shape[0] == nt:
            # (nt, nz): そのまま（全時刻同一でも2D維持）
            pass
        elif altitude.shape[1] == nt:
            # (nz, nt) → 転置して (nt, nz)
            altitude = altitude.T
        else:
            raise ValueError(
                f"altitude の形状が想定外です: {altitude.shape}. 期待: (nt, nz) または (nz, nt) または (nz,)"
            )
    elif altitude.ndim == 1:
        # (nz,) → 全時刻同一としてタイルして (nt, nz)
        altitude = np.tile(altitude, (nt, 1))
    else:
        raise ValueError(
            f"altitude の次元数が想定外です: ndim={altitude.ndim}. 期待: 1D or 2D"
        )

    result["altitude"] = altitude


    # その他の変数（u_bar, theta_bar など）
    for key in stacked:
        if key in [var_time, var_alt]:
            continue
        arr = stacked[key]
        arr = np.squeeze(arr)
        # 3次元なら (nt, nz, 1) → (nt, nz)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        result[key] = arr

    return result


def select_range(
    stacked_vars,
    varname,
    all_time_array,
    all_altitude_array,
    t_min=None,
    t_max=None,
    z_min=None,
    z_max=None
):
    """
    stacked_vars: stack_by_variable()の出力dict
    varname: 取り出したい変数名（例:"u_bar"）
    all_time_array: time軸（1次元np配列）
    all_altitude_array: altitude軸（1次元np配列）
    t_min, t_max: 時間[s]で抽出したい最小・最大（Noneなら制限なし）
    z_min, z_max: 高度[m]で抽出したい最小・最大（Noneなら制限なし）

    → 抜き出した部分配列、対応する時間配列、高度配列を返す
    """
    data2d = stacked_vars[varname]  # [ファイル数, 空間数]の2次元配列
    # 時間軸index抽出
    t_mask = np.ones_like(all_time_array, dtype=bool)
    if t_min is not None:
        t_mask &= all_time_array >= t_min
    if t_max is not None:
        t_mask &= all_time_array <= t_max
    # 高度軸index抽出
    z_mask = np.ones_like(all_altitude_array, dtype=bool)
    if z_min is not None:
        z_mask &= all_altitude_array >= z_min
    if z_max is not None:
        z_mask &= all_altitude_array <= z_max
    # 抽出
    sub_data2d = data2d[np.ix_(t_mask, z_mask)]
    sub_time = all_time_array[t_mask]
    sub_altitude = all_altitude_array[z_mask]
    return sub_data2d, sub_time, sub_altitude

"""
# ---- 使用例 ----
# output_dir = "output/Analytical/results_ver01/Mars/..."
# varnames = ["u_bar", "theta_bar", "K"]
# data_list = load_all_data(output_dir, varnames)
# stacked = stack_by_variable(data_list, varnames)
# print(stacked["u_bar"].shape)  # → (ファイル数, ...空間方向)

# データを準備
data_list = load_all_data(output_dir, varnames)
stacked = stack_by_variable(data_list, varnames)
all_time = ...         # ファイルから全timeをまとめたnp配列
all_altitude = ...     # どれかのファイルからaltitude（共通）

# 例: 最後の1日分（t >= 3*86400, t <= 4*86400）、高度0〜2000mだけ
sub_data, sub_time, sub_alt = select_range(
    stacked, "u_bar", all_time, all_altitude,
    t_min=3*86400, t_max=4*86400, z_min=0, z_max=2000
)
# sub_data.shape: [抽出した時間数, 抽出した高度数]
"""

def select_range_by_idx(
    stacked_vars,
    all_time_array,       # (nt,)
    all_altitude_array,   # (nt, nz) 前提（2次元のみ想定）
    t_min=None,
    t_max=None,
    z_min_idx=None,
    z_max_idx=None,
):
    """
    最小版: 入力は 2次元のみを想定。stacked_vars 内の (nt, nz) 配列すべてを同じ範囲でクロップする。
    - time は (nt,) を想定して t_min/t_max でマスク
    - altitude は (nt, nz) を想定して z_min_idx..z_max_idx（両端含む）で列を抽出
    - 戻り値の sub_alt は altitude[0, :] の切り出し（先頭行を代表に採用）
    """
    import numpy as np

    t = np.asarray(all_time_array)            # (nt,)
    alt2d = np.asarray(all_altitude_array)    # (nt, nz)
    nt, nz = alt2d.shape

    # 時間マスク（値ベース）
    t_mask = np.ones_like(t, dtype=bool)
    if t_min is not None:
        t_mask &= (t >= t_min)
    if t_max is not None:
        t_mask &= (t <= t_max)

    # 高度インデックス範囲（両端含む）
    if z_min_idx is None:
        z_min_idx = 0
    if z_max_idx is None:
        z_max_idx = nz - 1
    if z_min_idx < 0:
        z_min_idx += nz
    if z_max_idx < 0:
        z_max_idx += nz
    z_idx = np.arange(z_min_idx, z_max_idx + 1)

    # 出力用の time / altitude（altitude は先頭行を代表に）
    sub_time = t[t_mask]                  # (nt_sub,)
    #sub_alt  = alt2d[0, z_idx]            # (nz_sub,)
    sub_alt  = alt2d[np.ix_(t_mask, z_idx)]  # (nt_sub, nz_sub)

    # 全キーに一括適用（2次元のみ想定）
    new_stacked = {}
    for k, v in stacked_vars.items():
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] == nt and v.shape[1] == nz:
            new_stacked[k] = v[np.ix_(t_mask, z_idx)]
        else:
            new_stacked[k] = v  # 2次元以外/形状不一致はそのまま

    # time/altitude キーがある場合は更新（任意）
    if "time" in new_stacked:
        new_stacked["time"] = sub_time
    if "altitude" in new_stacked:
        new_stacked["altitude"] = sub_alt

    return new_stacked#, sub_time, sub_alt
