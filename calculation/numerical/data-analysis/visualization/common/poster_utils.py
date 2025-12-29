"""
Poster utilities:
- set_poster_fonts(): フォントを一括設定（タイトル/軸ラベル/目盛）
- poster_out():      通常パス -> *_poster.pdf に変換
- save_poster():     *_poster.pdf を保存（pcolormesh 等は自動ラスタライズ）
- apply_poster_mode(enabled=...): True の時だけ set_poster_fonts を適用
- save_if_poster(..., enabled=...): True の時だけ save_poster で保存
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh  # pcolormesh が返す

__all__ = [
    "set_poster_fonts",
    "poster_out",
    "save_poster",
    "apply_poster_mode",
    "save_if_poster",
]
_POSTER_ENABLED = False

def set_poster_fonts(*, title: int = 24, label: int = 20, tick: int = 18, legend: Optional[int] = None) -> None:
    if legend is None:
        legend = tick
    mpl.rcParams.update({
        "axes.titlesize": title,
        "axes.labelsize":  label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "figure.titlesize": title,
        # 印刷/編集向け（必要に応じて）
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

def poster_out(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + "_poster.pdf")

def _auto_rasterize_axes(ax: Axes) -> None:
    for coll in ax.collections:
        if isinstance(coll, QuadMesh):  # pcolormesh 等
            coll.set_rasterized(True)

def save_poster(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 500,
    rasterize: Literal["auto", "off"] = "auto",
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
) -> Path:
    if rasterize == "auto":
        for ax in fig.axes:
            _auto_rasterize_axes(ax)
    out = poster_out(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return out

# ---- ここから「フラグ有効時のみ」の薄いラッパ ----
def is_poster_enabled() -> bool:
    return _POSTER_ENABLED

def apply_poster_mode(enabled: bool, *, title=24, label=20, tick=18) -> None:
    """enabled=True の時だけフォント設定を適用。False なら何もしない。"""
    global _POSTER_ENABLED
    _POSTER_ENABLED = bool(enabled)          # ← フラグを保持
    if enabled:
        set_poster_fonts(title=title, label=label, tick=tick)

def save_if_poster(fig: plt.Figure, path: str | Path, *, enabled: bool | None = None, dpi: int = 500):
    """enabled=True の時だけ *_poster.pdf を保存。None の場合はグローバル設定を使う。"""
    if enabled is None:
        enabled = is_poster_enabled()        # ← 省略可に
    if enabled:
        return save_poster(fig, path, dpi=dpi)
    return None

def poster_scale(default: float = 1.5) -> float:
    """ポスター有効時は既定スケール(1.5)、無効時は 1.0 を返す。"""
    return default if is_poster_enabled() else 1.0
