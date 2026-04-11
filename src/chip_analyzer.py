"""
筹码集中度分析模块

两类看多信号：
  CHIP_CONVERGING      — 近15天70%筹码宽度线性收敛（斜率 < 0）
  CHIP_TIGHT_LOW_PROFIT — 70%筹码在价格±10%带内 + 宽度<7% + 获利比例<20%
  CHIP_WIDE_LOW_PROFIT  — 70%筹码在价格±10%带内 + 宽度>15% + 获利比例<20%
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW = 15          # 观察天数
BAND_PCT = 0.10      # 价格带 ±10%
WIDTH_TIGHT = 7.0    # 集中度"极紧"阈值（%）
WIDTH_WIDE = 15.0    # 集中度"极宽"阈值（%）
PROFIT_MAX = 20.0    # 获利比例上限（%）


class ChipAnalyzer:

    @staticmethod
    def analyze(code: str, current_price: float) -> dict:
        """
        返回筹码分析结果 dict，失败时返回空结构（不抛异常）。
        """
        try:
            df = ChipAnalyzer._fetch(code)
            if df is None or len(df) < 5:
                return ChipAnalyzer._empty()
            return ChipAnalyzer._compute(df, current_price)
        except Exception:
            return ChipAnalyzer._empty()

    # ── 内部方法 ──────────────────────────────────────────

    @staticmethod
    def _fetch(code: str) -> pd.DataFrame:
        import akshare as ak
        # 尝试东财筹码分布
        try:
            df = ak.stock_cyq_em(symbol=code, adjust="qfq")
            if df is not None and not df.empty and len(df) >= 5:
                return df.tail(WINDOW).reset_index(drop=True)
        except Exception:
            pass
        # Fallback: 尝试其他接口
        return None

    @staticmethod
    def _compute(df: pd.DataFrame, price: float) -> dict:
        # 列名兼容：东财接口返回 "70成本-低" / "70成本-高" / "获利比例"
        low_col    = "70成本-低"
        high_col   = "70成本-高"
        profit_col = "获利比例"

        df = df.copy()
        df["chip_width"] = (df[high_col] - df[low_col]) / df[low_col].replace(0, np.nan) * 100
        df["chip_width"] = df["chip_width"].fillna(0)

        latest       = df.iloc[-1]
        width_70     = float(latest["chip_width"])
        profit_ratio = float(latest[profit_col])
        low_70       = float(latest[low_col])
        high_70      = float(latest[high_col])

        # ① 筹码是否在价格 ±10% 带内
        band_low    = price * (1 - BAND_PCT)
        band_high   = price * (1 + BAND_PCT)
        chip_in_band = (low_70 >= band_low) and (high_70 <= band_high)

        # ② 15天收敛趋势（线性回归斜率）
        widths = df["chip_width"].values.astype(float)
        x      = np.arange(len(widths))
        slope  = float(np.polyfit(x, widths, 1)[0]) if len(widths) >= 3 else 0.0
        is_converging = slope < 0

        # ── 信号生成 ───────────────────────────────────────
        signals = []

        # 信号1：持续集中趋势
        if is_converging:
            signals.append("CHIP_CONVERGING")

        # 信号2：极端集中度 + 低获利比例
        if chip_in_band and profit_ratio < PROFIT_MAX:
            if width_70 < WIDTH_TIGHT:
                signals.append("CHIP_TIGHT_LOW_PROFIT")
            elif width_70 > WIDTH_WIDE:
                signals.append("CHIP_WIDE_LOW_PROFIT")

        return {
            "chip_width_70":      round(width_70, 2),
            "chip_profit_ratio":  round(profit_ratio, 2),
            "chip_low_70":        round(low_70, 2),
            "chip_high_70":       round(high_70, 2),
            "chip_in_band":       chip_in_band,
            "chip_slope":         round(slope, 4),
            "chip_is_converging": is_converging,
            "chip_signals":       signals,
            # 最近15天宽度序列，用于报告绘图
            "chip_width_series":  [round(w, 2) for w in widths.tolist()],
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "chip_width_70":      0.0,
            "chip_profit_ratio":  0.0,
            "chip_low_70":        0.0,
            "chip_high_70":       0.0,
            "chip_in_band":       False,
            "chip_slope":         0.0,
            "chip_is_converging": False,
            "chip_signals":       [],
            "chip_width_series":  [],
        }
