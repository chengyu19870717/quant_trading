"""
技术指标计算模块 - 基于 pandas，无需 ta-lib
"""
import pandas as pd
import numpy as np


class TechnicalIndicators:

    @staticmethod
    def calculate_all(data: dict) -> dict:
        """对 data['hist'] 计算全部指标，结果写回 data"""
        df = data["hist"].copy()
        df = TechnicalIndicators.calculate_ma(df)
        df = TechnicalIndicators.calculate_macd(df)
        df = TechnicalIndicators.calculate_kdj(df)
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        df = TechnicalIndicators.calculate_volume_ratio(df)
        data["hist"] = df

        # 把最新一行指标写到 data 顶层，方便评分器直接取用
        latest = df.iloc[-1]
        for col in ["ma5","ma10","ma20","ma30","ma60",
                    "dif","dea","macd_hist",
                    "kdj_k","kdj_d","kdj_j",
                    "bb_upper","bb_middle","bb_lower",
                    "vol_ratio"]:
            if col in df.columns:
                data[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        # 距离 MA20 的偏离率
        if data.get("ma20") and data["ma20"] != 0:
            data["distance_from_ma20"] = (data["price"] - data["ma20"]) / data["ma20"] * 100
        else:
            data["distance_from_ma20"] = 0

        return data

    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods=(5, 10, 20, 30, 60)) -> pd.DataFrame:
        for p in periods:
            df[f"ma{p}"] = df["收盘"].rolling(window=p).mean()
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        df["ema_fast"] = df["收盘"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["收盘"].ewm(span=slow, adjust=False).mean()
        df["dif"]      = df["ema_fast"] - df["ema_slow"]
        df["dea"]      = df["dif"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = (df["dif"] - df["dea"]) * 2
        return df

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n=9, m1=3, m2=3) -> pd.DataFrame:
        low_n  = df["最低"].rolling(window=n, min_periods=1).min()
        high_n = df["最高"].rolling(window=n, min_periods=1).max()
        denom  = high_n - low_n
        rsv    = np.where(denom == 0, 50, (df["收盘"] - low_n) / denom * 100)
        rsv_s  = pd.Series(rsv, index=df.index)
        df["kdj_k"] = rsv_s.ewm(com=m1 - 1, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(com=m2 - 1, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
        df["bb_middle"] = df["收盘"].rolling(window=window).mean()
        df["bb_std"]    = df["收盘"].rolling(window=window).std()
        df["bb_upper"]  = df["bb_middle"] + num_std * df["bb_std"]
        df["bb_lower"]  = df["bb_middle"] - num_std * df["bb_std"]
        return df

    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
        df["vol_ma5"]   = df["成交量"].rolling(window=5).mean()
        df["vol_ratio"] = df["成交量"] / df["vol_ma5"].replace(0, np.nan)
        df["vol_ratio"] = df["vol_ratio"].fillna(1.0)
        return df

    @staticmethod
    def get_trend_signal(data: dict) -> list:
        signals = []
        df = data["hist"]

        if len(df) < 3:
            return signals

        # 均线多头排列
        ma5, ma10, ma20 = data.get("ma5",0), data.get("ma10",0), data.get("ma20",0)
        if ma5 and ma10 and ma20 and ma5 > ma10 > ma20:
            signals.append("MA_BULLISH")
        elif ma5 and ma10 and ma20 and ma5 < ma10 < ma20:
            signals.append("MA_BEARISH")

        # MACD 金叉/死叉（最近两根）
        if len(df) >= 2:
            dif_cur, dif_pre = df["dif"].iloc[-1], df["dif"].iloc[-2]
            dea_cur, dea_pre = df["dea"].iloc[-1], df["dea"].iloc[-2]
            if pd.notna(dif_cur) and pd.notna(dea_cur):
                if dif_cur > dea_cur and dif_pre <= dea_pre:
                    signals.append("MACD_GOLDEN")
                elif dif_cur < dea_cur and dif_pre >= dea_pre:
                    signals.append("MACD_DEAD")

        # KDJ 超买/超卖
        k = data.get("kdj_k", 50)
        if k > 80:
            signals.append("KDJ_OVERBOUGHT")
        elif k < 20:
            signals.append("KDJ_OVERSOLD")

        # 布林带突破
        price = data["price"]
        bb_upper = data.get("bb_upper", 0)
        bb_lower = data.get("bb_lower", 0)
        if bb_upper and price > bb_upper:
            signals.append("BB_UPPER_BREAK")
        elif bb_lower and price < bb_lower:
            signals.append("BB_LOWER_BREAK")

        return signals
