"""
技术指标计算模块 - 基于 pandas，无需 ta-lib
"""
import pandas as pd
import numpy as np


class SignalStrength:
    """
    [优化 Phase 1.3] 量化技术信号的强度
    从布尔（有或无）改为连续值（0.0 ~ 1.0）
    """

    @staticmethod
    def ma_bullish_strength(ma5, ma10, ma20, ma60):
        """均线多头排列的强度评估"""
        if ma5 > ma10 > ma20 > ma60:
            return 1.0  # 完整多头
        elif ma5 > ma10 > ma20:
            return 0.7  # 部分多头
        elif ma5 > ma10:
            return 0.4  # 弱多头
        return 0.0

    @staticmethod
    def macd_strength(dif, dea, dif_pre=None, dea_pre=None):
        """MACD 金叉的强度评估"""
        if dif <= dea:
            return 0.0

        distance = abs(dif - dea)
        strength = min(1.0, distance / 0.5)

        if dif_pre is not None and dif_pre is not None and dif > dif_pre:
            strength = strength * 1.2

        return min(1.0, strength)

    @staticmethod
    def kdj_oversold_strength(k_current, k_previous=None):
        """KDJ 超卖反弹的强度"""
        if k_current > 20:
            return 0.0

        strength = (20 - k_current) / 20

        if k_previous is not None and k_current > k_previous:
            strength = strength * 1.2

        return min(1.0, strength)

    @staticmethod
    def momentum_strength(momentum_20d):
        """20日涨幅的强度评估"""
        if momentum_20d > 10:
            return min(1.0, momentum_20d / 30)
        elif momentum_20d < -10:
            return min(1.0, abs(momentum_20d) / 30)
        else:
            return 0.0


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
        df = TechnicalIndicators.calculate_rsi(df)
        df = TechnicalIndicators.calculate_momentum(df)
        df = TechnicalIndicators.calculate_obv(df)
        data["hist"] = df

        # 把最新一行指标写到 data 顶层，方便评分器直接取用
        latest = df.iloc[-1]

        for col in ["ma5","ma10","ma20","ma30","ma60",
                    "dif","dea","macd_hist",
                    "kdj_k","kdj_d","kdj_j",
                    "bb_upper","bb_middle","bb_lower",
                    "vol_ratio","rsi_14","momentum_20d","obv","obv_slope"]:
            if col in df.columns:
                data[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        # 前一日值（供 SignalStrength 判断方向加速度）
        if len(df) >= 2:
            prev = df.iloc[-2]
            for col, key in [("dif", "dif_pre"), ("dea", "dea_pre"), ("kdj_k", "kdj_k_pre")]:
                data[key] = float(prev[col]) if col in df.columns and pd.notna(prev[col]) else None

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
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        [优化 Phase 2] RSI 相对强弱指数计算
        基于 Wilder 的标准方法
        """
        delta = df["收盘"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50.0)
        return df

    @staticmethod
    def calculate_momentum(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df["momentum_20d"] = (df["收盘"] / df["收盘"].shift(period) - 1) * 100
        df["momentum_20d"] = df["momentum_20d"].fillna(0.0)
        return df

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """OBV 能量潮：成交量 × 方向（上涨+1/下跌-1）累加。
        再提取 10 日线性回归斜率（归一化到价格%），判断聪明资金方向。"""
        direction = np.sign(df["收盘"].diff()).fillna(0)
        df["obv"] = (direction * df["成交量"]).cumsum()

        # 斜率：10日 OBV 线性趋势 / 成交量均值，归一化为相对值
        if len(df) >= 10:
            obv_tail = df["obv"].values[-10:]
            x = np.arange(10)
            slope = float(np.polyfit(x, obv_tail, 1)[0])
            vol_mean = df["成交量"].tail(10).mean()
            df["obv_slope"] = slope / vol_mean if vol_mean > 0 else 0
        else:
            df["obv_slope"] = 0.0
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

        # [优化 Phase 2] RSI 超买超卖信号
        rsi = data.get("rsi_14", 50)
        if rsi > 70:
            signals.append("RSI_OVERBOUGHT")
        elif rsi < 30:
            signals.append("RSI_OVERSOLD")

        return signals
