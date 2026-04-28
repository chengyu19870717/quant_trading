"""
风控模块 — 止损位/仓位建议/最大回撤控制

止损位基于 ATR（平均真实波幅）而非固定百分比，自动适配每只股票的
实际波动特征：日均波动大的股票止损宽，日均波动小的止损窄，
避免被正常噪音震出或止损过于宽松。
"""
import pandas as pd


class RiskControl:

    # 风险等级：ATR 倍数 / 最大仓位 / 止盈倍数（相对止损距离）
    RISK_LEVELS = {
        "low":    {"atr_mult": 1.5, "max_position": 0.30, "profit_mult": 2.5},
        "medium": {"atr_mult": 2.0, "max_position": 0.20, "profit_mult": 2.0},
        "high":   {"atr_mult": 2.5, "max_position": 0.10, "profit_mult": 1.5},
        "danger": {"atr_mult": 3.0, "max_position": 0.00, "profit_mult": 1.0},
    }

    # ATR 止损的绝对下限（防止 ATR 异常小时止损过近）和上限（防止止损过宽）
    STOP_FLOOR_PCT = 0.03   # 最小止损距离：3%
    STOP_CEIL_PCT  = 0.15   # 最大止损距离：15%

    @staticmethod
    def get_advice(data: dict, probability: float) -> dict:
        risk_level = RiskControl._assess_risk(data, probability)
        cfg = RiskControl.RISK_LEVELS[risk_level]
        price = data["price"]

        atr = RiskControl._calc_atr(data)
        stop_dist = atr * cfg["atr_mult"]

        # 约束止损距离在合理范围内
        floor = price * RiskControl.STOP_FLOOR_PCT
        ceil  = price * RiskControl.STOP_CEIL_PCT
        stop_dist = max(floor, min(ceil, stop_dist))

        stop_loss   = round(price - stop_dist, 2)
        take_profit = round(price + stop_dist * cfg["profit_mult"], 2)

        return {
            "risk_level": risk_level,
            "risk_label": {
                "low": "低风险", "medium": "中风险",
                "high": "高风险", "danger": "危险",
            }[risk_level],
            "max_position": cfg["max_position"],
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "atr":          round(atr, 3),
            "stop_dist_pct": round(stop_dist / price * 100, 2),
            "advice": RiskControl._gen_advice(risk_level, probability, data, atr, price),
        }

    @staticmethod
    def _calc_atr(data: dict, period: int = 14) -> float:
        """计算14日 ATR，数据不足时 fallback 到价格的3%"""
        hist = data.get("hist")
        if hist is None or len(hist) < period + 1:
            return data.get("price", 0) * 0.03

        df = hist.tail(period + 1).copy()
        prev_close = df["收盘"].shift(1)
        tr = pd.concat([
            df["最高"] - df["最低"],
            (df["最高"] - prev_close).abs(),
            (df["最低"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.iloc[1:].mean())  # 跳过第一行（shift产生的NaN）

    @staticmethod
    def _assess_risk(data: dict, probability: float) -> str:
        if probability >= 65:
            return "low"
        elif probability >= 55:
            return "medium"
        elif probability >= 45:
            return "high"
        else:
            return "danger"

    @staticmethod
    def _gen_advice(risk_level: str, probability: float, data: dict,
                    atr: float, price: float) -> str:
        signals    = data.get("signals", [])
        change_pct = data.get("change_pct", 0)
        main_flow  = data.get("main_net_flow", 0)
        atr_pct    = atr / price * 100 if price else 0

        parts = []

        if risk_level == "low":
            parts.append("评分较高，可适量建仓")
            if "MA_BULLISH" in signals:
                parts.append("均线多头排列，趋势向好")
        elif risk_level == "medium":
            parts.append("评分一般，建议轻仓试探")
            if change_pct < -3:
                parts.append("今日跌幅较大，注意超跌反弹机会")
        elif risk_level == "high":
            parts.append("评分偏低，建议观望")
            if "MA_BEARISH" in signals:
                parts.append("均线空头排列，趋势偏空")
        else:
            parts.append("评分较低，不建议介入")
            if change_pct < -5:
                parts.append("今日大跌，注意风险")

        # 波动率提示
        if atr_pct > 5:
            parts.append(f"日均波动 {atr_pct:.1f}%，属高波动标的，注意仓位控制")
        elif atr_pct < 1:
            parts.append(f"日均波动 {atr_pct:.1f}%，流动性偏低")

        # 资金流向
        if main_flow < -5000:
            parts.append(f"主力持续流出 {abs(main_flow):.0f}万（3日均值），谨慎")
        elif main_flow > 5000:
            parts.append(f"主力持续流入 {main_flow:.0f}万（3日均值），关注")

        return "；".join(parts)
