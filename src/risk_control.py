"""
风控模块 — 止损位/仓位建议/最大回撤控制

用法:
    在 ai_scorer 评分后调用 RiskControl.get_advice(data, probability)
"""


class RiskControl:

    # 风险等级阈值
    RISK_LEVELS = {
        "low":      {"min_prob": 65, "max_position": 0.30, "stop_loss": -0.05},
        "medium":   {"min_prob": 55, "max_position": 0.20, "stop_loss": -0.08},
        "high":     {"min_prob": 45, "max_position": 0.10, "stop_loss": -0.10},
        "danger":   {"min_prob": 0,   "max_position": 0.0,  "stop_loss": -0.15},
    }

    @staticmethod
    def get_advice(data: dict, probability: float) -> dict:
        """
        根据评分和当前数据给出风控建议
        返回: {risk_level, max_position, stop_loss, take_profit, advice}
        """
        risk_level = RiskControl._assess_risk(data, probability)
        cfg = RiskControl.RISK_LEVELS[risk_level]

        # 止损位
        stop_loss = round(data["price"] * (1 + cfg["stop_loss"]), 2)

        # 止盈位（止损的 2 倍）
        take_profit = round(data["price"] * (1 - cfg["stop_loss"] * 2), 2)

        # 建议文案
        advice = RiskControl._gen_advice(risk_level, probability, data)

        return {
            "risk_level": risk_level,
            "risk_label": {
                "low": "低风险",
                "medium": "中风险",
                "high": "高风险",
                "danger": "危险",
            }[risk_level],
            "max_position": cfg["max_position"],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "advice": advice,
        }

    @staticmethod
    def _assess_risk(data: dict, probability: float) -> str:
        """综合评估风险等级"""
        if probability >= 65:
            return "low"
        elif probability >= 55:
            return "medium"
        elif probability >= 45:
            return "high"
        else:
            return "danger"

    @staticmethod
    def _gen_advice(risk_level: str, probability: float, data: dict) -> str:
        signals = data.get("signals", [])
        change_pct = data.get("change_pct", 0)

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

        # 资金流向警告
        main_flow = data.get("main_net_flow", 0)
        if main_flow < -5000:
            parts.append(f"主力资金流出 {abs(main_flow):.0f}万，谨慎")
        elif main_flow > 5000:
            parts.append(f"主力资金流入 {main_flow:.0f}万，关注")

        return "；".join(parts)
