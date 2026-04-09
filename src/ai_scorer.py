"""
AI 综合评分引擎 - 基于多维度加权计算明日上涨概率
"""


class AIStockScorer:

    WEIGHTS = {
        "technical":   0.30,
        "fundamental": 0.20,
        "money_flow":  0.20,
        "sentiment":   0.15,
        "chip":        0.15,
    }

    @classmethod
    def calculate_probability(cls, data: dict) -> float:
        scores = cls.get_detailed_scores(data)
        total = (
            scores["tech_score"]      * cls.WEIGHTS["technical"] +
            scores["fund_score"]      * cls.WEIGHTS["fundamental"] +
            scores["money_score"]     * cls.WEIGHTS["money_flow"] +
            scores["sentiment_score"] * cls.WEIGHTS["sentiment"] +
            scores["chip_score"]      * cls.WEIGHTS["chip"]
        )
        return round(total, 1)

    @classmethod
    def get_detailed_scores(cls, data: dict) -> dict:
        return {
            "tech_score":      cls._calc_technical_score(data),
            "fund_score":      cls._calc_fundamental_score(data),
            "money_score":     cls._calc_money_flow_score(data),
            "sentiment_score": cls._calc_sentiment_score(data),
            "chip_score":      cls._calc_chip_score(data),
        }

    @staticmethod
    def _calc_technical_score(data: dict) -> float:
        score = 50
        signals = data.get("signals", [])

        bullish = {"MA_BULLISH", "MACD_GOLDEN", "KDJ_OVERSOLD", "BB_LOWER_BREAK"}
        bearish = {"MA_BEARISH", "MACD_DEAD", "KDJ_OVERBOUGHT", "BB_UPPER_BREAK"}

        for sig in signals:
            if sig in bullish:
                score += 8
            elif sig in bearish:
                score -= 8

        # 偏离 MA20 过远扣分
        dist = abs(data.get("distance_from_ma20", 0))
        if dist > 10:
            score -= 10
        elif dist > 5:
            score -= 5

        return max(0.0, min(100.0, score))

    @staticmethod
    def _calc_fundamental_score(data: dict) -> float:
        ind = data.get("indicators", {})
        score = 50

        gm = ind.get("gross_margin", 0)
        if gm > 30:
            score += 15
        elif gm > 20:
            score += 10
        elif gm < 5:
            score -= 10

        roe = ind.get("roe", 0)
        if roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        elif roe < 0:
            score -= 15

        if ind.get("revenue_growth", 0) > 20:
            score += 10
        if ind.get("profit_growth", 0) > 20:
            score += 10

        pe = data.get("pe") or 0
        pb = data.get("pb") or 0
        if 0 < pe < 20:
            score += 10
        elif pe > 100 or pe < 0:
            score -= 10
        if 0 < pb < 2:
            score += 5

        return max(0.0, min(100.0, score))

    @staticmethod
    def _calc_money_flow_score(data: dict) -> float:
        score = 50
        main_net   = data.get("main_net_flow", 0)        # 万元
        circ_cap   = data.get("circulation_market_cap", 0)  # 亿元

        if circ_cap > 0:
            flow_ratio = main_net / (circ_cap * 10000) * 100
        else:
            flow_ratio = 0

        if flow_ratio > 5:
            score += 25
        elif flow_ratio > 3:
            score += 15
        elif flow_ratio > 1:
            score += 5
        elif flow_ratio < -3:
            score -= 20
        elif flow_ratio < -1:
            score -= 10

        return max(0.0, min(100.0, score))

    @staticmethod
    def _calc_sentiment_score(data: dict) -> float:
        score = 50

        turnover = data.get("turnover_rate", 0)
        if turnover > 10:
            score += 15
        elif turnover > 5:
            score += 8
        elif turnover < 1:
            score -= 10

        vol_ratio = data.get("vol_ratio", 1)
        if vol_ratio > 2:
            score += 15
        elif vol_ratio > 1.5:
            score += 8
        elif vol_ratio < 0.5:
            score -= 10

        return max(0.0, min(100.0, score))

    @staticmethod
    def _calc_chip_score(data: dict) -> float:
        """
        筹码集中度评分（满分100，基准50）

        加分规则（均为看多信号）：
          +20  CHIP_CONVERGING：近15天筹码持续收敛
          +20  CHIP_TIGHT_LOW_PROFIT：极紧集中 + 低获利
          +15  CHIP_WIDE_LOW_PROFIT：大范围低获利（套牢盘多，有解套反弹动力）

        辅助微调：
          获利比例越低（筹码越套），在其他条件成立时额外加分
        """
        score = 50
        chip_signals = data.get("chip_signals", [])

        if "CHIP_CONVERGING" in chip_signals:
            score += 20
        if "CHIP_TIGHT_LOW_PROFIT" in chip_signals:
            score += 20
        if "CHIP_WIDE_LOW_PROFIT" in chip_signals:
            score += 15

        # 获利比例极低（<10%）时额外+5，说明筹码大量套牢
        profit = data.get("chip_profit_ratio", 50)
        if chip_signals and profit < 10:
            score += 5

        return max(0.0, min(100.0, score))
