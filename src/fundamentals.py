"""
基本面分析模块
"""


class FundamentalAnalyzer:

    @staticmethod
    def calculate_indicators(financial_data: dict) -> dict:
        return {
            "gross_margin":   financial_data.get("gross_margin", 0),
            "net_margin":     financial_data.get("net_margin", 0),
            "roe":            financial_data.get("roe", 0),
            "revenue_growth": financial_data.get("revenue_growth", 0),
            "profit_growth":  financial_data.get("profit_growth", 0),
            "pe":             financial_data.get("pe", 0),
            "pb":             financial_data.get("pb", 0),
            "main_net_flow":  financial_data.get("main_net_flow", 0),
        }

    @staticmethod
    def get_valuation_score(pe, pb) -> float:
        score = 50
        pe = pe or 0
        pb = pb or 0

        if pe < 0:
            score -= 20
        elif 0 < pe < 20:
            score += 20
        elif pe < 50:
            score += 10
        elif pe > 100:
            score -= 20

        if 0 < pb < 2:
            score += 15
        elif pb < 5:
            score += 10
        elif pb > 10:
            score -= 15

        return max(0.0, min(100.0, score))

    @staticmethod
    def get_momentum_score(price_change_5d: float, price_change_20d: float, vol_ratio: float) -> float:
        score = 50

        if price_change_5d > 5:
            score += 15
        elif price_change_5d < -5:
            score -= 15

        if price_change_20d > 10:
            score += 15
        elif price_change_20d < -10:
            score -= 15

        if vol_ratio > 2:
            score += 10
        elif vol_ratio < 0.5:
            score -= 10

        return max(0.0, min(100.0, score))
