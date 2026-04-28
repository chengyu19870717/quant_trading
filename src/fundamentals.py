"""
基本面分析模块
"""


class FundamentalAnalyzer:

    @staticmethod
    def calculate_indicators(financial_data: dict) -> dict:
        """将 data_collector 采集的财务数据透传为标准指标字典"""
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
