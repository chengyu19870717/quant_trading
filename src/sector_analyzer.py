"""
板块分析模块 — 板块轮动 / 行业分类
"""


# 监控股票的行业板块映射
SECTOR_MAP = {
    "300244": {"name": "迪安诊断", "sector": "医疗服务", "concept": ["体外诊断", "第三方检测"]},
    "301396": {"name": "宏景科技", "sector": "计算机应用", "concept": ["智慧城市", "信息化"]},
    "300364": {"name": "中文在线", "sector": "传媒", "concept": ["数字阅读", "IP运营", "AI语料"]},
    "603881": {"name": "数据港",   "sector": "计算机设备", "concept": ["数据中心", "云计算", "算力"]},
    "002173": {"name": "创新医疗", "sector": "医疗服务", "concept": ["综合医院", "康复医疗"]},
}


class SectorAnalyzer:

    @staticmethod
    def analyze(stocks: list) -> dict:
        """
        分析板块分布和轮动
        stocks: [{code, name, probability, change_pct, ...}]
        返回: {sectors: {sector_name: {stocks: [], avg_prob, avg_change, count}}}
        """
        sectors = {}
        for s in stocks:
            code = s.get("code", "")
            info = SECTOR_MAP.get(code, {"name": s.get("name", code), "sector": "未知", "concept": []})
            sector = info["sector"]
            if sector not in sectors:
                sectors[sector] = {"stocks": [], "avg_prob": 0, "avg_change": 0, "count": 0, "concepts": []}
            sectors[sector]["stocks"].append({
                "code": code,
                "name": info["name"],
                "probability": s.get("probability", 50),
                "change_pct": s.get("change_pct", 0),
            })
            sectors[sector]["count"] += 1
            for c in info.get("concept", []):
                if c not in sectors[sector]["concepts"]:
                    sectors[sector]["concepts"].append(c)

        # 计算板块平均值
        for name, sec in sectors.items():
            if sec["count"] > 0:
                sec["avg_prob"] = round(sum(s["probability"] for s in sec["stocks"]) / sec["count"], 1)
                sec["avg_change"] = round(sum(s["change_pct"] for s in sec["stocks"]) / sec["count"], 2)

        return {"sectors": sectors}

    @staticmethod
    def generate_report(analysis: dict) -> str:
        """生成板块分析 Markdown 区块"""
        sectors = analysis.get("sectors", {})
        if not sectors:
            return ""

        lines = ["## 🏭 板块分布\n"]
        lines.append("| 板块 | 标的数量 | 平均上涨概率 | 平均涨跌幅 | 相关概念 |")
        lines.append("|---|---|---|---|---|")

        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]["avg_prob"], reverse=True)
        for name, sec in sorted_sectors:
            concepts = " / ".join(sec.get("concepts", []))
            stocks_str = ", ".join(f"{s['name']}({s['probability']}%)" for s in sec["stocks"])
            lines.append(f"| {name} | {sec['count']} | {sec['avg_prob']}% | {sec['avg_change']:+.2f}% | {concepts} |")
            lines.append(f"| | {stocks_str} | | | |")

        lines.append("")
        return "\n".join(lines)
