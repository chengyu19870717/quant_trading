"""
每日复盘报告生成器
"""
from datetime import datetime
from pathlib import Path


REPORT_DIR = Path(__file__).parent.parent / "reports"


class DailyReportGenerator:

    def __init__(self, date: str):
        self.date   = date
        self.stocks = []
        self.yesterday_verify = None  # 昨日预测验证数据

    def add_stock(self, stock_data: dict):
        self.stocks.append(stock_data)

    def set_yesterday_verify(self, verify_data: list):
        """设置昨日预测验证数据: [{name, code, pred_prob, actual_change}]"""
        self.yesterday_verify = verify_data

    def generate_markdown(self) -> str:
        sorted_stocks = sorted(self.stocks, key=lambda x: x.get("probability", 0), reverse=True)

        report = f"""# 📊 每日股票复盘报告

**日期**: {self.date}
**生成时间**: {datetime.now().strftime('%H:%M:%S')}
**监控标的**: {len(self.stocks)} 只

"""
        # 昨日预测验证
        if self.yesterday_verify:
            report += self._generate_verify_section()

        report += """---

## 重点股票分析

"""
        for rank, stock in enumerate(sorted_stocks, 1):
            report += self._format_stock(stock, rank)

        report += self._generate_summary(sorted_stocks)
        return report

    def _generate_verify_section(self) -> str:
        """生成昨日预测验证区块"""
        lines = ["## 📈 昨日预测验证\n"]
        lines.append("| 股票 | 昨日预测上涨概率 | 今日实际涨跌 | 预测结果 |")
        lines.append("|---|---|---|---|")

        correct = 0
        total = 0
        for v in self.yesterday_verify:
            pred = v.get("pred_prob", 50)
            actual = v.get("actual_change", 0)
            pred_up = pred >= 50
            actual_up = actual >= 0
            hit = pred_up == actual_up
            if hit:
                correct += 1
            total += 1
            emoji = "✅" if hit else "❌"
            direction = "▲" if actual_up else "▼"
            lines.append(f"| {v.get('name', '')}({v.get('code', '')}) | {pred:.1f}% | {direction}{abs(actual):.2f}% | {emoji} |")

        acc = correct / total * 100 if total else 0
        lines.append(f"\n> **近期预测准确率**: {acc:.0f}% ({correct}/{total})")
        lines.append("")
        return "\n".join(lines)

    def _format_stock(self, s: dict, rank: int) -> str:
        prob  = s.get("probability", 50)
        emoji = "🟢" if prob >= 60 else ("🟡" if prob >= 45 else "🔴")
        arrow = "▲" if s.get("change_pct", 0) >= 0 else "▼"

        signals_str = ", ".join(s.get("signals", [])) or "无明显信号"

        def stars(score):
            n = round(score / 20)
            return "★" * n + "☆" * (5 - n)

        macd_sig = ""
        if "MACD_GOLDEN" in s.get("signals", []):
            macd_sig = "金叉 ✅"
        elif "MACD_DEAD" in s.get("signals", []):
            macd_sig = "死叉 ❌"
        else:
            macd_sig = "—"

        kdj_sig = ""
        k = s.get("kdj_k", 50)
        if k > 80:
            kdj_sig = f"超买({k:.0f}) ⚠️"
        elif k < 20:
            kdj_sig = f"超卖({k:.0f}) ✅"
        else:
            kdj_sig = f"正常({k:.0f})"

        flow = s.get("main_net_flow", 0)
        flow_str = f"{flow:+.0f}万" if flow != 0 else "—"
        flow_sig = "✅ 净流入" if flow > 0 else ("❌ 净流出" if flow < 0 else "—")

        pe_str = f"{s['pe']:.1f}" if s.get("pe") else "—"
        pb_str = f"{s['pb']:.2f}" if s.get("pb") else "—"

        # ── 筹码部分 ──────────────────────────────────────
        chip_w      = s.get("chip_width_70", 0)
        chip_profit = s.get("chip_profit_ratio", 0)
        chip_low    = s.get("chip_low_70", 0)
        chip_high   = s.get("chip_high_70", 0)
        chip_sigs   = s.get("chip_signals", [])
        chip_slope  = s.get("chip_slope", 0)

        chip_trend_str = f"收敛({'↘ ' + f'{chip_slope:.4f}'})" if s.get("chip_is_converging") else f"扩散({'↗ ' + f'{chip_slope:.4f}'})"

        chip_sig_parts = []
        if "CHIP_CONVERGING"       in chip_sigs: chip_sig_parts.append("持续集中✅")
        if "CHIP_TIGHT_LOW_PROFIT" in chip_sigs: chip_sig_parts.append("极紧+低获利✅")
        if "CHIP_WIDE_LOW_PROFIT"  in chip_sigs: chip_sig_parts.append("极宽+低获利✅")
        chip_sig_str = " / ".join(chip_sig_parts) if chip_sig_parts else "无"

        chip_width_bar = ""
        series = s.get("chip_width_series", [])
        if series:
            mini_max = max(series) if max(series) > 0 else 1
            chip_width_bar = " ".join(
                "▇" if w >= mini_max * 0.75 else ("▅" if w >= mini_max * 0.5 else ("▃" if w >= mini_max * 0.25 else "▁"))
                for w in series
            )

        return f"""
### {rank}. {s['name']} ({s['code']}) {emoji}

**收盘价**: {s['price']:.2f}元 | **涨跌**: {arrow}{abs(s.get('change_pct',0)):.2f}% | **明日上涨概率**: **{prob}%**

| 指标 | 数值 | 信号 |
|------|------|------|
| 主力资金 | {flow_str} | {flow_sig} |
| MACD | DIF={s.get('dif',0):.3f} / DEA={s.get('dea',0):.3f} | {macd_sig} |
| KDJ | K={k:.1f} / D={s.get('kdj_d',50):.1f} | {kdj_sig} |
| 布林带 | 上={s.get('bb_upper',0):.2f} / 下={s.get('bb_lower',0):.2f} | {'突破上轨 ⚠️' if 'BB_UPPER_BREAK' in s.get('signals',[]) else '正常'} |
| 换手率 | {s.get('turnover_rate',0):.2f}% | {'🔥 活跃' if s.get('turnover_rate',0)>5 else '冷清'} |
| 量比 | {s.get('vol_ratio',1):.2f}x | {'放量' if s.get('vol_ratio',1)>1.5 else '缩量' if s.get('vol_ratio',1)<0.7 else '正常'} |
| PE / PB | {pe_str} / {pb_str} | — |

**技术信号**: {signals_str}

#### 筹码集中度分析（近15日）

| 指标 | 数值 |
|------|------|
| 70%筹码区间 | {chip_low:.2f} ~ {chip_high:.2f} 元 |
| 筹码宽度 | {chip_w:.2f}% |
| 获利比例 | {chip_profit:.1f}% |
| 15日收敛趋势 | {chip_trend_str} |
| 筹码信号 | {chip_sig_str} |

> 15日宽度变化趋势：{chip_width_bar}

**综合评分**:
- 技术面: {stars(s.get('tech_score',50))} ({s.get('tech_score',50):.0f})
- 基本面: {stars(s.get('fund_score',50))} ({s.get('fund_score',50):.0f})
- 资金面: {stars(s.get('money_score',50))} ({s.get('money_score',50):.0f})
- 情绪面: {stars(s.get('sentiment_score',50))} ({s.get('sentiment_score',50):.0f})
- 筹码面: {stars(s.get('chip_score',50))} ({s.get('chip_score',50):.0f})

---
"""

    def _generate_summary(self, stocks: list) -> str:
        top    = [s for s in stocks if s.get("probability", 0) >= 60]
        caution= [s for s in stocks if s.get("probability", 0) < 40]

        top_names     = "、".join(f"**{s['name']}**({s['probability']}%)" for s in top) or "无"
        caution_names = "、".join(f"**{s['name']}**" for s in caution) or "无"

        return f"""
## 📌 明日操作建议

| 类型 | 标的 |
|------|------|
| 重点关注（概率≥60%）| {top_names} |
| 谨慎观望（概率<40%）| {caution_names} |

> ⚠️ 本报告为量化模型辅助决策，不构成投资建议。股市有风险，入市需谨慎。

---
*报告由量化框架自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    def save(self) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{self.date}_report.md"
        path.write_text(self.generate_markdown(), encoding="utf-8")
        return path

    def save_with_sector(self, sector_report: str) -> Path:
        """保存含板块分析的报告"""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{self.date}_report.md"
        md = self.generate_markdown()
        # 在重点股票分析前插入板块报告
        if sector_report:
            md = md.replace("---\n\n## 重点股票分析", sector_report + "\n---\n\n## 重点股票分析")
        path.write_text(md, encoding="utf-8")
        return path
