#!/usr/bin/env python3
"""
量化交易框架主程序
用法：
    python main.py                      # 使用今日日期
    python main.py --date 2026-04-09    # 指定日期
    python main.py --stock 300244       # 只分析单只股票
    python main.py --backtest --days 60 # 历史回测
"""
import sys
import json
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collector import StockDataCollector
from indicators import TechnicalIndicators
from fundamentals import FundamentalAnalyzer
from ai_scorer import AIStockScorer
from reporter import DailyReportGenerator
from chip_analyzer import ChipAnalyzer
from risk_control import RiskControl
from sector_analyzer import SectorAnalyzer, SECTOR_MAP


LOG_DIR = Path(__file__).parent / "reports"


def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "quant.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_watchlist(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["stocks"]


def analyze_stock(collector: StockDataCollector, code: str, name: str, date: str) -> dict:
    log(f"  采集 {name}({code})...")

    # 1. 行情数据
    data = collector.get_stock_data(code, date)
    data["name"] = name

    # 2. 技术指标
    data = TechnicalIndicators.calculate_all(data)
    data["signals"] = TechnicalIndicators.get_trend_signal(data)

    # 3. 基本面
    fin = collector.get_financial_data(code)
    data["indicators"] = FundamentalAnalyzer.calculate_indicators(fin)

    # 4. 筹码集中度分析
    chip = ChipAnalyzer.analyze(code, data["price"])
    data.update(chip)
    data["signals"] = data.get("signals", []) + chip["chip_signals"]

    # 5. AI 评分
    data["probability"] = AIStockScorer.calculate_probability(data)
    data.update(AIStockScorer.get_detailed_scores(data))

    # 6. 风控建议
    data["risk"] = RiskControl.get_advice(data, data["probability"])

    # 7. 板块信息
    sec = SECTOR_MAP.get(code, {})
    data["sector"] = sec.get("sector", "未知")
    data["concepts"] = sec.get("concept", [])

    return data


def load_yesterday_verify(date: str, stocks: list) -> list:
    """加载昨日预测验证数据"""
    from datetime import datetime as dt
    try:
        d = dt.strptime(date, "%Y-%m-%d")
        # 找上一个交易日（简单减去 1-3 天）
        for i in range(1, 4):
            prev_date = (d - timedelta(days=i)).strftime("%Y-%m-%d")
            prev_path = LOG_DIR / f"{prev_date.replace('-', '')}_report.json"
            if prev_path.exists():
                with open(prev_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                verify = []
                for s in stocks:
                    code, name = s
                    prev_stock = next((p for p in prev_data if p["code"] == code), None)
                    if prev_stock:
                        verify.append({
                            "name": name,
                            "code": code,
                            "pred_prob": prev_stock.get("probability", 50),
                            "actual_change": s[2] if len(s) > 2 else 0,  # 需要传入实际涨跌
                        })
                return verify
    except Exception:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(description="量化框架每日复盘")
    parser.add_argument("--date",      default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--watchlist", default="config/watchlist.json")
    parser.add_argument("--stock",     default=None, help="只分析单只股票代码")
    parser.add_argument("--backtest",  action="store_true", help="历史回测模式")
    parser.add_argument("--days",      default=60, type=int, help="回测天数")
    args = parser.parse_args()

    # 回测模式
    if args.backtest:
        from backtester import Backtester
        bt = Backtester(args.watchlist)
        results = bt.run(days=args.days, single_stock=args.stock)
        bt.print_summary(results)
        report_path = bt.save_report(results)
        log(f"回测报告已保存: {report_path}")
        return

    log(f"===== 量化框架启动 | {args.date} =====")

    stocks = load_watchlist(args.watchlist)
    if args.stock:
        stocks = [(c, n) for c, n in stocks if c == args.stock]
        if not stocks:
            stocks = [(args.stock, args.stock)]

    collector = StockDataCollector()
    reporter  = DailyReportGenerator(args.date)
    results   = []

    for code, name in stocks:
        try:
            data = analyze_stock(collector, code, name, args.date)
            reporter.add_stock(data)
            results.append(data)
            risk = data.get("risk", {})
            log(f"  ✅ {name}: 概率={data['probability']}%  风险={risk.get('risk_label', '—')}  信号={data['signals']}")
        except Exception as e:
            log(f"  ❌ {name}({code}) 失败: {e}\n{traceback.format_exc()}")

    if not results:
        log("无有效数据，退出")
        return

    # 板块分析
    sector_analysis = SectorAnalyzer.analyze(results)
    sector_report = SectorAnalyzer.generate_report(sector_analysis)

    # 保存报告（含板块分析）
    report_path = reporter.save_with_sector(sector_report)
    log(f"报告已保存: {report_path}")

    # 保存 JSON 数据（供次日验证用）
    json_path = LOG_DIR / f"{args.date.replace('-', '')}_report.json"
    json_data = []
    for d in results:
        json_data.append({
            "code": d["code"],
            "name": d["name"],
            "probability": d["probability"],
            "price": d["price"],
            "change_pct": d.get("change_pct", 0),
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 控制台排行
    print("\n" + "=" * 70)
    print(f"📊 明日上涨概率排行  [{args.date}]")
    print("=" * 70)
    for i, d in enumerate(sorted(results, key=lambda x: x["probability"], reverse=True), 1):
        bar   = "█" * int(d["probability"] / 10) + "░" * (10 - int(d["probability"] / 10))
        emoji = "🟢" if d["probability"] >= 60 else ("🟡" if d["probability"] >= 45 else "🔴")
        risk_label = d.get("risk", {}).get("risk_label", "")
        print(f"{i}. {d['name']:8s} {bar} {d['probability']:5.1f}%  {emoji}  [{risk_label}]  信号:{','.join(d['signals']) or '—'}")

    # 板块排行
    if sector_analysis.get("sectors"):
        print("\n" + "─" * 70)
        print("🏭 板块平均上涨概率")
        print("─" * 70)
        for name, sec in sorted(sector_analysis["sectors"].items(), key=lambda x: x[1]["avg_prob"], reverse=True):
            print(f"  {name:12s}  {sec['avg_prob']:.1f}%  ({sec['count']}只)")

    print("=" * 70)
    print(f"\n完整报告：{report_path}\n")


if __name__ == "__main__":
    main()
