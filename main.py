#!/usr/bin/env python3
"""
量化交易框架主程序
用法：
    python main.py                      # 使用今日日期
    python main.py --date 2026-04-09    # 指定日期
    python main.py --stock 300244       # 只分析单只股票
"""
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collector import StockDataCollector
from indicators import TechnicalIndicators
from fundamentals import FundamentalAnalyzer
from ai_scorer import AIStockScorer
from reporter import DailyReportGenerator
from chip_analyzer import ChipAnalyzer


LOG_DIR = Path(__file__).parent / "logs"


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
    # 将筹码信号并入主信号列表
    data["signals"] = data.get("signals", []) + chip["chip_signals"]

    # 5. AI 评分
    data["probability"] = AIStockScorer.calculate_probability(data)
    data.update(AIStockScorer.get_detailed_scores(data))

    return data


def main():
    parser = argparse.ArgumentParser(description="量化框架每日复盘")
    parser.add_argument("--date",      default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--watchlist", default="config/watchlist.json")
    parser.add_argument("--stock",     default=None, help="只分析单只股票代码")
    args = parser.parse_args()

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
            log(f"  ✅ {name}: 概率={data['probability']}%  信号={data['signals']}")
        except Exception as e:
            log(f"  ❌ {name}({code}) 失败: {e}\n{traceback.format_exc()}")

    if not results:
        log("无有效数据，退出")
        return

    # 保存报告
    report_path = reporter.save()
    log(f"报告已保存: {report_path}")

    # 控制台排行
    print("\n" + "=" * 50)
    print(f"📊 明日上涨概率排行  [{args.date}]")
    print("=" * 50)
    for i, d in enumerate(sorted(results, key=lambda x: x["probability"], reverse=True), 1):
        bar   = "█" * int(d["probability"] / 10) + "░" * (10 - int(d["probability"] / 10))
        emoji = "🟢" if d["probability"] >= 60 else ("🟡" if d["probability"] >= 45 else "🔴")
        print(f"{i}. {d['name']:8s} {bar} {d['probability']:5.1f}%  {emoji}  信号:{','.join(d['signals']) or '—'}")
    print("=" * 50)
    print(f"\n完整报告：{report_path}\n")


if __name__ == "__main__":
    main()
