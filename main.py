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


def analyze_stock(collector: StockDataCollector, code: str, name: str, date: str,
                  market_temp: float = 1.0, market_temp_desc: str = "",
                  mkt: dict = None) -> dict:
    log(f"  采集 {name}({code})...")

    # 1. 行情数据
    data = collector.get_stock_data(code, date)
    data["name"] = name

    # 注入市场数据，供 AdaptiveWeights.determine_condition 使用
    if mkt:
        data["hs300_change_pct"] = mkt.get("hs300_change", 0)
        data["north_net_flow"]   = mkt.get("north_flow", 0)   # 亿元

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
    data["probability"] = AIStockScorer.calculate_probability(data, market_temp)
    data.update(AIStockScorer.get_detailed_scores(data))
    data["reason"] = AIStockScorer.generate_reason(data, market_temp_desc)

    # 6. 风控建议
    data["risk"] = RiskControl.get_advice(data, data["probability"])

    # 7. 板块信息
    sec = SECTOR_MAP.get(code, {})
    data["sector"] = sec.get("sector", "未知")
    data["concepts"] = sec.get("concept", [])

    return data


def load_yesterday_verify(date: str, today_results: list) -> list:
    """
    加载昨日预测并与今日实际涨跌对比。
    today_results: 今日已采集完成的 stock data 列表，含 code / change_pct。
    """
    from datetime import datetime as dt
    try:
        today_map = {d["code"]: d.get("change_pct", 0) for d in today_results}
        d = dt.strptime(date, "%Y-%m-%d")
        for i in range(1, 5):   # 最多往前找 4 天（应对节假日连休）
            prev_date = (d - timedelta(days=i)).strftime("%Y-%m-%d")
            prev_path = LOG_DIR / f"{prev_date.replace('-', '')}_report.json"
            if not prev_path.exists():
                continue
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            verify = []
            for prev_stock in prev_data:
                code = prev_stock["code"]
                if code not in today_map:
                    continue
                verify.append({
                    "name":          prev_stock.get("name", code),
                    "code":          code,
                    "pred_prob":     prev_stock.get("probability", 50),
                    "actual_change": today_map[code],   # 今日真实涨跌幅
                })
            if verify:
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

    # 拉取市场温度（统一拉一次，注入每只股票分析）
    log("拉取市场温度...")
    mkt = collector.get_market_temperature()
    log(f"  市场温度: {mkt['temp_desc']} | 系数={mkt['temperature']} | 北向={mkt['north_flow']:.1f}亿")
    reporter.market_temp_desc = mkt["temp_desc"]

    for code, name in stocks:
        try:
            data = analyze_stock(collector, code, name, args.date,
                                 market_temp=mkt["temperature"],
                                 market_temp_desc=mkt["temp_desc"],
                                 mkt=mkt)
            reporter.add_stock(data)
            results.append(data)
            risk = data.get("risk", {})
            log(f"  ✅ {name}: 概率={data['probability']}%  风险={risk.get('risk_label', '—')}  信号={data['signals']}")
        except Exception as e:
            log(f"  ❌ {name}({code}) 失败: {e}\n{traceback.format_exc()}")

    if not results:
        log("无有效数据，退出")
        return

    # 昨日预测验证（今日数据采集完成后回填实际涨跌）
    verify = load_yesterday_verify(args.date, results)
    if verify:
        reporter.set_yesterday_verify(verify)
        correct = sum(1 for v in verify if (v["pred_prob"] >= 50) == (v["actual_change"] >= 0))
        log(f"  昨日验证: {correct}/{len(verify)} 预测正确 ({correct/len(verify)*100:.0f}%)")

    # 板块分析
    sector_analysis = SectorAnalyzer.analyze(results)
    sector_report = SectorAnalyzer.generate_report(sector_analysis)

    # 保存报告（含板块分析）
    report_path = reporter.save_with_sector(sector_report)
    log(f"报告已保存: {report_path}")

    # 保存 JSON 数据（供 investment_hub 展示用）
    json_path = LOG_DIR / f"{args.date.replace('-', '')}_report.json"
    json_data = []
    for d in results:
        risk = d.get("risk", {})
        ind  = d.get("indicators", {})
        json_data.append({
            # ── 基本信息 ──
            "code":            d["code"],
            "name":            d["name"],
            "date":            args.date,
            "market_temp":     mkt.get("temp_desc", ""),
            "sector":          d.get("sector", ""),
            "concepts":        d.get("concepts", [])[:3],

            # ── 行情 ──
            "price":           round(d.get("price", 0), 2),
            "open":            round(d.get("open", 0), 2),
            "high":            round(d.get("high", 0), 2),
            "low":             round(d.get("low", 0), 2),
            "change_pct":      round(d.get("change_pct", 0), 2),
            "volume":          round(d.get("volume", 0), 0),
            "turnover_rate":   round(d.get("turnover_rate", 0), 2),
            "amplitude":       round(d.get("amplitude", 0), 2),
            "main_net_flow":   round(d.get("main_net_flow", 0), 0),
            "float_mv":        round(d.get("float_mv", 0) / 1e8, 2),  # 亿

            # ── 技术指标 ──
            "ma5":             round(d.get("ma5",  0), 2),
            "ma10":            round(d.get("ma10", 0), 2),
            "ma20":            round(d.get("ma20", 0), 2),
            "ma30":            round(d.get("ma30", 0), 2),
            "ma60":            round(d.get("ma60", 0), 2),
            "distance_from_ma20": round(d.get("distance_from_ma20", 0), 2),
            "dif":             round(d.get("dif",  0), 4),
            "dea":             round(d.get("dea",  0), 4),
            "macd_hist":       round(d.get("macd_hist", 0), 4),
            "kdj_k":           round(d.get("kdj_k", 0), 1),
            "kdj_d":           round(d.get("kdj_d", 0), 1),
            "kdj_j":           round(d.get("kdj_j", 0), 1),
            "bb_upper":        round(d.get("bb_upper", 0), 2),
            "bb_middle":       round(d.get("bb_middle", 0), 2),
            "bb_lower":        round(d.get("bb_lower", 0), 2),
            "vol_ratio":       round(d.get("vol_ratio", 0), 2),
            "signals":         [s for s in d.get("signals", []) if isinstance(s, str)],

            # ── 基本面 ──
            # pe/pb 优先取行情数据（ind 里的来自利润表，可能为 0）
            "pe":              round((ind.get("pe") or d.get("pe")) or 0, 1),
            "pb":              round((ind.get("pb") or d.get("pb")) or 0, 2),
            # roe 需要资产负债表，当前版本未采集，保留字段但值为 0 表示未获取
            "roe":             round(ind.get("roe", 0) or 0, 1),
            "gross_margin":    round(ind.get("gross_margin", 0) or 0, 1),
            "net_margin":      round(ind.get("net_margin", 0) or 0, 1),
            "revenue_growth":  round(ind.get("revenue_growth", 0) or 0, 1),
            "profit_growth":   round(ind.get("profit_growth", 0) or 0, 1),

            # ── 筹码（修正 key 名与 chip_analyzer 输出一致）──
            "chip_width":      round(d.get("chip_width_70", 0), 2),
            "low_profit_ratio": round(d.get("chip_profit_ratio", 0), 1),
            "converging_trend": d.get("chip_is_converging", False),

            # ── AI 评分 ──
            "probability":     round(d["probability"], 1),
            "tech_score":      round(d.get("tech_score",  0), 1),
            "fund_score":      round(d.get("fund_score",  0), 1),
            "money_score":     round(d.get("money_score", 0), 1),
            "sentiment_score": round(d.get("sentiment_score", 0), 1),
            "chip_score":      round(d.get("chip_score",  0), 1),
            "reason":          d.get("reason", ""),

            # ── 风控 ──
            "risk_label":      risk.get("risk_label", ""),
            "risk_advice":     risk.get("advice", ""),
            "max_position":    risk.get("max_position", 0),
            "stop_loss":       round(risk.get("stop_loss", 0), 2),
            "take_profit":     round(risk.get("take_profit", 0), 2),
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 控制台排行
    print("\n" + "=" * 70)
    print(f"📊 明日上涨概率排行  [{args.date}]  {mkt['temp_desc']}")
    print("=" * 70)
    for i, d in enumerate(sorted(results, key=lambda x: x["probability"], reverse=True), 1):
        bar   = "█" * int(d["probability"] / 10) + "░" * (10 - int(d["probability"] / 10))
        emoji = "🟢" if d["probability"] >= 60 else ("🟡" if d["probability"] >= 45 else "🔴")
        risk_label = d.get("risk", {}).get("risk_label", "")
        print(f"{i}. {d['name']:8s} {bar} {d['probability']:5.1f}%  {emoji}  [{risk_label}]")
        print(f"   └ {d.get('reason', '—')}")

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
