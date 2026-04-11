"""
历史回测模块 — 验证评分体系有效性

用法:
    python -m src.backtester --days 60
    python -m src.backtester --stock 300244 --days 90
"""
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import StockDataCollector
from indicators import TechnicalIndicators
from fundamentals import FundamentalAnalyzer
from ai_scorer import AIStockScorer
from chip_analyzer import ChipAnalyzer


class Backtester:
    """对监控股票池进行历史回测"""

    def __init__(self, watchlist_path: str = "config/watchlist.json"):
        with open(watchlist_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.stocks = cfg["stocks"]
        self.collector = StockDataCollector()

    def run(self, days: int = 60, single_stock: str = None) -> dict:
        """
        回测最近 N 个交易日
        返回: {stock_code: {dates: [], predictions: [], actuals: [], accuracy: float}}
        """
        stocks = self.stocks
        if single_stock:
            stocks = [(c, n) for c, n in stocks if c == single_stock]
            if not stocks:
                stocks = [(single_stock, single_stock)]

        end_date = datetime.now()
        results = {}

        for code, name in stocks:
            log(f"回测 {name}({code})...")
            # 获取历史行情（多取 20 天作为缓冲）
            try:
                data = self.collector.get_stock_data(code, end_date.strftime("%Y-%m-%d"))
                hist = data["hist"]
                if hist is None or hist.empty or len(hist) < days + 20:
                    log(f"  ⚠️ 历史数据不足（{len(hist) if hist is not None else 0} 行），跳过")
                    continue
            except Exception as e:
                log(f"  ❌ 获取数据失败: {e}")
                continue

            # 取最近 days 个交易日
            test_hist = hist.tail(days + 1).reset_index(drop=True)
            dates, predictions, actuals = [], [], []

            for i in range(1, len(test_hist)):
                try:
                    # 模拟当日可用数据（用 i 天前的历史）
                    window = test_hist.iloc[:i+1].copy()
                    price = float(window.iloc[-1]["收盘"])
                    prev_close = float(window.iloc[-2]["收盘"])
                    change_pct = (price - prev_close) / prev_close * 100

                    # 计算技术指标
                    tmp_data = {"hist": window, "price": price, "change_pct": change_pct}
                    tmp_data = TechnicalIndicators.calculate_all(tmp_data)
                    tmp_data["signals"] = TechnicalIndicators.get_trend_signal(tmp_data)

                    # 简化基本面（历史回测中不重复请求财务数据）
                    tmp_data["indicators"] = {
                        "gross_margin": 20, "net_margin": 10, "roe": 8,
                        "revenue_growth": 10, "profit_growth": 10,
                    }
                    tmp_data["pe"] = None
                    tmp_data["pb"] = None

                    # 简化资金流向
                    tmp_data["main_net_flow"] = 0
                    tmp_data["circulation_market_cap"] = 0

                    # 简化情绪面
                    tmp_data["turnover_rate"] = float(window.iloc[-1].get("换手率", 0)) * 100
                    tmp_data["vol_ratio"] = float(window.iloc[-1].get("vol_ratio", 1))

                    # 简化筹码
                    tmp_data["chip_signals"] = []
                    tmp_data["chip_profit_ratio"] = 50

                    # 评分
                    prob = AIStockScorer.calculate_probability(tmp_data)

                    # 实际涨跌：次日收盘价 vs 当日收盘价
                    if i + 1 < len(test_hist):
                        next_close = float(test_hist.iloc[i + 1]["收盘"])
                        actual_up = next_close > price
                    else:
                        continue  # 最后一天无次日数据

                    dates.append(str(window.iloc[-1]["日期"])[:10])
                    predictions.append(prob)
                    actuals.append(actual_up)

                except Exception:
                    continue

            if not dates:
                log(f"  ⚠️ 无有效回测数据")
                continue

            # 计算准确率
            pred_binary = [p >= 50 for p in predictions]
            correct = sum(1 for p, a in zip(pred_binary, actuals) if p == a)
            accuracy = correct / len(actuals) * 100 if actuals else 0

            # 计算胜率（预测上涨的股票中实际涨的比例）
            up_preds = [(p, a) for p, a in zip(pred_binary, actuals) if p]
            win_rate = sum(1 for _, a in up_preds if a) / len(up_preds) * 100 if up_preds else 0

            # 计算夏普比率（简化版）
            returns = np.diff(np.array(predictions)) / 100
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0

            results[code] = {
                "name": name,
                "dates": dates,
                "predictions": predictions,
                "actuals": actuals,
                "total_days": len(actuals),
                "accuracy": round(accuracy, 1),
                "win_rate": round(win_rate, 1),
                "sharpe": round(sharpe, 2),
                "avg_prob": round(np.mean(predictions), 1),
                "prob_std": round(np.std(predictions), 1),
            }
            log(f"  ✅ {name}: 准确率={accuracy:.1f}% 胜率={win_rate:.1f}% 夏普={sharpe:.2f} 天数={len(actuals)}")

        return results

    def print_summary(self, results: dict):
        """打印回测汇总"""
        if not results:
            print("无有效回测结果")
            return

        print("\n" + "=" * 70)
        print(f"📊 历史回测报告")
        print("=" * 70)

        all_accuracies = []
        for code, r in results.items():
            print(f"\n  {r['name']}({code})")
            print(f"    回测天数: {r['total_days']} 天")
            print(f"    预测准确率: {r['accuracy']}%")
            print(f"    上涨胜率: {r['win_rate']}%")
            print(f"    夏普比率: {r['sharpe']}")
            print(f"    平均概率: {r['avg_prob']}% (标准差: {r['prob_std']})")
            all_accuracies.append(r['accuracy'])

        print(f"\n  {'─' * 60}")
        print(f"  整体平均准确率: {np.mean(all_accuracies):.1f}%")
        print(f"  最佳标的: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]} ({max(r['accuracy'] for r in results.values())}%)")
        print(f"  最差标的: {min(results.items(), key=lambda x: x[1]['accuracy'])[0]} ({min(r['accuracy'] for r in results.values())}%)")
        print("=" * 70)

    def save_report(self, results: dict, path: str = "reports/backtest.md"):
        """保存回测报告"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        lines = [f"# 📊 历史回测报告\n", f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]

        all_accuracies = []
        for code, r in results.items():
            all_accuracies.append(r['accuracy'])
            lines.append(f"\n## {r['name']} ({code})")
            lines.append(f"| 指标 | 数值 |")
            lines.append(f"|---|---|")
            lines.append(f"| 回测天数 | {r['total_days']} 天 |")
            lines.append(f"| 预测准确率 | {r['accuracy']}% |")
            lines.append(f"| 上涨胜率 | {r['win_rate']}% |")
            lines.append(f"| 夏普比率 | {r['sharpe']} |")
            lines.append(f"| 平均概率 | {r['avg_prob']}% |")
            lines.append(f"| 概率标准差 | {r['prob_std']} |")

        lines.append(f"\n## 汇总")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|---|---|")
        lines.append(f"| 整体平均准确率 | {np.mean(all_accuracies):.1f}% |")
        lines.append(f"| 最佳标的 | {max(results.items(), key=lambda x: x[1]['accuracy'])[0]} |")
        lines.append(f"| 最差标的 | {min(results.items(), key=lambda x: x[1]['accuracy'])[0]} |")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量化框架历史回测")
    parser.add_argument("--days", default=60, type=int, help="回测天数")
    parser.add_argument("--stock", default=None, help="单只股票回测")
    parser.add_argument("--watchlist", default="config/watchlist.json")
    args = parser.parse_args()

    bt = Backtester(args.watchlist)
    results = bt.run(days=args.days, single_stock=args.stock)
    bt.print_summary(results)
    report_path = bt.save_report(results)
    print(f"\n回测报告已保存: {report_path}")
