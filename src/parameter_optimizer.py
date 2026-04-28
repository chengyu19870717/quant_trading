"""
[优化 Phase 2.3] 参数自优化 - 基于回测相关性的自动参数调整
周度参数优化：根据最近回测成绩，自动调整得分计算中的各项参数
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class ParameterOptimizer:
    """
    参数自优化引擎

    工作流程：
    1. 每周末后，收集过去7天的回测数据（胜率、收益）
    2. 计算各参数与成绩的相关性
    3. 对相关性高的参数进行微调（+/-5%）
    4. 保存新参数配置，下周使用
    """

    # 可优化的参数列表（技术面、基本面、资金面等）
    OPTIMIZABLE_PARAMS = {
        "technical": [
            "signal_score", "multi_signal_bonus", "ma20_far_penalty", "ma20_mid_penalty"
        ],
        "fundamental": [
            "gm_high_score", "gm_mid_score", "roe_high_score", "roe_mid_score",
            "pe_good_score", "pe_bad_penalty", "pb_good_score"
        ],
        "money_flow": [
            "flow_very_high_score", "flow_high_score", "flow_mid_score",
            "flow_high_penalty", "flow_mid_penalty"
        ],
        "sentiment": [
            "turnover_high_score", "turnover_mid_score", "vol_high_score",
            "vol_mid_score", "change_high_score", "change_mid_score"
        ],
        "chip": [
            "converging_score", "tight_low_profit_score", "wide_low_profit_score"
        ]
    }

    # 调整步长（相对幅度）
    ADJUSTMENT_STEP = 0.05  # ±5%

    # 相关性阈值（只有相关性 > 此值才做调整）
    CORRELATION_THRESHOLD = 0.3

    @classmethod
    def collect_backtest_data(cls, backtest_results: List[Dict]) -> Dict:
        """
        收集回测数据并统计周度指标

        Args:
            backtest_results: 回测结果列表，每项包含：
                {
                    "date": "2026-04-27",
                    "stock_code": "300244",
                    "signal": "strong_buy" / "mild_buy" / "neutral" / "sell",
                    "predicted_prob": 65.5,
                    "actual_return": 2.3,  # 第二天实际涨跌幅 (%)
                    "correct": True  # 是否预测正确（涨跌方向对）
                }
        """
        if not backtest_results:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "strong_buy_rate": 0.0,
            }

        total = len(backtest_results)
        correct = sum(1 for r in backtest_results if r.get("correct", False))
        win_rate = correct / total if total > 0 else 0.0

        returns = [r.get("actual_return", 0) for r in backtest_results]
        avg_return = sum(returns) / len(returns) if returns else 0.0

        strong_buy_count = sum(1 for r in backtest_results if r.get("signal") == "strong_buy")
        strong_buy_rate = strong_buy_count / total if total > 0 else 0.0

        return {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "strong_buy_rate": strong_buy_rate,
            "correct_trades": correct,
        }

    @classmethod
    def calculate_correlation(cls, param_values: List[float], performance: List[float]) -> float:
        """
        计算参数值与性能的皮尔逊相关系数

        Args:
            param_values: 参数的历史值列表
            performance: 对应时期的性能值列表（胜率或平均收益）

        Returns:
            相关系数 (-1.0 ~ 1.0)
        """
        if len(param_values) < 2 or len(param_values) != len(performance):
            return 0.0

        n = len(param_values)
        mean_param = sum(param_values) / n
        mean_perf = sum(performance) / n

        cov = sum((param_values[i] - mean_param) * (performance[i] - mean_perf) for i in range(n))
        var_param = sum((p - mean_param) ** 2 for p in param_values)
        var_perf = sum((p - mean_perf) ** 2 for p in performance)

        if var_param == 0 or var_perf == 0:
            return 0.0

        correlation = cov / (var_param ** 0.5 * var_perf ** 0.5)
        return correlation

    @classmethod
    def adjust_parameter(cls, current_value: float, correlation: float) -> float:
        """
        根据相关系数调整参数值

        逻辑：
        - 正相关：参数增大时性能提升 → 继续增大
        - 负相关：参数增大时性能下降 → 减小参数
        - 低相关性：无调整

        Args:
            current_value: 当前参数值
            correlation: 与性能的相关系数 (-1.0 ~ 1.0)

        Returns:
            调整后的参数值
        """
        if abs(correlation) < cls.CORRELATION_THRESHOLD:
            return current_value  # 相关性太低，不调整

        if correlation > 0:
            # 正相关：增大参数
            adjusted = current_value * (1 + cls.ADJUSTMENT_STEP)
        else:
            # 负相关：减小参数
            adjusted = current_value * (1 - cls.ADJUSTMENT_STEP)

        # 参数下限：0.1（防止参数过小）
        return max(0.1, adjusted)

    @classmethod
    def optimize_weekly(cls, backtest_history: List[Dict], current_params: Dict) -> Dict:
        """
        执行周度参数优化

        Args:
            backtest_history: 过去7天的回测结果列表
            current_params: 当前参数配置（从 scorer_params.json 读取的格式）

        Returns:
            优化后的参数配置
        """
        # 统计周度性能指标
        weekly_perf = cls.collect_backtest_data(backtest_history)
        win_rate = weekly_perf["win_rate"]

        # 如果周度胜率太低，不做优化（防止过度调整）
        if weekly_perf["total_trades"] < 5 or win_rate < 0.3:
            return current_params

        optimized = json.loads(json.dumps(current_params))  # 深拷贝

        # 针对每个可优化的参数进行相关性分析和调整
        for factor, param_names in cls.OPTIMIZABLE_PARAMS.items():
            if factor not in optimized:
                continue

            for param_name in param_names:
                if param_name not in optimized[factor]:
                    continue

                current_value = optimized[factor][param_name]

                # 模拟计算相关性（简化版：基于周度胜率）
                # 实际应用中，这里应该读取历史参数和成绩数据库
                if isinstance(current_value, (int, float)):
                    # 基于周度胜率和参数历史，估算相关系数
                    # 简化逻辑：胜率高时，稍微增大评分参数
                    if win_rate > 0.55:
                        correlation = 0.35  # 中等正相关
                    elif win_rate < 0.45:
                        correlation = -0.35  # 中等负相关
                    else:
                        correlation = 0.0  # 低相关

                    adjusted_value = cls.adjust_parameter(current_value, correlation)
                    optimized[factor][param_name] = round(adjusted_value, 2)

        return optimized

    @classmethod
    def save_optimized_params(cls, optimized_params: Dict, config_path: Path = None):
        """
        保存优化后的参数配置到 JSON 文件

        Args:
            optimized_params: 优化后的参数字典
            config_path: 配置文件路径，默认为 ~/Desktop/quant_trading/config/scorer_params.json
        """
        if config_path is None:
            config_path = Path.home() / "Desktop" / "quant_trading" / "config" / "scorer_params.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 添加优化时间戳
        optimized_params["_last_optimized"] = datetime.now().isoformat()

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(optimized_params, f, indent=2, ensure_ascii=False)

    @classmethod
    def compute_factor_ic(cls, report_dir: Path = None) -> Dict:
        """
        从历史日报 JSON 中计算每个因子的 IC（信息系数）。

        原理：读取 T 日的各维度分数（tech/fund/money/sentiment/chip），
        配对 T+1 日同只股票的实际 change_pct，计算皮尔逊相关系数。
        IC > 0 说明该因子高分 → 次日实际涨，IC 越高因子越有效。

        返回 {factor: ic_value, ...}，并附带样本数和建议权重调整方向。
        """
        if report_dir is None:
            report_dir = Path.home() / "Desktop" / "quant_trading" / "reports"

        # 收集所有日报 JSON，按日期排序
        jsons = sorted(report_dir.glob("*_report.json"))
        if len(jsons) < 2:
            return {}

        # 构建 {date: {code: {scores + change_pct}}} 映射
        daily: Dict[str, Dict[str, Dict]] = {}
        for p in jsons:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                date_key = p.stem[:8]  # 20260428
                daily[date_key] = {
                    d["code"]: {
                        "tech":      d.get("tech_score", 50),
                        "fund":      d.get("fund_score", 50),
                        "money":     d.get("money_score", 50),
                        "sentiment": d.get("sentiment_score", 50),
                        "chip":      d.get("chip_score", 50),
                        "change_pct": d.get("change_pct", 0),
                    }
                    for d in data
                }
            except Exception:
                continue

        sorted_dates = sorted(daily.keys())
        factors = ["tech", "fund", "money", "sentiment", "chip"]
        paired: Dict[str, List] = {f: [] for f in factors}
        returns: List[float] = []

        for i in range(len(sorted_dates) - 1):
            d0, d1 = sorted_dates[i], sorted_dates[i + 1]
            for code, row0 in daily[d0].items():
                if code not in daily[d1]:
                    continue
                actual_ret = daily[d1][code]["change_pct"]
                for f in factors:
                    paired[f].append(row0[f])
                returns.append(actual_ret)

        if not returns:
            return {}

        ic_result = {}
        for f in factors:
            ic = cls.calculate_correlation(paired[f], returns)
            ic_result[f] = round(ic, 4)

        return ic_result

    @classmethod
    def suggest_weight_adjustment(cls, ic: Dict[str, float]) -> Dict[str, str]:
        """
        根据因子 IC 给出权重调整建议（不自动写入，供用户参考）。
        IC > 0.05 → 建议提权；IC < -0.02 → 建议降权；其余中性。
        """
        suggestions = {}
        for factor, val in ic.items():
            if val > 0.05:
                suggestions[factor] = f"↑ 提权（IC={val:.3f}，有效）"
            elif val < -0.02:
                suggestions[factor] = f"↓ 降权（IC={val:.3f}，负向）"
            else:
                suggestions[factor] = f"→ 维持（IC={val:.3f}，弱相关）"
        return suggestions

    @classmethod
    def get_optimization_report(cls, before: Dict, after: Dict) -> str:
        """
        生成参数优化报告（变化摘要）

        Args:
            before: 优化前的参数
            after: 优化后的参数

        Returns:
            人类可读的变化摘要
        """
        changes = []

        for factor in cls.OPTIMIZABLE_PARAMS:
            if factor not in before or factor not in after:
                continue

            for param_name in cls.OPTIMIZABLE_PARAMS[factor]:
                if param_name not in before[factor] or param_name not in after[factor]:
                    continue

                old_val = before[factor][param_name]
                new_val = after[factor][param_name]

                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if old_val != new_val:
                        change_pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                        changes.append(
                            f"  {factor}.{param_name}: {old_val:.2f} → {new_val:.2f} ({change_pct:+.1f}%)"
                        )

        if not changes:
            return "无参数调整（胜率不足或数据不足）"

        report = "参数优化结果：\n" + "\n".join(changes)
        return report
