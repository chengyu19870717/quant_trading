"""
[优化 Phase 2.2] 动态权重调整 - 根据市场温度调整因子权重
根据大盘温度（牛市/熊市/震荡），动态调整技术面、基本面、资金面、情绪面、筹码面的权重
"""


class AdaptiveWeights:
    """
    根据市场条件动态调整因子权重

    市场条件分类：
    - BULL: 大盘上升趋势强劲，技术面信号更可靠 → 技术权重↑，基本面权重↓
    - BEAR: 大盘下行趋势明显，基本面安全性重要 → 基本面权重↑，技术权重↓
    - SHOCK: 大盘震荡无明确方向，资金面和情绪面更重要 → 资金/情绪权重↑
    """

    # 基础权重（作为参考/默认值）
    BASE_WEIGHTS = {
        "technical": 0.30,
        "fundamental": 0.20,
        "money_flow": 0.20,
        "sentiment": 0.15,
        "chip": 0.15,
    }

    # 牛市权重：技术面、资金面强势，看好趋势跟随
    BULL_WEIGHTS = {
        "technical": 0.40,     # ↑ 趋势更可靠
        "fundamental": 0.12,   # ↓ 减少基本面权重
        "money_flow": 0.25,    # ↑ 资金推动明显
        "sentiment": 0.13,     # - 情绪适度
        "chip": 0.10,          # ↓ 筹码面次要
    }

    # 熊市权重：基本面、筹码面更重要，规避风险
    BEAR_WEIGHTS = {
        "technical": 0.15,     # ↓ 技术面信号易失效
        "fundamental": 0.35,   # ↑ 看基本面安全性
        "money_flow": 0.15,    # ↓ 资金面可能虚假
        "sentiment": 0.12,     # ↓ 情绪面悲观
        "chip": 0.23,          # ↑ 筹码锁定很重要
    }

    # 震荡权重：资金面、情绪面、筹码面更重要
    SHOCK_WEIGHTS = {
        "technical": 0.20,     # ↓ 技术面噪音多
        "fundamental": 0.15,   # ↓ 缺少方向感
        "money_flow": 0.28,    # ↑ 资金动向是关键
        "sentiment": 0.22,     # ↑ 情绪反复变化
        "chip": 0.15,          # - 筹码面参考
    }

    @classmethod
    def get_weights(cls, market_condition: str) -> dict:
        """
        根据市场条件获取对应的权重配置

        Args:
            market_condition: "bull"、"bear"、"shock" 中的一个

        Returns:
            权重字典，格式与 BASE_WEIGHTS 相同
        """
        condition = market_condition.lower()

        if condition == "bull":
            weights = cls.BULL_WEIGHTS.copy()
        elif condition == "bear":
            weights = cls.BEAR_WEIGHTS.copy()
        elif condition == "shock":
            weights = cls.SHOCK_WEIGHTS.copy()
        else:
            # 未知条件回退到基础权重
            weights = cls.BASE_WEIGHTS.copy()

        # 验证权重和为 1.0（容错差值 0.001）
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            # 如果权重和不等于 1，重新标准化
            for key in weights:
                weights[key] = weights[key] / weight_sum

        return weights

    @classmethod
    def determine_condition(cls, market_data: dict) -> str:
        """
        根据市场数据判断当前市场条件

        判断逻辑：
        - 沪深300日涨幅 > 1.5% 且净流入 > 0 → BULL
        - 沪深300日跌幅 > 1.5% 或净流入 < -5亿 → BEAR
        - 其他情况 → SHOCK

        Args:
            market_data: 包含以下字段的字典
                - "hs300_change_pct": 沪深300涨跌幅 (%)
                - "north_net_flow": 北向资金净流入 (万元)

        Returns:
            "bull" 或 "bear" 或 "shock"
        """
        hs300_change = market_data.get("hs300_change_pct", 0)
        # north_net_flow 单位：亿元（与 get_market_temperature 返回的 north_flow 一致）
        north_flow = market_data.get("north_net_flow", 0)

        # 牛市判断：大盘上涨且北向资金净流入
        if hs300_change > 1.5 and north_flow > 0:
            return "bull"

        # 熊市判断：大盘下跌或北向资金大幅净流出（>5亿）
        if hs300_change < -1.5 or north_flow < -5:
            return "bear"

        # 默认震荡
        return "shock"

    @classmethod
    def adjust_score_by_condition(cls, score: float, market_condition: str) -> float:
        """
        根据市场条件对原始评分进行微调
        （可选的二级调整，用于强化市场温度的作用）

        Args:
            score: 原始评分 (0-100)
            market_condition: "bull"、"bear"、"shock"

        Returns:
            调整后的评分
        """
        base_score = 50
        deviation = score - base_score

        if market_condition == "bull":
            # 牛市环境：看多信号更强
            adjusted = base_score + deviation * 1.15
        elif market_condition == "bear":
            # 熊市环境：看空信号更强（收缩评分）
            adjusted = base_score + deviation * 0.85
        else:  # shock
            # 震荡环境：维持原样
            adjusted = score

        return max(0.0, min(100.0, adjusted))
