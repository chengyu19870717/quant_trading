"""
AI 综合评分引擎 - 基于多维度加权计算明日上涨概率

细项参数从 ~/Desktop/quant_trading/config/scorer_params.json 读取（由程钰的百宝箱同步）。
JSON 不存在时自动回退到代码内置默认值，不影响运行。
"""
import json
from pathlib import Path

# ── 默认细项参数（与 investment_hub 初始化值保持一致） ──────
_DEFAULT_PARAMS = {
    "technical": {
        "signal_score":      20,  # [优化] 从12→20，扩大基线分差
        "multi_signal_bonus": 10,
        "ma20_far_penalty":  15,
        "ma20_mid_penalty":   8,
    },
    "fundamental": {
        "gm_high_score":       15,
        "gm_mid_score":        10,
        "gm_low_penalty":      10,
        "roe_high_score":      15,
        "roe_mid_score":       10,
        "roe_neg_penalty":     15,
        "rev_growth_score":    10,
        "profit_growth_score": 10,
        "pe_good_score":       10,
        "pe_missing_penalty":  15,
        "pe_bad_penalty":      10,
        "pb_good_score":        5,
        "pb_missing_penalty":  10,
    },
    "money_flow": {
        "flow_very_high_score": 30,
        "flow_high_score":      20,
        "flow_mid_score":       10,
        "flow_high_penalty":    25,
        "flow_mid_penalty":     15,
    },
    "sentiment": {
        "turnover_high_score":  15,
        "turnover_mid_score":    8,
        "turnover_low_penalty": 10,
        "vol_high_score":       15,
        "vol_mid_score":         8,
        "vol_low_penalty":      10,
        "change_high_score":    10,
        "change_mid_score":      5,
    },
    "chip": {
        "converging_score":        20,
        "tight_low_profit_score":  20,
        "wide_low_profit_score":   15,
        "low_profit_bonus":         5,
        "narrow_width_bonus":       5,
    },
}

_CONFIG_PATH = Path.home() / "Desktop" / "quant_trading" / "config" / "scorer_params.json"
_cached_params: dict | None = None


def _load_params() -> dict:
    global _cached_params
    if _cached_params is not None:
        return _cached_params
    try:
        if _CONFIG_PATH.exists():
            raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            # 深合并：仅覆盖存在的 key，保留默认值中额外的 key
            merged = {fk: {**_DEFAULT_PARAMS.get(fk, {}), **raw.get(fk, {})}
                      for fk in _DEFAULT_PARAMS}
            _cached_params = merged
            return _cached_params
    except Exception:
        pass
    _cached_params = _DEFAULT_PARAMS
    return _cached_params


def _get_stock_weights(stock_code: str) -> dict:
    """
    获取指定股票的权重配置。
    优先级：股票级覆盖 > 全局默认权重 > 类属性默认值
    
    Args:
        stock_code: 股票代码，如 "300244"
    
    Returns:
        权重的字典，如 {"technical": 0.60, "fundamental": 0.10, ...}
    """
    params = _load_params()
    
    # 尝试获取股票级覆盖配置
    stock_overrides = params.get("_stock_overrides", {})
    if stock_code in stock_overrides:
        stock_config = stock_overrides[stock_code]
        weights = {}
        for factor in ["technical", "fundamental", "money_flow", "sentiment", "chip"]:
            if factor in stock_config and stock_config[factor].get("is_active", True):
                weights[factor] = stock_config[factor]["weight"]
        
        # 检查是否配置完整（5 个因子都有）
        if len(weights) == 5:
            return weights
    
    # 回退到全局默认权重
    try:
        raw = params.get("_weights", {})
        if raw:
            weights = {}
            for factor in ["technical", "fundamental", "money_flow", "sentiment", "chip"]:
                if factor in raw and raw[factor].get("is_active", True):
                    weights[factor] = raw[factor]["weight"]
            
            if len(weights) == 5:
                return weights
    except Exception:
        pass
    
    # 最终回退到类属性默认值
    return AIStockScorer.WEIGHTS.copy()


def normalize_by_rank(stocks_data: list, score_key: str = "probability") -> list:
    """
    为每只股票附加排名位置和相对强弱信号，保留原始概率不覆盖。

    原设计用硬编码 70/50/30 三档覆盖原始概率，会丢失分数差异（90分和51分
    都变成70%）。现改为只附加 rank_position 和 rank_signal，让原始概率保持
    连续性，供前端参考使用。
    """
    if not stocks_data:
        return stocks_data

    sorted_stocks = sorted(stocks_data, key=lambda x: x.get(score_key, 0), reverse=True)
    n = len(sorted_stocks)
    top_cut = max(1, n // 3)
    bot_cut = max(2, n * 2 // 3)

    for i, stock in enumerate(sorted_stocks):
        stock["rank_position"] = i + 1
        if i < top_cut:
            stock["rank_signal"] = "🟢强势"
        elif i < bot_cut:
            stock["rank_signal"] = "🟡中性"
        else:
            stock["rank_signal"] = "🔴弱势"

    return sorted_stocks


class AIStockScorer:

    WEIGHTS = {
        "technical":   0.30,
        "fundamental": 0.20,
        "money_flow":  0.20,
        "sentiment":   0.15,
        "chip":        0.15,
    }

    @classmethod
    def _p(cls, factor: str, key: str) -> float:
        """取细项参数值，自动回退到默认值"""
        return _load_params().get(factor, {}).get(key, _DEFAULT_PARAMS[factor][key])

    @classmethod
    def _get_weights(cls) -> dict:
        """从配置文件读取因子主权重，若无配置则用类属性默认值"""
        try:
            raw = _load_params().get("_weights", {})
            if raw:
                return {fk: v["weight"] for fk, v in raw.items() if v.get("is_active", True)}
        except Exception:
            pass
        return cls.WEIGHTS

    @classmethod
    def reload_params(cls):
        """强制重新加载参数（同步后调用）"""
        global _cached_params
        _cached_params = None

    @classmethod
    def calculate_probability(cls, data: dict, market_temperature: float = 1.0) -> float:
        """
        计算明日上涨概率。
        market_temperature: 市场温度系数（0.85~1.15），由沪深300涨跌+北向资金决定。
        筹码数据缺失时，权重自动重分配到其他四项。

        [优化 Phase 2.2] 集成动态权重调整：根据市场条件（牛/熊/震荡）动态调整各因子权重

        [优化 Phase 1] 删除第二层拉伸，直接使用原始分数，扩大分差
        """
        from src.adaptive_weights import AdaptiveWeights

        scores = cls.get_detailed_scores(data)
        chip_missing = data.get("chip_data_missing", False)

        # [Fix] 从 data 里取已注入的市场数据字段，传给 determine_condition
        market_meta = {
            "hs300_change_pct": data.get("hs300_change_pct", 0),
            "north_net_flow":   data.get("north_net_flow", 0),
        }
        market_condition = AdaptiveWeights.determine_condition(market_meta)
        w = AdaptiveWeights.get_weights(market_condition)

        if chip_missing:
            # 筹码权重重分配给其余四项
            wm = {"technical": 0.35, "fundamental": 0.23, "money_flow": 0.25, "sentiment": 0.17}
            raw = (
                scores["tech_score"]      * wm["technical"] +
                scores["fund_score"]      * wm["fundamental"] +
                scores["money_score"]     * wm["money_flow"] +
                scores["sentiment_score"] * wm["sentiment"]
            )
        else:
            raw = (
                scores["tech_score"]      * w.get("technical",   cls.WEIGHTS["technical"]) +
                scores["fund_score"]      * w.get("fundamental", cls.WEIGHTS["fundamental"]) +
                scores["money_score"]     * w.get("money_flow",  cls.WEIGHTS["money_flow"]) +
                scores["sentiment_score"] * w.get("sentiment",   cls.WEIGHTS["sentiment"]) +
                scores["chip_score"]      * w.get("chip",        cls.WEIGHTS["chip"])
            )

        # [优化] 删除第二层拉伸，直接用原始分数
        # 之前的两层拉伸导致所有股票都收敛到 40~65 分
        # stretched = 25 + (raw / 100) * 60
        # adjusted = 50 + (stretched - 50) * market_temperature

        # 新方案：直接应用市场温度，保留原始分差
        adjusted = 50 + (raw - 50) * market_temperature

        # 记录市场条件信息供后续分析使用
        data["market_condition"] = market_condition
        data["adaptive_weights"] = w

        return round(max(10.0, min(90.0, adjusted)), 1)

    @classmethod
    def get_detailed_scores(cls, data: dict) -> dict:
        return {
            "tech_score":      cls._calc_technical_score(data),
            "fund_score":      cls._calc_fundamental_score(data),
            "money_score":     cls._calc_money_flow_score(data),
            "sentiment_score": cls._calc_sentiment_score(data),
            "chip_score":      cls._calc_chip_score(data),
        }

    @classmethod
    def generate_reason(cls, data: dict, market_temp_desc: str = "") -> str:
        """
        生成2~3句简短的判断依据文字，用于排行榜展示。
        """
        parts = []
        signals = data.get("signals", [])
        money   = data.get("main_net_flow", 0)
        chip_missing = data.get("chip_data_missing", False)

        # 技术面
        bull_tech = [s for s in signals if s in {"MA_BULLISH", "MACD_GOLDEN", "KDJ_OVERSOLD", "BB_LOWER_BREAK"}]
        bear_tech = [s for s in signals if s in {"MA_BEARISH", "MACD_DEAD", "KDJ_OVERBOUGHT", "BB_UPPER_BREAK"}]
        sig_names = {
            "MA_BULLISH": "均线多头", "MACD_GOLDEN": "MACD金叉", "KDJ_OVERSOLD": "KDJ超卖反弹",
            "BB_LOWER_BREAK": "触及布林下轨", "MA_BEARISH": "均线空头", "MACD_DEAD": "MACD死叉",
            "KDJ_OVERBOUGHT": "KDJ超买", "BB_UPPER_BREAK": "突破布林上轨",
        }
        if bull_tech:
            parts.append("技术面 " + "、".join(sig_names.get(s, s) for s in bull_tech))
        elif bear_tech:
            parts.append("技术面 " + "、".join(sig_names.get(s, s) for s in bear_tech))

        # 资金面
        flow_ratio = money / (data.get("circulation_market_cap", 1) * 10000) * 100 if data.get("circulation_market_cap", 0) > 0 else 0
        if money > 0:
            parts.append(f"主力净流入{abs(money):.0f}万（占流通市值{flow_ratio:.1f}%）")
        elif money < 0:
            parts.append(f"主力净流出{abs(money):.0f}万（占流通市值{abs(flow_ratio):.1f}%）")

        # 筹码面
        if not chip_missing:
            chip_sigs = data.get("chip_signals", [])
            if "CHIP_TIGHT_LOW_PROFIT" in chip_sigs:
                parts.append(f"筹码高度集中（宽度{data.get('chip_width_70',0):.1f}%）且套牢盘多，上方压力小")
            elif "CHIP_CONVERGING" in chip_sigs:
                parts.append(f"筹码持续收敛（{data.get('chip_slope',0):+.3f}），筹码锁定性增强")
            elif "CHIP_WIDE_LOW_PROFIT" in chip_sigs:
                parts.append("大量套牢盘集中，有解套反弹动力")

        # 市场温度
        if market_temp_desc:
            parts.append(market_temp_desc)

        return "；".join(parts) if parts else "各维度信号中性，无明显驱动"

    @classmethod
    def _calc_technical_score(cls, data: dict) -> float:
        from src.indicators import SignalStrength
        from src.signals_confirmation import SignalConfirmation

        score = 50
        signals = data.get("signals", [])
        p = lambda k: cls._p("technical", k)

        # [优化 Phase 1.3] 使用信号强度加权

        # MA 多头
        if "MA_BULLISH" in signals:
            strength = SignalStrength.ma_bullish_strength(
                data.get("ma5", 0), data.get("ma10", 0),
                data.get("ma20", 0), data.get("ma60", 0)
            )
            score += int(20 * strength)

        # MACD 金叉
        if "MACD_GOLDEN" in signals:
            strength = SignalStrength.macd_strength(
                data.get("dif", 0), data.get("dea", 0),
                data.get("dif_pre"), data.get("dea_pre")
            )
            score += int(15 * strength)

        # KDJ 超卖
        if "KDJ_OVERSOLD" in signals:
            strength = SignalStrength.kdj_oversold_strength(
                data.get("kdj_k", 50), data.get("kdj_k_pre")
            )
            score += int(15 * strength)

        # 看空信号
        if "MA_BEARISH" in signals:
            strength = 1.0 - SignalStrength.ma_bullish_strength(
                data.get("ma5", 0), data.get("ma10", 0),
                data.get("ma20", 0), data.get("ma60", 0)
            )
            score -= int(20 * strength)

        if "MACD_DEAD" in signals:
            strength = SignalStrength.macd_strength(
                data.get("dif", 0), data.get("dea", 0),
                data.get("dif_pre"), data.get("dea_pre")
            )
            score -= int(15 * strength)

        if "KDJ_OVERBOUGHT" in signals:
            kdj_k = data.get("kdj_k", 50)
            strength = min(1.0, (kdj_k - 80) / 20) if kdj_k > 80 else 0
            score -= int(15 * strength)

        # 偏离 MA20 过远扣分
        dist = abs(data.get("distance_from_ma20", 0))
        if dist > 10:
            score -= p("ma20_far_penalty")
        elif dist > 5:
            score -= p("ma20_mid_penalty")

        # [优化 Phase 1] Momentum 因子（20日涨幅）
        momentum_20d = data.get("momentum_20d", 0)
        momentum_strength = SignalStrength.momentum_strength(momentum_20d)
        if momentum_20d > 10:
            score += int(15 * momentum_strength)
        elif momentum_20d < -10:
            score -= int(15 * momentum_strength)

        # [优化 Phase 2.1] 信号确认加成 - 识别"趋势+反转"强买信号
        if signals:
            confirmation = SignalConfirmation.analyze(signals)
            score += confirmation["confirmation_score"]

            # 记录确认信息到 data，供后续分析使用
            data["signal_confirmation"] = confirmation

        return max(0.0, min(100.0, score))

    @classmethod
    def _calc_fundamental_score(cls, data: dict) -> float:
        ind = data.get("indicators", {})
        score = 50
        p = lambda k: cls._p("fundamental", k)

        gm = ind.get("gross_margin", 0)
        if gm > 30:
            score += p("gm_high_score")
        elif gm > 20:
            score += p("gm_mid_score")
        elif gm < 5:
            score -= p("gm_low_penalty")

        roe = ind.get("roe", 0)
        if roe > 15:
            score += p("roe_high_score")
        elif roe > 10:
            score += p("roe_mid_score")
        elif roe < 0:
            score -= p("roe_neg_penalty")

        if ind.get("revenue_growth", 0) > 20:
            score += p("rev_growth_score")
        if ind.get("profit_growth", 0) > 20:
            score += p("profit_growth_score")

        # [优化 Phase 1.2] 改进 PE 处理 - 用行业平均值替代缺失数据
        INDUSTRY_PE = {
            "食品饮料": 25, "医药生物": 35, "电子": 22, "计算机": 28,
            "电气设备": 18, "电力": 12, "房地产": 8, "银行": 8,
            "非银金融": 15, "化工": 14, "机械设备": 16, "汽车": 10,
            "纺织": 12, "造纸": 10, "建筑": 8, "钢铁": 8, "煤炭": 6,
        }

        pe = data.get("pe")
        if pe is None or pe == 0:
            # 用行业平均 PE 代替
            sector = data.get("sector", "unknown")
            industry_pe = INDUSTRY_PE.get(sector, 15)
            # 仅在异常情况下才扣分
            if industry_pe and 10 < industry_pe < 30:
                pass  # 中等 PE，不扣分也不加分
            else:
                score -= p("pe_missing_penalty")
        elif 0 < pe < 20:
            score += p("pe_good_score")
        elif pe > 100 or pe < 0:
            score -= p("pe_bad_penalty")

        # [优化 Phase 1.2] 改进 PB 处理
        pb = data.get("pb")
        if pb is None or pb == 0:
            # 不扣分，中性处理
            pass
        elif 0 < pb < 1.5:
            score += p("pb_good_score")
        elif pb > 3:
            score -= int(p("pb_good_score") * 0.5)  # 减少扣分幅度

        return max(0.0, min(100.0, score))

    @classmethod
    def _calc_money_flow_score(cls, data: dict) -> float:
        score = 50
        main_net   = data.get("main_net_flow", 0)
        circ_cap   = data.get("circulation_market_cap", 0)
        p = lambda k: cls._p("money_flow", k)

        if circ_cap > 0:
            flow_ratio = main_net / (circ_cap * 10000) * 100
        else:
            flow_ratio = 0

        if flow_ratio > 5:
            score += p("flow_very_high_score")
        elif flow_ratio > 3:
            score += p("flow_high_score")
        elif flow_ratio > 1:
            score += p("flow_mid_score")
        elif flow_ratio < -3:
            score -= p("flow_high_penalty")
        elif flow_ratio < -1:
            score -= p("flow_mid_penalty")

        return max(0.0, min(100.0, score))

    @classmethod
    def _calc_sentiment_score(cls, data: dict) -> float:
        score = 50
        p = lambda k: cls._p("sentiment", k)
        change_pct = data.get("change_pct", 0)
        rising = change_pct > 0

        # 换手率需联合涨跌方向判断：
        # 高换手+上涨 = 买气旺盛（+），高换手+下跌 = 加速出逃（-），低换手 = 缩量
        turnover = data.get("turnover_rate", 0)
        if turnover > 10:
            score += p("turnover_high_score") if rising else -p("turnover_high_score")
        elif turnover > 5:
            score += p("turnover_mid_score") if rising else -p("turnover_mid_score")
        elif turnover < 1:
            score -= p("turnover_low_penalty")

        # 量比同理：放量上涨是共识买入，放量下跌是恐慌抛售
        vol_ratio = data.get("vol_ratio", 1)
        if vol_ratio > 2:
            score += p("vol_high_score") if rising else -p("vol_high_score")
        elif vol_ratio > 1.5:
            score += p("vol_mid_score") if rising else -p("vol_mid_score")
        elif vol_ratio < 0.5:
            score -= p("vol_low_penalty")

        # 涨跌幅绝对值
        abs_chg = abs(change_pct)
        if abs_chg > 5:
            score += p("change_high_score") if rising else -p("change_high_score")
        elif abs_chg > 3:
            score += p("change_mid_score") if rising else -p("change_mid_score")

        return max(0.0, min(100.0, score))

    @classmethod
    def _calc_chip_score(cls, data: dict) -> float:
        """
        筹码集中度评分（满分100，基准50）

        加分规则（均为看多信号）：
          CHIP_CONVERGING：近15天筹码持续收敛
          CHIP_TIGHT_LOW_PROFIT：极紧集中 + 低获利
          CHIP_WIDE_LOW_PROFIT：大范围低获利（套牢盘多，有解套反弹动力）
        """
        score = 50
        chip_signals = data.get("chip_signals", [])
        p = lambda k: cls._p("chip", k)

        if "CHIP_CONVERGING" in chip_signals:
            score += p("converging_score")
        if "CHIP_TIGHT_LOW_PROFIT" in chip_signals:
            score += p("tight_low_profit_score")
        if "CHIP_WIDE_LOW_PROFIT" in chip_signals:
            score += p("wide_low_profit_score")

        profit = data.get("chip_profit_ratio", 50)
        if chip_signals and profit < 10:
            score += p("low_profit_bonus")

        width = data.get("chip_width_70", 50)
        if width > 0 and width < 5:
            score += p("narrow_width_bonus")

        return max(0.0, min(100.0, score))
