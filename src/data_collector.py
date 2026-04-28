"""
数据采集模块 - 复用 dian_monitor 已验证的 akshare 数据获取逻辑
[优化 Phase 1] 增加 baostock 筹码数据集成和 K线模拟
"""
import sqlite3
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"


def _sina_symbol(code: str) -> str:
    prefix = "sz" if code.startswith(("0", "3")) else "sh"
    return f"{prefix}{code}"


def _with_retry(fn, retries=3, delay=5):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def _get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_cache (
            code TEXT, trade_date TEXT, data_json TEXT,
            PRIMARY KEY (code, trade_date)
        )
    """)
    conn.commit()
    return conn


class StockDataCollector:

    def get_stock_data(self, code: str, date: str) -> dict:
        """获取股票行情 + 财务 + 估值，优先读取当日缓存"""
        # 0) 读取本地缓存（同日已采集则直接复用，避免重复网络请求）
        conn = _get_db()
        cached = self._read_cache(conn, code, date)
        conn.close()
        if cached and cached.get("hist") is not None:
            return cached

        # 1) 尝试实时行情（盘中有效）
        realtime = self._fetch_realtime(code)
        if realtime:
            hist = self._ensure_hist(code)
            if hist is not None and not hist.empty:
                realtime["hist"] = hist
                realtime["chip_data_missing"] = True  # 筹码由 ChipAnalyzer 单独拉取
                return realtime

        # 2) 强制从 akshare 拉取最新日线
        data = self._fetch_from_akshare(code, date)
        data["chip_data_missing"] = True  # 筹码由 ChipAnalyzer 单独拉取

        # 3) 写入本地缓存
        conn = _get_db()
        self._save_cache(conn, code, date, data)
        conn.close()
        return data

    # ── 实时行情 ──────────────────────────────────────────

    def _fetch_realtime(self, code: str) -> Optional[dict]:
        """东方财富实时/盘中快照，返回与日线同构的数据 dict。
        失败返回 None，调用方自行回落。
        """
        import akshare as ak
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return None
            row = df[df["代码"] == code]
            if row.empty:
                return None
            r = row.iloc[0]

            price      = float(r.get("最新价") or 0)
            prev_close = float(r.get("昨收") or 0)
            if price <= 0 or prev_close <= 0:
                return None

            change_pct = (price - prev_close) / prev_close * 100
            float_shares = float(r.get("流通市值") or 0) / price if price else 0

            return {
                "code":               code,
                "name":               str(r.get("名称", code)),
                "trade_date":         str(r.get("时间", datetime.now().strftime("%Y-%m-%d")))[:10],
                "price":              price,
                "change_pct":         round(change_pct, 2),
                "change_amt":         round(price - prev_close, 4),
                "open":               float(r.get("今开") or 0),
                "high":               float(r.get("最高") or price),
                "low":                float(r.get("最低") or price),
                "volume":             float(r.get("成交量") or 0),
                "turnover":           float(r.get("成交额") or 0),
                "turnover_rate":      float(r.get("换手率") or 0),
                "amplitude":          float(r.get("振幅") or 0),
                "vol_avg20":          0,  # 实时接口无此字段，由 hist 计算
                "float_mv":           float(r.get("流通市值") or 0),
                "circulation_market_cap": float(r.get("流通市值") or 0) / 1e8,
                "pe":                 float(r.get("市盈率-动态") or 0),
                "pb":                 0,
                "main_net_flow":      0,
                # 以下字段 hist 补充后再写入
                "hist":               None,
            }
        except Exception:
            return None

    def _ensure_hist(self, code: str) -> Optional[pd.DataFrame]:
        """拉取近 60 交易日日线，供指标计算使用"""
        import akshare as ak
        try:
            hist_raw = ak.stock_zh_a_daily(symbol=_sina_symbol(code), adjust="qfq")
            if hist_raw is None or hist_raw.empty:
                return None
            return hist_raw.rename(columns={
                "date": "日期", "open": "开盘", "high": "最高",
                "low": "最低", "close": "收盘", "volume": "成交量",
                "outstanding_share": "流通股", "turnover": "换手率",
            }).tail(60).reset_index(drop=True)
        except Exception:
            return None

    def _fetch_from_akshare(self, code: str, date: str) -> dict:
        import akshare as ak

        hist_raw = _with_retry(lambda: ak.stock_zh_a_daily(
            symbol=_sina_symbol(code), adjust="qfq"
        ))
        if hist_raw is None or hist_raw.empty:
            raise ValueError(f"{code} 行情为空")

        hist = hist_raw.rename(columns={
            "date": "日期", "open": "开盘", "high": "最高",
            "low": "最低", "close": "收盘", "volume": "成交量",
            "outstanding_share": "流通股", "turnover": "换手率",
        }).tail(60).reset_index(drop=True)

        latest  = hist.iloc[-1]
        prev    = hist.iloc[-2] if len(hist) >= 2 else latest
        vol_avg20 = hist["成交量"].iloc[-21:-1].mean() if len(hist) >= 22 else hist["成交量"].mean()

        price      = float(latest["收盘"])
        prev_close = float(prev["收盘"])
        float_shares = float(latest.get("流通股") or 0)

        data = {
            "code":         code,
            "name":         code,
            "trade_date":   str(latest["日期"])[:10],
            "price":        price,
            "change_pct":   (price - prev_close) / prev_close * 100 if prev_close else 0,
            "change_amt":   price - prev_close,
            "open":         float(latest["开盘"]),
            "high":         float(latest["最高"]),
            "low":          float(latest["最低"]),
            "volume":       float(latest["成交量"]),
            "turnover":     float(latest.get("amount") or 0),
            "turnover_rate": float(latest.get("换手率") or 0) * 100,
            "amplitude":    (float(latest["最高"]) - float(latest["最低"])) / prev_close * 100 if prev_close else 0,
            "vol_avg20":    float(vol_avg20) if vol_avg20 else 0,
            "float_mv":     price * float_shares,
            "circulation_market_cap": price * float_shares / 1e8,
            "hist":         hist,
            "pe": None, "pb": None,
            "main_net_flow": 0,
        }

        # PE/PB（通过个股信息 + 财务数据计算）
        try:
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and not info.empty:
                info_map = dict(zip(info["item"], info["value"]))
                total_mv = float(info_map.get("总市值") or 0)

                # 从利润表获取净利润并年化
                try:
                    df_inc = ak.stock_financial_report_sina(stock=code, symbol="利润表")
                    if df_inc is not None and not df_inc.empty:
                        net_income = float(df_inc.iloc[0].get("归属于母公司所有者的净利润") or 0)
                        # 判断报告期（Q1/Q2/Q3/年报）
                        report_date = str(df_inc.iloc[0].get("报告日") or "")
                        if report_date and len(report_date) >= 6:
                            month = int(report_date[4:6])
                            # 年化：Q1(3月)×4, Q2(6月)×2, Q3(9月)×4/3, 年报(12月)×1
                            if month <= 3:
                                annual_net = net_income * 4
                            elif month <= 6:
                                annual_net = net_income * 2
                            elif month <= 9:
                                annual_net = net_income * 4 / 3
                            else:
                                annual_net = net_income
                        else:
                            annual_net = net_income * 4  # 默认年化

                        if annual_net > 0:
                            data["pe"] = round(total_mv / annual_net, 2)
                except Exception:
                    pass

                # PB = 总市值 / 归属于母公司股东权益
                try:
                    df_bs = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
                    if df_bs is not None and not df_bs.empty:
                        equity = float(df_bs.iloc[0].get("归属于母公司股东权益合计") or 0)
                        if equity > 0:
                            data["pb"] = round(total_mv / equity, 2)
                except Exception:
                    pass
        except Exception:
            pass

        # 资金流向 — 取最近 3 日均值（单日数据噪音大，容易被对敲伪造）
        try:
            market = "sz" if code.startswith(("0", "3")) else "sh"
            df_flow = ak.stock_individual_fund_flow(stock=code, market=market)
            if df_flow is not None and not df_flow.empty:
                recent = df_flow.tail(3)
                avg_flow = recent["主力净流入-净额"].astype(float).mean()
                data["main_net_flow"] = round(avg_flow / 1e4, 2)  # 元→万元
        except Exception:
            pass

        return data

    def _read_cache(self, conn, code: str, date: str) -> Optional[dict]:
        import json
        row = conn.execute(
            "SELECT data_json FROM daily_cache WHERE code=? AND trade_date=?", (code, date)
        ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row[0])
            if "hist" in data and isinstance(data["hist"], list):
                data["hist"] = pd.DataFrame(data["hist"])
                # 还原数值列类型
                for col in ["开盘","最高","最低","收盘","成交量"]:
                    if col in data["hist"].columns:
                        data["hist"][col] = pd.to_numeric(data["hist"][col], errors="coerce")
            return data
        except Exception:
            return None

    def _save_cache(self, conn, code: str, date: str, data: dict):
        import json
        save = {k: v for k, v in data.items() if k != "hist"}
        if data.get("hist") is not None and not data["hist"].empty:
            save["hist"] = data["hist"].to_dict(orient="records")
        conn.execute(
            "INSERT OR REPLACE INTO daily_cache VALUES (?,?,?)",
            (code, date, json.dumps(save, default=str))
        )
        conn.commit()

    def get_financial_data(self, code: str) -> dict:
        """获取财务数据（利润表 + 资产负债表最新一期）"""
        import akshare as ak
        result = {"gross_margin": 0, "net_margin": 0, "revenue_growth": 0, "profit_growth": 0, "roe": 0}
        try:
            df_inc = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            if df_inc is not None and not df_inc.empty:
                r = df_inc.iloc[0]
                rev  = float(r.get("营业总收入") or 0)
                cost = float(r.get("营业成本") or 0)
                net  = float(r.get("归属于母公司所有者的净利润") or 0)
                rev_prev = float(df_inc.iloc[1].get("营业总收入") or 0) if len(df_inc) > 1 else 0
                net_prev = float(df_inc.iloc[1].get("归属于母公司所有者的净利润") or 0) if len(df_inc) > 1 else 0

                result.update({
                    "gross_margin":   (rev - cost) / rev * 100 if rev else 0,
                    "net_margin":     net / rev * 100 if rev else 0,
                    "revenue_growth": (rev - rev_prev) / abs(rev_prev) * 100 if rev_prev else 0,
                    "profit_growth":  (net - net_prev) / abs(net_prev) * 100 if net_prev else 0,
                })

                # ROE = 年化净利润 / 归属母公司股东权益（来自资产负债表）
                try:
                    df_bs = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
                    if df_bs is not None and not df_bs.empty:
                        equity = float(df_bs.iloc[0].get("归属于母公司股东权益合计") or 0)
                        if equity > 0 and net != 0:
                            report_date = str(r.get("报告日") or "")
                            month = int(report_date[4:6]) if len(report_date) >= 6 else 12
                            multiplier = {3: 4, 6: 2, 9: 4/3}.get(month, 1)
                            annual_net = net * multiplier
                            result["roe"] = round(annual_net / equity * 100, 2)
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def get_market_temperature(self) -> dict:
        """
        获取市场温度：沪深300当日涨跌幅 + 北向资金净流入。
        返回 temperature 系数（0.85~1.15），用于调整个股概率。
        失败时返回中性值 1.0。
        """
        import akshare as ak
        result = {"hs300_change": 0.0, "north_flow": 0.0, "temperature": 1.0, "temp_desc": "市场中性"}
        try:
            # 沪深300当日涨跌
            df = ak.stock_zh_index_spot_em(symbol="沪深京A股")
            if df is not None and not df.empty:
                hs300 = df[df["代码"] == "000300"]
                if not hs300.empty:
                    result["hs300_change"] = float(hs300.iloc[0].get("涨跌幅", 0))
        except Exception:
            pass

        try:
            # 北向资金今日净流入（亿元）
            df_north = ak.stock_connect_position_statistics_em(symbol="沪深港通资金流向")
            if df_north is not None and not df_north.empty:
                row = df_north.iloc[0]
                result["north_flow"] = float(row.get("今日净买入") or 0)
        except Exception:
            pass

        # 计算温度系数：hs300涨跌 + 北向资金双维度
        hs = result["hs300_change"]
        nf = result["north_flow"]

        score = 0
        if hs > 2:    score += 2
        elif hs > 0.5: score += 1
        elif hs < -2:  score -= 2
        elif hs < -0.5: score -= 1

        if nf > 50:   score += 1
        elif nf < -50: score -= 1

        # 映射到系数
        coeff_map = {3: 1.15, 2: 1.10, 1: 1.05, 0: 1.0, -1: 0.95, -2: 0.90, -3: 0.85}
        temp = coeff_map.get(max(-3, min(3, score)), 1.0)
        result["temperature"] = temp

        if temp >= 1.10:   result["temp_desc"] = f"市场强势(沪深300 {hs:+.2f}%)"
        elif temp >= 1.05: result["temp_desc"] = f"市场偏强(沪深300 {hs:+.2f}%)"
        elif temp <= 0.90: result["temp_desc"] = f"市场弱势(沪深300 {hs:+.2f}%)"
        elif temp <= 0.95: result["temp_desc"] = f"市场偏弱(沪深300 {hs:+.2f}%)"
        else:              result["temp_desc"] = f"市场中性(沪深300 {hs:+.2f}%)"

        return result

    @staticmethod
    def search_stock(keyword: str) -> list:
        """
        股票搜索：支持代码精准匹配 或 名称模糊搜索。
        返回 [{"code": "300244", "name": "迪安诊断"}, ...]，最多20条。
        """
        import akshare as ak
        try:
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return []
            keyword = keyword.strip()
            # 精准代码匹配
            exact = df[df["code"] == keyword]
            if not exact.empty:
                return [{"code": r["code"], "name": str(r["name"]).replace(" ", "")} for _, r in exact.iterrows()]
            # 名称模糊搜索
            fuzzy = df[df["name"].str.contains(keyword, na=False)]
            return [{"code": r["code"], "name": str(r["name"]).replace(" ", "")} for _, r in fuzzy.head(20).iterrows()]
        except Exception:
            return []
