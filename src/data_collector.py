"""
数据采集模块 - 复用 dian_monitor 已验证的 akshare 数据获取逻辑
"""
import sqlite3
import warnings
import time
from datetime import datetime
from pathlib import Path

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
        """获取股票行情 + 财务 + 估值，带本地缓存"""
        conn = _get_db()
        row = conn.execute(
            "SELECT data_json FROM daily_cache WHERE code=? AND trade_date=?",
            (code, date)
        ).fetchone()

        if row:
            import json
            data = json.loads(row[0])
            # hist 存的是 records 格式，恢复为 DataFrame
            data["hist"] = pd.DataFrame(data["hist"])
            conn.close()
            return data

        data = self._fetch_from_akshare(code, date)
        self._save_cache(conn, code, date, data)
        conn.close()
        return data

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

        # PE/PB（失败静默跳过）
        try:
            df_val = ak.stock_a_lg_indicator(symbol=code)
            if df_val is not None and not df_val.empty:
                r = df_val.iloc[-1]
                data["pe"] = float(r.get("pe") or 0)
                data["pb"] = float(r.get("pb") or 0)
        except Exception:
            pass

        # 资金流向（失败静默跳过）
        try:
            df_flow = ak.stock_individual_fund_flow(stock=code, market="sz" if code.startswith(("0","3")) else "sh")
            if df_flow is not None and not df_flow.empty:
                latest_flow = df_flow.iloc[-1]
                data["main_net_flow"] = float(latest_flow.get("主力净流入-净额") or 0) / 1e4  # 转万元
        except Exception:
            pass

        return data

    def _save_cache(self, conn, code: str, date: str, data: dict):
        import json
        save = {k: v for k, v in data.items() if k != "hist"}
        save["hist"] = data["hist"].to_dict(orient="records")
        conn.execute(
            "INSERT OR REPLACE INTO daily_cache VALUES (?,?,?)",
            (code, date, json.dumps(save, default=str))
        )
        conn.commit()

    def get_financial_data(self, code: str) -> dict:
        """获取财务数据（利润表最新一期）"""
        import akshare as ak
        try:
            df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            if df is not None and not df.empty:
                r = df.iloc[0]
                rev  = float(r.get("营业总收入") or 0)
                cost = float(r.get("营业成本") or 0)
                net  = float(r.get("归属于母公司所有者的净利润") or 0)

                rev_prev = float(df.iloc[1].get("营业总收入") or 0) if len(df) > 1 else 0
                net_prev = float(df.iloc[1].get("归属于母公司所有者的净利润") or 0) if len(df) > 1 else 0

                return {
                    "gross_margin":    (rev - cost) / rev * 100 if rev else 0,
                    "net_margin":      net / rev * 100 if rev else 0,
                    "revenue_growth":  (rev - rev_prev) / abs(rev_prev) * 100 if rev_prev else 0,
                    "profit_growth":   (net - net_prev) / abs(net_prev) * 100 if net_prev else 0,
                    "roe":             0,
                }
        except Exception:
            pass
        return {"gross_margin": 0, "net_margin": 0, "revenue_growth": 0, "profit_growth": 0, "roe": 0}
