"""
历史行情增量下载器

存储位置：~/Desktop/quant_trading/data/hist_daily.db
表结构：hist_daily(code, date, open, high, low, close, volume, amount, turnover_rate)
PRIMARY KEY (code, date) — 幂等，重复写入安全。

支持三年日线数据，供回测因子分析使用。
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "hist_daily.db"

# ── sina 代码格式工具 ───────────────────────────────────────────
def _sina_symbol(code: str) -> str:
    if code.startswith(("6",)):
        return "sh" + code
    return "sz" + code


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hist_daily (
            code          TEXT    NOT NULL,
            date          TEXT    NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            volume        REAL,
            amount        REAL,
            turnover_rate REAL,
            PRIMARY KEY (code, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_code_date ON hist_daily(code, date)")
    conn.commit()


class HistoryDownloader:
    """
    增量下载股票日线历史数据。

    progress_cb(code, msg): 可选进度回调，msg 为字符串描述。
    """

    def __init__(self, progress_cb: Optional[Callable] = None):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._progress_cb = progress_cb or (lambda code, msg: None)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DB_PATH))
        _init_db(conn)
        return conn

    def _emit(self, code: str, msg: str):
        self._progress_cb(code, msg)

    def existing_dates(self, conn: sqlite3.Connection, code: str) -> set:
        rows = conn.execute(
            "SELECT date FROM hist_daily WHERE code=?", (code,)
        ).fetchall()
        return {r[0] for r in rows}

    def download_stock(self, code: str, years: int = 3) -> dict:
        """
        下载指定股票近 years 年日线数据，仅写入本地库中缺失的日期。
        返回 {"code", "inserted", "skipped", "total_in_db"}
        """
        import akshare as ak

        self._emit(code, "拉取行情数据…")
        try:
            raw = ak.stock_zh_a_daily(symbol=_sina_symbol(code), adjust="qfq")
        except Exception as e:
            self._emit(code, f"❌ 拉取失败: {e}")
            return {"code": code, "error": str(e)}

        if raw is None or raw.empty:
            self._emit(code, "❌ 返回数据为空")
            return {"code": code, "error": "数据为空"}

        # 过滤近 years 年
        cutoff = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        raw["date"] = raw["date"].astype(str)
        raw = raw[raw["date"] >= cutoff].copy()

        # 列名映射（akshare sina 接口标准列）
        col_map = {
            "date":              "date",
            "open":              "open",
            "high":              "high",
            "low":               "low",
            "close":             "close",
            "volume":            "volume",
            "outstanding_share": "amount",    # 部分接口此列实为流通股，amount 设为 0
            "turnover":          "turnover_rate",
        }
        df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
        # 如果接口没有 amount / turnover_rate 列，补 0
        for col in ("amount", "turnover_rate"):
            if col not in df.columns:
                df[col] = 0.0

        conn = self._conn()
        existing = self.existing_dates(conn, code)
        to_insert = df[~df["date"].isin(existing)]

        inserted = 0
        if not to_insert.empty:
            self._emit(code, f"写入 {len(to_insert)} 条新数据…")
            rows = []
            for _, row in to_insert.iterrows():
                rows.append((
                    code,
                    str(row["date"]),
                    float(row.get("open", 0) or 0),
                    float(row.get("high", 0) or 0),
                    float(row.get("low", 0) or 0),
                    float(row.get("close", 0) or 0),
                    float(row.get("volume", 0) or 0),
                    float(row.get("amount", 0) or 0),
                    float(row.get("turnover_rate", 0) or 0),
                ))
            conn.executemany(
                "INSERT OR IGNORE INTO hist_daily VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
            conn.commit()
            inserted = len(rows)
        else:
            self._emit(code, "本地数据已是最新，无需更新")

        total = conn.execute(
            "SELECT COUNT(*) FROM hist_daily WHERE code=?", (code,)
        ).fetchone()[0]
        conn.close()

        skipped = len(df) - inserted
        self._emit(code, f"✅ 完成：新增 {inserted} 条，已跳过 {skipped} 条，库中共 {total} 条")
        return {"code": code, "inserted": inserted, "skipped": skipped, "total_in_db": total}

    def download_all(self, stocks: list, years: int = 3) -> list:
        """下载 watchlist 全部股票，返回每只的结果列表"""
        results = []
        for i, (code, name) in enumerate(stocks):
            self._emit(code, f"[{i+1}/{len(stocks)}] {name}({code})")
            r = self.download_stock(code, years)
            r["name"] = name
            results.append(r)
        return results

    @staticmethod
    def load_hist(code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从本地 hist_daily 读取指定股票的日线 DataFrame。
        列名与 data_collector 保持一致（中文列名）供 TechnicalIndicators 使用。
        """
        if not DB_PATH.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(str(DB_PATH))
        query = "SELECT date,open,high,low,close,volume,amount,turnover_rate FROM hist_daily WHERE code=?"
        params: list = [code]
        if start_date:
            query += " AND date >= ?"; params.append(start_date)
        if end_date:
            query += " AND date <= ?"; params.append(end_date)
        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        df = df.rename(columns={
            "date":          "日期",
            "open":          "开盘",
            "high":          "最高",
            "low":           "最低",
            "close":         "收盘",
            "volume":        "成交量",
            "amount":        "成交额",
            "turnover_rate": "换手率",
        })
        df["日期"] = pd.to_datetime(df["日期"])
        for col in ["开盘","最高","最低","收盘","成交量","成交额","换手率"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df.reset_index(drop=True)

    @staticmethod
    def get_stock_summary() -> list:
        """返回各股票在库中的数据统计，供前端展示"""
        if not DB_PATH.exists():
            return []
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("""
            SELECT code, COUNT(*) as cnt,
                   MIN(date) as min_date, MAX(date) as max_date
            FROM hist_daily GROUP BY code
        """).fetchall()
        conn.close()
        return [{"code": r[0], "count": r[1], "from": r[2], "to": r[3]} for r in rows]
