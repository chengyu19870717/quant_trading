#!/usr/bin/env python3
"""
数据增强：下载沪深300指数 + 个股主力资金流 + 扩充历史K线
用法：python data_enrichment.py
"""
import json
import sqlite3
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import akshare as ak

HIST_DB   = Path(__file__).parent / "data/hist_daily.db"
WATCHLIST = Path(__file__).parent / "config/watchlist.json"

MARKET_MAP = {"60": "sh", "00": "sz", "30": "sz", "68": "sh"}


def get_market(code: str) -> str:
    return MARKET_MAP.get(code[:2], "sz")


def init_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_daily (
            code TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL, PRIMARY KEY (code, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS money_flow (
            code TEXT, date TEXT,
            main_net REAL, main_pct REAL,
            big_net REAL, big_pct REAL,
            mid_net REAL, mid_pct REAL,
            small_net REAL, small_pct REAL,
            PRIMARY KEY (code, date)
        )
    """)
    conn.commit()


def download_index(conn, code="sh000300", name="沪深300"):
    print(f"  下载指数 {name}({code})…")
    df = ak.stock_zh_index_daily(symbol=code)
    df = df.rename(columns={"date":"date","open":"open","high":"high",
                             "low":"low","close":"close","volume":"volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = code
    df = df[["code","date","open","high","low","close","volume"]]
    conn.executemany(
        "INSERT OR REPLACE INTO index_daily VALUES (?,?,?,?,?,?,?)",
        df.itertuples(index=False, name=None)
    )
    conn.commit()
    print(f"    ✅ {len(df)} 条，{df['date'].min()} ~ {df['date'].max()}")


def download_money_flow(conn, code: str, market: str):
    try:
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        df = df.rename(columns={
            "日期": "date",
            "主力净流入-净额": "main_net", "主力净流入-净占比": "main_pct",
            "超大单净流入-净额": "big_net",  "超大单净流入-净占比": "big_pct",
            "中单净流入-净额": "mid_net",   "中单净流入-净占比": "mid_pct",
            "小单净流入-净额": "small_net", "小单净流入-净占比": "small_pct",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        cols = ["code","date","main_net","main_pct","big_net","big_pct",
                "mid_net","mid_pct","small_net","small_pct"]
        df = df[cols].dropna(subset=["main_net"])
        conn.executemany(
            "INSERT OR REPLACE INTO money_flow VALUES (?,?,?,?,?,?,?,?,?,?)",
            df.itertuples(index=False, name=None)
        )
        conn.commit()
        return len(df)
    except Exception as e:
        print(f"    ⚠️  资金流下载失败: {e}")
        return 0


def extend_kline(conn, code: str, market: str):
    """尝试用 akshare 拉更长的历史 K 线（后复权）"""
    try:
        mkt = "1" if market == "sh" else "0"
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                 start_date="20220101", end_date="20261231",
                                 adjust="hfq")
        df = df.rename(columns={
            "日期":"date","开盘":"open","最高":"high","最低":"low",
            "收盘":"close","成交量":"volume","成交额":"amount"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        df = df[["code","date","open","high","low","close","volume"]].dropna()
        # 只插入新的（不覆盖已有数据）
        existing = set(r[0] for r in conn.execute(
            "SELECT date FROM hist_daily WHERE code=?", (code,)).fetchall())
        new_rows = df[~df["date"].isin(existing)]
        if not new_rows.empty:
            conn.executemany(
                "INSERT OR IGNORE INTO hist_daily VALUES (?,?,?,?,?,?,?)",
                new_rows[["code","date","open","high","low","close","volume"]
                         ].itertuples(index=False, name=None)
            )
            conn.commit()
            return len(new_rows)
        return 0
    except Exception as e:
        print(f"    ⚠️  K线扩充失败: {e}")
        return 0


def main():
    stocks = json.loads(WATCHLIST.read_text())["stocks"]
    conn   = sqlite3.connect(HIST_DB)
    init_tables(conn)

    # 1. 下载沪深300指数
    print("\n【1/3】下载指数数据")
    download_index(conn, "sh000300", "沪深300")
    download_index(conn, "sh000001", "上证指数")
    time.sleep(0.5)

    # 2. 下载资金流
    print("\n【2/3】下载主力资金流")
    for code, name in stocks:
        mkt = get_market(code)
        n = download_money_flow(conn, code, mkt)
        print(f"  {name}({code}): {n} 条")
        time.sleep(0.3)

    # 3. 扩充 K 线历史
    print("\n【3/3】扩充历史 K 线（尝试补充2022年起数据）")
    for code, name in stocks:
        mkt = get_market(code)
        before = conn.execute(
            "SELECT COUNT(*) FROM hist_daily WHERE code=?", (code,)
        ).fetchone()[0]
        n = extend_kline(conn, code, mkt)
        after = conn.execute(
            "SELECT COUNT(*) FROM hist_daily WHERE code=?", (code,)
        ).fetchone()[0]
        date_range = conn.execute(
            "SELECT MIN(date),MAX(date) FROM hist_daily WHERE code=?", (code,)
        ).fetchone()
        print(f"  {name}({code}): {before}→{after}条  {date_range[0]}~{date_range[1]}")
        time.sleep(0.5)

    conn.close()
    print("\n✅ 数据增强完成")


if __name__ == "__main__":
    main()
