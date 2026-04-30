#!/usr/bin/env python3
"""
通用专属模型优化脚本
用法：
    python optimize_custom_model.py 603881
    python optimize_custom_model.py all          # 对所有监控股票跑优化
目标：IC>0.1, ICIR>0.6, 准确率>60%, 上涨胜率>60%
优化完成后自动注册到 investment_hub DB 并导出 custom_models.json
"""
import sys
import json
import time
import sqlite3
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest_web import (
    calc_base_indicators, add_rolling_zscore, add_cross_features,
    score_custom, calc_custom_metrics
)

INVEST_DB    = Path.home() / ".baibao" / "baibao.db"
HIST_DB      = Path(__file__).parent / "data/hist_daily.db"
WATCHLIST    = Path(__file__).parent / "config/watchlist.json"
CUSTOM_JSON  = Path(__file__).parent / "config/custom_models.json"

ALL_FEATURES = [
    "body_z","hl_range_z","candle_q_z","gap_z","macd_hist_z","kdj_j_z",
    "bb_pos_z","rsi_z","vol_ratio_z","vol_accel_z",
    "body_vol_z","body_macd_z","gap_vol_z","cq_vol_z","macd_vol_z","bb_rsi_z",
]
N_FEAT = len(ALL_FEATURES)

TARGET_IC   = 0.10
TARGET_ICIR = 0.60
TARGET_ACC  = 60.0
TARGET_UPWR = 60.0


# ── 数据加载 ────────────────────────────────────────────────
def fetch_kline(code: str) -> pd.DataFrame:
    conn = sqlite3.connect(HIST_DB)
    df = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume FROM hist_daily WHERE code=? ORDER BY date",
        conn, params=(code,)
    )
    conn.close()
    return df


def build_feature_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = calc_base_indicators(df_raw)
    base = ["body","hl_range","candle_q","gap","macd_hist","kdj_j",
            "bb_pos","rsi","vol_ratio","vol_accel","bb_squeeze",
            "ret1","ret3","ret5","ma5_slope","dist_ma5","dist_ma20",
            "vol_slope","pos_in_range"]
    df = add_rolling_zscore(df, base)
    df = add_cross_features(df)
    df["next_ret"] = df["close"].shift(-1) / df["close"] * 100 - 100
    df["next_up"]  = df["close"].shift(-1) > df["close"]
    return df.dropna(subset=ALL_FEATURES + ["next_ret","next_up"]).reset_index(drop=True)


# ── 评分核心 ────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, weights: np.ndarray, threshold: float = 0.25) -> dict:
    """给定权重，计算 IC/ICIR/Acc/UpWR，用于优化目标"""
    feat_mat = df[ALL_FEATURES].values
    scores   = feat_mat @ weights
    rets     = df["next_ret"].values
    actual_b = df["next_up"].values

    warmup = min(30, max(20, len(df) // 4))
    scores  = scores[warmup:]
    rets    = rets[warmup:]
    actual_b = actual_b[warmup:]

    if len(scores) < 30:
        return {"ic": 0, "icir": 0, "accuracy": 0, "up_win_rate": 0}

    ic = float(np.corrcoef(scores, rets)[0, 1]) if np.std(scores) > 0 else 0
    if np.isnan(ic): ic = 0

    win, step = 60, 10
    rolling_ics = []
    for i in range(0, len(scores) - win, step):
        p_w, r_w = scores[i:i+win], rets[i:i+win]
        if np.std(p_w) > 0:
            c = float(np.corrcoef(p_w, r_w)[0, 1])
            if not np.isnan(c): rolling_ics.append(c)
    ics_arr = np.array(rolling_ics)
    icir = float(ics_arr.mean() / ics_arr.std()) if len(ics_arr) > 1 and ics_arr.std() > 0 else 0

    thr_val  = np.percentile(scores, (1 - threshold) * 100)
    pred_up  = scores >= thr_val
    acc      = float((pred_up == actual_b).mean() * 100)
    up_wr    = float(actual_b[pred_up].mean() * 100) if pred_up.sum() > 0 else 0

    return {"ic": round(ic, 4), "icir": round(icir, 4),
            "accuracy": round(acc, 2), "up_win_rate": round(up_wr, 2)}


def score_fn(m: dict) -> float:
    """综合得分：四个指标的归一化加权和（越高越好）"""
    ic_s   = min(m["ic"]   / TARGET_IC,   2.0)
    icir_s = min(m["icir"] / TARGET_ICIR, 2.0)
    acc_s  = min(m["accuracy"]   / TARGET_ACC,  2.0)
    upwr_s = min(m["up_win_rate"]/ TARGET_UPWR, 2.0)
    return ic_s * 0.3 + icir_s * 0.3 + acc_s * 0.2 + upwr_s * 0.2


def all_targets_met(m: dict) -> bool:
    return (m["ic"] >= TARGET_IC and m["icir"] >= TARGET_ICIR
            and m["accuracy"] >= TARGET_ACC and m["up_win_rate"] >= TARGET_UPWR)


# ── 三阶段优化 ──────────────────────────────────────────────
def optimize(df: pd.DataFrame, code: str, name: str,
             threshold: float = 0.25, max_rounds: int = 8) -> tuple:
    """
    返回 (best_weights, best_metrics)
    多轮：随机搜索 → 爬山 → 扰动，直到满足全部目标或用完轮次
    """
    rng = np.random.default_rng(42)
    best_w = rng.standard_normal(N_FEAT)
    best_m = evaluate(df, best_w, threshold)
    best_s = score_fn(best_m)

    for rnd in range(1, max_rounds + 1):
        print(f"  [{code}] 第{rnd}轮 IC={best_m['ic']:.4f} ICIR={best_m['icir']:.4f} "
              f"Acc={best_m['accuracy']:.1f}% UpWR={best_m['up_win_rate']:.1f}%", flush=True)

        if all_targets_met(best_m):
            print(f"  ✅ 全部目标已满足，提前结束", flush=True)
            break

        # Phase 1: 随机搜索
        n_random = 20000 if rnd == 1 else 8000
        batch = rng.standard_normal((n_random, N_FEAT))
        for i in range(n_random):
            w = batch[i]
            m = evaluate(df, w, threshold)
            s = score_fn(m)
            if s > best_s:
                best_s, best_w, best_m = s, w.copy(), m

        # Phase 2: 爬山（坐标轴方向逐个微调）
        step_size = 0.5
        no_improve = 0
        for _ in range(600):
            improved = False
            for j in rng.permutation(N_FEAT):
                for delta in [step_size, -step_size]:
                    trial = best_w.copy()
                    trial[j] += delta
                    m = evaluate(df, trial, threshold)
                    s = score_fn(m)
                    if s > best_s + 1e-6:
                        best_s, best_w, best_m = s, trial.copy(), m
                        improved = True
            if not improved:
                step_size *= 0.7
                no_improve += 1
                if no_improve >= 5:
                    break

        # Phase 3: 扰动（逃出局部最优）
        for _ in range(30):
            noise = rng.standard_normal(N_FEAT) * 1.5
            trial = best_w + noise
            m = evaluate(df, trial, threshold)
            s = score_fn(m)
            if s > best_s:
                best_s, best_w, best_m = s, trial.copy(), m

    return best_w, best_m


# ── 注册模型到 DB 和 JSON ────────────────────────────────────
def save_model(code: str, name: str, weights: np.ndarray, metrics: dict,
               threshold: float, sample_days: int):
    conn = sqlite3.connect(INVEST_DB)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO stock_custom_models
            (code, name, features, weights, threshold, ic, icir, accuracy,
             up_win_rate, sample_days, description, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(code) DO UPDATE SET
            name=excluded.name, features=excluded.features,
            weights=excluded.weights, threshold=excluded.threshold,
            ic=excluded.ic, icir=excluded.icir,
            accuracy=excluded.accuracy, up_win_rate=excluded.up_win_rate,
            sample_days=excluded.sample_days,
            description=excluded.description, updated_at=excluded.updated_at
    """, (
        code, f"{name}专属模型v1",
        json.dumps(ALL_FEATURES, ensure_ascii=False),
        json.dumps([round(float(w), 4) for w in weights]),
        threshold,
        metrics["ic"], metrics["icir"], metrics["accuracy"], metrics["up_win_rate"],
        sample_days,
        f"16特征+交叉特征，top-{int(threshold*100)}%分位数阈值，自动优化",
        datetime.now().isoformat(), datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()

    # 更新 custom_models.json
    try:
        existing = json.loads(CUSTOM_JSON.read_text()) if CUSTOM_JSON.exists() else {}
    except Exception:
        existing = {}

    existing[code] = {
        "name": f"{name}专属模型v1",
        "features": ALL_FEATURES,
        "weights": [round(float(w), 4) for w in weights],
        "threshold": threshold,
        "metrics": metrics,
    }
    CUSTOM_JSON.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"  ✅ 模型已保存: DB + {CUSTOM_JSON}", flush=True)


# ── 单股优化入口 ────────────────────────────────────────────
def run_stock(code: str, name: str):
    print(f"\n{'='*60}")
    print(f"  {name}（{code}）专属模型优化")
    print(f"{'='*60}")

    df_raw = fetch_kline(code)
    if df_raw.empty:
        print(f"  ❌ 无历史数据，请先在 investment_hub 下载 {code}")
        return None

    df = build_feature_df(df_raw)
    print(f"  数据：{len(df_raw)} 行原始 → {len(df)} 行有效特征")

    if len(df) < 100:
        print(f"  ❌ 有效数据不足100条，跳过")
        return None

    t0 = time.time()
    best_w, best_m = optimize(df, code, name)
    elapsed = time.time() - t0

    print(f"\n  ─── 最终结果（耗时 {elapsed:.0f}s）───")
    print(f"  IC={best_m['ic']:.4f}  ICIR={best_m['icir']:.4f}  "
          f"Acc={best_m['accuracy']:.1f}%  UpWR={best_m['up_win_rate']:.1f}%")
    print(f"  {'✅ 全部目标达成' if all_targets_met(best_m) else '⚠️  未能完全达成目标，保存当前最优'}")

    print(f"\n  特征权重：")
    pairs = sorted(zip(ALL_FEATURES, best_w), key=lambda x: abs(x[1]), reverse=True)
    for feat, w in pairs:
        bar = "█" * int(abs(w) * 2) if abs(w) < 10 else "█" * 10
        print(f"    {feat:<18} {w:+.3f}  {bar}")

    save_model(code, name, best_w, best_m, threshold=0.25, sample_days=len(df))
    return best_m


# ── 主程序 ──────────────────────────────────────────────────
def load_watchlist():
    data = json.loads(WATCHLIST.read_text())
    return [(c, n) for c, n in data["stocks"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("code", help="股票代码 或 'all'")
    args = parser.parse_args()

    watchlist = load_watchlist()
    wl_dict   = dict(watchlist)

    if args.code == "all":
        targets = [(c, n) for c, n in watchlist if c != "300244"]  # 迪安诊断已优化
    else:
        name = wl_dict.get(args.code, args.code)
        targets = [(args.code, name)]

    results = {}
    for code, name in targets:
        m = run_stock(code, name)
        if m:
            results[code] = {"name": name, **m}

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  批量优化汇总")
        print(f"{'='*60}")
        print(f"  {'代码':<8} {'名称':<10} {'IC':>7} {'ICIR':>7} {'准确率':>8} {'上涨胜率':>9} {'达标'}")
        print(f"  {'-'*60}")
        for code, r in results.items():
            ok = all_targets_met(r)
            print(f"  {code:<8} {r['name']:<10} {r['ic']:>7.4f} {r['icir']:>7.4f} "
                  f"{r['accuracy']:>7.1f}%  {r['up_win_rate']:>7.1f}%  {'✅' if ok else '⚠️'}")

    print("\n完成。")


if __name__ == "__main__":
    main()
