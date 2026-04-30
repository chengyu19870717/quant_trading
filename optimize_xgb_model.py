#!/usr/bin/env python3
"""
XGBoost 专属模型优化 v2
改进点：
  1. 预测目标：next_return > 0.5%（过滤微涨跌噪音）
  2. 相对强度特征：个股涨幅 - 沪深300当日涨幅
  3. 资金流特征：主力净流入占比 z-score
  4. 自适应 z-score 窗口（60/120，按股票波动周期选优）
  5. 非线性模型：XGBoost（自动学习特征交叉）
  6. 时序交叉验证：TimeSeriesSplit（防止未来泄露）
  7. 模型保存：XGB 原生格式(.json) + 指标写入 DB/JSON
用法：
  python optimize_xgb_model.py all
  python optimize_xgb_model.py 603881
"""
import sys
import json
import sqlite3
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from backtest_web import (
    calc_base_indicators, add_rolling_zscore, add_cross_features
)

HIST_DB     = Path(__file__).parent / "data/hist_daily.db"
INVEST_DB   = Path.home() / ".baibao" / "baibao.db"
WATCHLIST   = Path(__file__).parent / "config/watchlist.json"
CUSTOM_JSON = Path(__file__).parent / "config/custom_models.json"
MODEL_DIR   = Path(__file__).parent / "config/xgb_models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_IC   = 0.10
TARGET_ICIR = 0.60
TARGET_ACC  = 60.0
TARGET_UPWR = 60.0
RETURN_THRESHOLD = 0.5   # 涨幅超过 0.5% 才算"上涨"信号

BASE_TECH_FEATURES = [
    "body_z","hl_range_z","candle_q_z","gap_z","macd_hist_z","kdj_j_z",
    "bb_pos_z","rsi_z","vol_ratio_z","vol_accel_z",
    "body_vol_z","body_macd_z","gap_vol_z","cq_vol_z","macd_vol_z","bb_rsi_z",
]


# ── 数据加载 ──────────────────────────────────────────────────
def fetch_kline(code: str) -> pd.DataFrame:
    conn = sqlite3.connect(HIST_DB)
    df = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume FROM hist_daily WHERE code=? ORDER BY date",
        conn, params=(code,)
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_index() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(HIST_DB)
        df = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE code='sh000300' ORDER BY date",
            conn
        )
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        df["index_ret"] = df["close"].pct_change() * 100
        return df[["date", "index_ret"]].dropna()
    except Exception:
        return pd.DataFrame()


def fetch_money_flow(code: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(HIST_DB)
        df = pd.read_sql_query(
            "SELECT date, main_pct, big_pct FROM money_flow WHERE code=? ORDER BY date",
            conn, params=(code,)
        )
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def build_feature_df(df_raw: pd.DataFrame, index_df: pd.DataFrame,
                     flow_df: pd.DataFrame, zscore_win: int = 60) -> pd.DataFrame:
    df = calc_base_indicators(df_raw)

    base = ["body","hl_range","candle_q","gap","macd_hist","kdj_j",
            "bb_pos","rsi","vol_ratio","vol_accel","bb_squeeze",
            "ret1","ret3","ret5","ma5_slope","dist_ma5","dist_ma20",
            "vol_slope","pos_in_range"]
    df = add_rolling_zscore(df, base, win=zscore_win)
    df = add_cross_features(df, win=zscore_win)

    # 相对大盘强度
    if not index_df.empty:
        df = df.merge(index_df, on="date", how="left")
        df["rel_str"]    = df["change_pct"] - df["index_ret"].fillna(0)
        df["rel_str_5d"] = df["rel_str"].rolling(5).mean()
        df = add_rolling_zscore(df, ["rel_str", "rel_str_5d"], win=zscore_win)
    else:
        df["rel_str_z"] = 0.0
        df["rel_str_5d_z"] = 0.0

    # 主力资金流
    if not flow_df.empty:
        df = df.merge(flow_df, on="date", how="left")
        df["main_pct"] = df["main_pct"].fillna(0)
        df["big_pct"]  = df["big_pct"].fillna(0)
        df = add_rolling_zscore(df, ["main_pct", "big_pct"], win=zscore_win)
    else:
        df["main_pct_z"] = 0.0
        df["big_pct_z"]  = 0.0

    # 目标变量
    df["next_ret"]    = df["close"].shift(-1) / df["close"] * 100 - 100
    df["target"]      = (df["next_ret"] > RETURN_THRESHOLD).astype(int)
    df["next_up_raw"] = df["close"].shift(-1) > df["close"]

    all_feat = BASE_TECH_FEATURES + ["rel_str_z","rel_str_5d_z","main_pct_z","big_pct_z"]
    df = df.dropna(subset=all_feat + ["next_ret", "target"]).reset_index(drop=True)
    return df


# ── 评估指标 ──────────────────────────────────────────────────
def calc_metrics(scores: np.ndarray, df: pd.DataFrame) -> dict:
    rets     = df["next_ret"].values
    actual_b = df["next_up_raw"].values
    thr_val  = np.percentile(scores, 75)
    pred_up  = scores >= thr_val

    ic = float(np.corrcoef(scores, rets)[0, 1]) if np.std(scores) > 0 else 0
    if np.isnan(ic): ic = 0

    win, step = 60, 10
    ics = []
    for i in range(0, len(scores) - win, step):
        p_w, r_w = scores[i:i+win], rets[i:i+win]
        if np.std(p_w) > 0:
            c = float(np.corrcoef(p_w, r_w)[0, 1])
            if not np.isnan(c): ics.append(c)
    ics_arr = np.array(ics)
    icir = float(ics_arr.mean() / ics_arr.std()) if len(ics_arr) > 1 and ics_arr.std() > 0 else 0

    acc   = float((pred_up == actual_b).mean() * 100)
    up_wr = float(actual_b[pred_up].mean() * 100) if pred_up.sum() > 0 else 0

    return {
        "ic": round(ic, 4), "icir": round(icir, 4),
        "accuracy": round(acc, 2), "up_win_rate": round(up_wr, 2),
        "total_days": len(scores),
    }


def all_targets_met(m: dict) -> bool:
    return (m["ic"] >= TARGET_IC and m["icir"] >= TARGET_ICIR
            and m["accuracy"] >= TARGET_ACC and m["up_win_rate"] >= TARGET_UPWR)


# ── 自适应窗口 ────────────────────────────────────────────────
def pick_zscore_win(df_raw: pd.DataFrame) -> int:
    rets = df_raw["close"].pct_change().dropna()
    ac1  = rets.autocorr(1)
    ac5  = rets.autocorr(5)
    return 120 if (abs(ac1) >= 0.02 or abs(ac5) >= 0.05) else 60


# ── 超参搜索 ──────────────────────────────────────────────────
PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_weight": 3,
     "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
     "colsample_bytree": 0.7, "min_child_weight": 5,
     "gamma": 0.2, "reg_alpha": 0.2, "reg_lambda": 1.5, "n_estimators": 400},
    {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.7,
     "colsample_bytree": 0.8, "min_child_weight": 5,
     "gamma": 0.05, "reg_alpha": 0.05, "reg_lambda": 2.0, "n_estimators": 500},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9,
     "colsample_bytree": 0.6, "min_child_weight": 3,
     "gamma": 0.3, "reg_alpha": 0.3, "reg_lambda": 1.0, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.02, "subsample": 0.8,
     "colsample_bytree": 0.9, "min_child_weight": 10,
     "gamma": 0.1, "reg_alpha": 0.5, "reg_lambda": 2.0, "n_estimators": 600},
]


def make_model(params: dict, spw: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )


def hyperparam_search(df: pd.DataFrame, features: list, spw: float) -> dict:
    X = df[features].values
    y = df["target"].values
    tscv = TimeSeriesSplit(n_splits=5)

    best_score  = -999
    best_params = PARAM_GRID[0]

    for params in PARAM_GRID:
        fold_scores = []
        for tr_idx, va_idx in tscv.split(X):
            if len(va_idx) < 30:
                continue
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = make_model(params, spw)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      verbose=False)
            proba    = model.predict_proba(X_va)[:, 1]
            fold_df  = df.iloc[va_idx].copy()
            m        = calc_metrics(proba, fold_df)
            s = (min(m["ic"]/TARGET_IC, 2)*0.3 + min(m["icir"]/TARGET_ICIR, 2)*0.3
                 + min(m["accuracy"]/TARGET_ACC, 2)*0.2
                 + min(m["up_win_rate"]/TARGET_UPWR, 2)*0.2)
            fold_scores.append(s)

        if fold_scores and np.mean(fold_scores) > best_score:
            best_score  = np.mean(fold_scores)
            best_params = params

    return best_params


# ── 保存模型 ──────────────────────────────────────────────────
def save_model(code: str, name: str, model: xgb.XGBClassifier,
               features: list, metrics: dict, zscore_win: int, sample_days: int):
    # 用 XGBoost 原生 JSON 格式保存（安全）
    model_path = MODEL_DIR / f"{code}_xgb.json"
    model.save_model(str(model_path))

    # 写 DB
    conn = sqlite3.connect(INVEST_DB)
    conn.execute("""
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
        code, f"{name}XGB模型v2",
        json.dumps(features, ensure_ascii=False),
        json.dumps([]),
        0.25,
        metrics["ic"], metrics["icir"], metrics["accuracy"], metrics["up_win_rate"],
        sample_days,
        f"XGBoost，目标>0.5%涨幅，含相对强度+资金流，z-score窗口={zscore_win}",
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
        "name": f"{name}XGB模型v2",
        "features": features,
        "weights": [],
        "threshold": 0.25,
        "model_type": "xgboost",
        "model_path": str(model_path),
        "zscore_win": zscore_win,
        "metrics": metrics,
    }
    CUSTOM_JSON.write_text(json.dumps(existing, ensure_ascii=False, indent=2))


# ── 单股优化 ──────────────────────────────────────────────────
def run_stock(code: str, name: str) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  {name}（{code}）XGBoost 专属模型优化")
    print(f"{'='*60}")

    df_raw  = fetch_kline(code)
    idx_df  = fetch_index()
    flow_df = fetch_money_flow(code)

    if df_raw.empty:
        print("  ❌ 无历史数据"); return None

    zscore_win = pick_zscore_win(df_raw)
    print(f"  数据：{len(df_raw)} 行 | z-score窗口={zscore_win} | "
          f"指数={'有' if not idx_df.empty else '无'} | "
          f"资金流={'有' if not flow_df.empty else '无'}")

    use_features = BASE_TECH_FEATURES + ["rel_str_z","rel_str_5d_z"]
    if not flow_df.empty:
        use_features += ["main_pct_z", "big_pct_z"]

    df = build_feature_df(df_raw, idx_df, flow_df, zscore_win)
    if len(df) < 80:
        print("  ❌ 有效样本不足80条"); return None

    pos_rate = df["target"].mean()
    spw = round((1 - pos_rate) / pos_rate, 2) if pos_rate > 0 else 1.0
    print(f"  有效样本={len(df)} | 上涨信号比={pos_rate:.1%} | scale_pos_weight={spw}")

    # 超参搜索
    print(f"  超参搜索（5折时序CV，{len(use_features)}特征）…", flush=True)
    best_params = hyperparam_search(df, use_features, spw)
    print(f"  最优超参: depth={best_params['max_depth']} lr={best_params['learning_rate']} "
          f"trees={best_params['n_estimators']}")

    # 全量训练 + 测试集评估（最后20%）
    split    = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    X_tr, y_tr = train_df[use_features].values, train_df["target"].values
    X_te       = test_df[use_features].values

    y_te  = test_df["target"].values
    model = make_model(best_params, spw)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    scores  = model.predict_proba(X_te)[:, 1]
    metrics = calc_metrics(scores, test_df)

    print(f"\n  ─── 测试集（最后 {len(test_df)} 天）───")
    print(f"  IC={metrics['ic']:.4f}  ICIR={metrics['icir']:.4f}  "
          f"Acc={metrics['accuracy']:.1f}%  UpWR={metrics['up_win_rate']:.1f}%")
    print(f"  {'✅ 全部目标达成' if all_targets_met(metrics) else '⚠️  部分目标未达，保存当前最优'}")

    # 特征重要性
    imp = sorted(zip(use_features, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print(f"\n  特征重要性 Top10：")
    for feat, score in imp[:10]:
        bar = "█" * int(score * 300)
        print(f"    {feat:<20} {score:.4f}  {bar}")

    save_model(code, name, model, use_features, metrics, zscore_win, len(df))
    print(f"  ✅ 模型已保存: {MODEL_DIR}/{code}_xgb.json")
    return metrics


# ── 主程序 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("code", help="股票代码 或 'all'")
    args = parser.parse_args()

    stocks  = json.loads(WATCHLIST.read_text())["stocks"]
    wl_dict = dict(stocks)

    targets = stocks if args.code == "all" else [(args.code, wl_dict.get(args.code, args.code))]

    results = {}
    for code, name in targets:
        m = run_stock(code, name)
        if m:
            results[code] = {"name": name, **m}

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  XGBoost 批量优化汇总")
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
