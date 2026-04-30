#!/usr/bin/env python3
"""
供 investment_hub 调用的回测脚本。
逐行输出 JSON 进度，最后一行输出 {"__RESULT__": true, ...} 汇总。

若 config/custom_models.json 中有该股票的专属模型，优先使用；
否则退回内置五因子权重方案。

用法:
  python backtest_web.py                     # 全部监控股票
  python backtest_web.py --code 300244       # 单只股票
  python backtest_web.py --days 360          # 指定回测周期
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import sqlite3

def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)

# ── 加载自定义模型注册表 ──────────────────────────────────
def load_custom_models() -> dict:
    p = Path(__file__).parent / "config" / "custom_models.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_watchlist():
    p = Path(__file__).parent / "config" / "watchlist.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("stocks", [])

# ── 读取本地历史数据库 ────────────────────────────────────
def fetch_kline_from_db(code: str) -> pd.DataFrame:
    db = Path(__file__).parent / "data" / "hist_daily.db"
    if not db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db))
    df = pd.read_sql_query(
        "SELECT * FROM hist_daily WHERE code=? ORDER BY date", conn, params=(code,)
    )
    conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── 技术指标计算（通用）──────────────────────────────────
def calc_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].values

    df["change_pct"] = df["close"].pct_change(1) * 100
    df["ret1"]  = df["change_pct"]
    df["ret3"]  = df["close"].pct_change(3) * 100
    df["ret5"]  = df["close"].pct_change(5) * 100

    df["body"]      = (df["close"] - df["open"]) / df["open"] * 100
    df["hl_range"]  = (df["high"] - df["low"]) / df["close"] * 100
    df["candle_q"]  = df["body"].abs() / (df["hl_range"] + 1e-4)
    df["gap"]       = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    df["ma5"]  = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma5_slope"] = df["ma5"].diff(1) / df["ma5"] * 100
    df["dist_ma5"]  = (df["close"] - df["ma5"])  / df["ma5"]  * 100
    df["dist_ma20"] = (df["close"] - df["ma20"]) / df["ma20"] * 100

    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = e12 - e26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["dif"] - df["dea"]) * 2

    lo  = df["low"].rolling(9, min_periods=1).min()
    hi  = df["high"].rolling(9, min_periods=1).max()
    rsv = np.where((hi - lo) == 0, 50, (close - lo.values) / (hi - lo).values * 100)
    df["kdj_k"] = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    bb_std = df["close"].rolling(20).std()
    df["bb_middle"] = df["ma20"]
    df["bb_upper"]  = df["bb_middle"] + 2 * bb_std
    df["bb_lower"]  = df["bb_middle"] - 2 * bb_std
    df["bb_pos"]    = (df["close"] - df["bb_lower"]) / (4 * bb_std + 1e-9) * 100
    df["bb_squeeze"]= bb_std / df["ma20"] * 100

    vm5  = df["volume"].rolling(5).mean()
    vm10 = df["volume"].rolling(10).mean()
    df["vol_ratio"]  = df["volume"] / vm5.where(vm5 > 0, 1)
    df["vol_accel"]  = vm5 / vm10.where(vm10 > 0, 1)
    df["vol_slope"]  = df["vol_ratio"].diff(3)

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    df["pos_in_range"] = (df["close"] - df["low"].rolling(20).min()) / \
        (df["high"].rolling(20).max() - df["low"].rolling(20).min() + 1e-9) * 100

    return df

# ── Rolling Z-score 标准化 ────────────────────────────────
def add_rolling_zscore(df: pd.DataFrame, cols: list, win: int = 60) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        mu = df[c].rolling(win, min_periods=20).mean()
        st = df[c].rolling(win, min_periods=20).std()
        df[c + "_z"] = (df[c] - mu) / (st + 1e-9)
    return df

# ── 构造交叉特征 ─────────────────────────────────────────
def add_cross_features(df: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    def rzsc(s):
        mu = s.rolling(win, min_periods=20).mean()
        st = s.rolling(win, min_periods=20).std()
        return (s - mu) / (st + 1e-9)

    cross = {
        "body_vol":   df["body_z"] * df["vol_ratio_z"],
        "body_macd":  df["body_z"] * df["macd_hist_z"],
        "gap_vol":    df["gap_z"]  * df["vol_ratio_z"],
        "cq_vol":     df["candle_q_z"] * df["vol_ratio_z"],
        "macd_vol":   df["macd_hist_z"] * df["vol_ratio_z"],
        "bb_rsi":     (-df["bb_pos_z"]) * (-df["rsi_z"] + 1.5),
    }
    for name, series in cross.items():
        df[name + "_z"] = rzsc(series)
    return df

# ── 专属模型评分 ─────────────────────────────────────────
def score_custom(df: pd.DataFrame, model: dict) -> pd.Series:
    features = model["features"]
    weights  = np.array(model["weights"])
    X = np.column_stack([df[f].values for f in features])
    return pd.Series(X @ weights, index=df.index)

# ── 内置五因子评分（保持原有逻辑）───────────────────────
BUILTIN_SCHEMES = {
    "当前配置":   {"technical": 0.30, "fundamental": 0.20, "money_flow": 0.20, "sentiment": 0.15, "chip": 0.15},
    "技术面 50%": {"technical": 0.50, "fundamental": 0.15, "money_flow": 0.15, "sentiment": 0.10, "chip": 0.10},
    "技术面 60%": {"technical": 0.60, "fundamental": 0.10, "money_flow": 0.10, "sentiment": 0.10, "chip": 0.10},
    "技术面 70%": {"technical": 0.70, "fundamental": 0.10, "money_flow": 0.10, "sentiment": 0.05, "chip": 0.05},
    "均衡配置":   {"technical": 0.25, "fundamental": 0.25, "money_flow": 0.25, "sentiment": 0.15, "chip": 0.10},
    "资金主导":   {"technical": 0.20, "fundamental": 0.15, "money_flow": 0.40, "sentiment": 0.15, "chip": 0.10},
}

def score_builtin_row(row, weights):
    t = 50
    if row.get("ma5", 0) > row.get("ma10", 0) > row.get("ma20", 0):   t += 15
    elif row.get("ma5", 0) < row.get("ma10", 0) < row.get("ma20", 0): t -= 15
    t += 10 if row.get("dif", 0) > row.get("dea", 0) else -10
    k = row.get("kdj_k", 50)
    if k < 20: t += 10
    elif k > 80: t -= 10
    c = row.get("close", 0)
    if c < row.get("bb_lower", 0): t += 10
    elif c > row.get("bb_upper", 0): t -= 10
    t = max(0, min(100, t))
    f = 55
    m = 50 + (8 if row.get("vol_ratio", 1) > 1.5 else -8 if row.get("vol_ratio", 1) < 0.7 else 0)
    s = 50
    ch = row.get("change_pct", 0)
    if ch > 5: s += 15
    elif ch > 3: s += 8
    elif ch < -5: s -= 15
    elif ch < -3: s -= 8
    chip = 50
    raw = (t * weights.get("technical", 0.30) + f * weights.get("fundamental", 0.20) +
           m * weights.get("money_flow", 0.20) + s * weights.get("sentiment", 0.15) +
           chip * weights.get("chip", 0.15))
    return max(10, min(90, round(25 + raw / 100 * 60, 1)))

# ── 通用指标计算 ─────────────────────────────────────────
def calc_metrics(preds, acts):
    if len(preds) < 10:
        return {}
    rets     = np.array([a["return"] for a in acts])
    actual_b = np.array([a["up"] for a in acts])
    pred_arr = np.array(preds)

    ic = float(np.corrcoef(pred_arr, rets)[0, 1]) if np.std(pred_arr) > 0 else 0
    if np.isnan(ic): ic = 0

    win_size = 60
    rolling_ics = []
    for i in range(0, len(pred_arr) - win_size, 10):
        p_w, r_w = pred_arr[i:i+win_size], rets[i:i+win_size]
        if np.std(p_w) > 0:
            c = float(np.corrcoef(p_w, r_w)[0, 1])
            if not np.isnan(c): rolling_ics.append(c)
    ics = np.array(rolling_ics)
    icir = float(ics.mean() / ics.std()) if len(ics) > 1 and ics.std() > 0 else 0

    # 用分位数阈值区分预测上/下（内置模型用 50 作阈值，自定义模型在回测时已标记）
    thr   = np.median(pred_arr)
    pred_b = pred_arr >= thr
    acc   = float((pred_b == actual_b).mean() * 100)
    up_wr = float(actual_b[pred_b].mean() * 100) if pred_b.sum() > 0 else 0

    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0
    cum    = np.cumsum(rets)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak)) if len(cum) > 0 else 0

    return {
        "accuracy":     round(acc, 1),
        "up_win_rate":  round(up_wr, 1),
        "avg_return":   round(float(np.mean(rets)), 3),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "ic":           round(ic, 4),
        "icir":         round(icir, 4),
        "total_days":   len(preds),
    }

# ── 专属模型回测 ─────────────────────────────────────────
def run_custom_backtest(df: pd.DataFrame, model: dict):
    """用专属模型跑回测，返回 (preds, actuals)"""
    threshold = model.get("threshold", 0.25)
    score_series = score_custom(df, model)
    preds, actuals = [], []

    warmup = min(30, max(20, len(df) // 4))
    for i in range(warmup, len(df) - 1):
        score_val = score_series.iloc[i]
        if np.isnan(score_val): continue
        nc = df.iloc[i + 1]["close"]
        cc = df.iloc[i]["close"]
        preds.append(score_val)
        actuals.append({"up": nc > cc, "return": (nc - cc) / cc * 100})

    if not preds:
        return [], []

    # 用分位数阈值重新计算方向正确率（覆盖 calc_metrics 中的 median）
    preds_arr = np.array(preds)
    acts_up   = np.array([a["up"] for a in actuals])
    thr_val   = np.percentile(preds_arr, (1 - threshold) * 100)
    pred_b    = preds_arr >= thr_val
    # 把阈值信息打包进 actuals 供 calc_metrics 用（这里直接返回已算好的分类结果）
    for idx, a in enumerate(actuals):
        a["pred_up"] = bool(pred_b[idx])
    return preds, actuals

def calc_custom_metrics(preds, actuals):
    """专属模型专用指标计算（尊重 pred_up 字段）"""
    if len(preds) < 10:
        return {}
    rets     = np.array([a["return"] for a in actuals])
    pred_arr = np.array(preds)
    actual_b = np.array([a["up"] for a in actuals])
    pred_b   = np.array([a["pred_up"] for a in actuals])

    ic = float(np.corrcoef(pred_arr, rets)[0, 1]) if np.std(pred_arr) > 0 else 0
    if np.isnan(ic): ic = 0

    win_size = 60
    ics = []
    for i in range(0, len(pred_arr) - win_size, 10):
        p_w, r_w = pred_arr[i:i+win_size], rets[i:i+win_size]
        if np.std(p_w) > 0:
            c = float(np.corrcoef(p_w, r_w)[0, 1])
            if not np.isnan(c): ics.append(c)
    ics = np.array(ics)
    icir = float(ics.mean() / ics.std()) if len(ics) > 1 and ics.std() > 0 else 0

    acc   = float((pred_b == actual_b).mean() * 100)
    up_wr = float(actual_b[pred_b].mean() * 100) if pred_b.sum() > 0 else 0
    sharpe= float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0
    cum   = np.cumsum(rets)
    peak  = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak)) if len(cum) > 0 else 0

    return {
        "accuracy":     round(acc, 1),
        "up_win_rate":  round(up_wr, 1),
        "avg_return":   round(float(np.mean(rets)), 3),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "ic":           round(ic, 4),
        "icir":         round(icir, 4),
        "total_days":   len(preds),
    }

# ── 内置五因子回测 ────────────────────────────────────────
def run_builtin_backtest(df: pd.DataFrame, weights: dict):
    warmup = min(30, max(20, len(df) // 4))
    preds, actuals = [], []
    for i in range(warmup, len(df) - 1):
        row  = df.iloc[i].to_dict()
        prob = score_builtin_row(row, weights)
        nc   = df.iloc[i + 1]["close"]
        cc   = row["close"]
        preds.append(prob)
        actuals.append({"up": nc > cc, "return": (nc - cc) / cc * 100})
    return preds, actuals

# ── 读取指数日线（用于 XGBoost 相对强度特征）────────────
def _fetch_index_df() -> pd.DataFrame:
    try:
        db = Path(__file__).parent / "data" / "hist_daily.db"
        conn = sqlite3.connect(str(db))
        df = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE code='sh000300' ORDER BY date", conn
        )
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        df["index_ret"] = df["close"].pct_change() * 100
        return df[["date", "index_ret"]].dropna()
    except Exception:
        return pd.DataFrame()

def _fetch_flow_df(code: str) -> pd.DataFrame:
    try:
        db = Path(__file__).parent / "data" / "hist_daily.db"
        conn = sqlite3.connect(str(db))
        df = pd.read_sql_query(
            "SELECT date, main_pct, big_pct FROM money_flow WHERE code=? ORDER BY date",
            conn, params=(code,)
        )
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()

# ── 准备专属模型所需的特征 DataFrame ─────────────────────
def prepare_custom_df(df_raw: pd.DataFrame, model: dict, code: str = "") -> pd.DataFrame:
    """计算并标准化专属模型需要的所有特征列（支持线性/XGBoost 两种）"""
    zscore_win = model.get("zscore_win", 60)
    df = calc_base_indicators(df_raw)
    base_feats = ["body","hl_range","candle_q","gap","macd_hist","kdj_j",
                  "bb_pos","rsi","vol_ratio","vol_accel","bb_squeeze",
                  "ret1","ret3","ret5","ma5_slope","dist_ma5","dist_ma20",
                  "vol_slope","pos_in_range"]
    df = add_rolling_zscore(df, base_feats, win=zscore_win)
    df = add_cross_features(df, win=zscore_win)

    # 相对强度特征（XGBoost 模型需要）
    if "rel_str_z" in model.get("features", []):
        idx_df = _fetch_index_df()
        if not idx_df.empty:
            df = df.merge(idx_df, on="date", how="left")
            df["rel_str"]    = df["change_pct"] - df["index_ret"].fillna(0)
            df["rel_str_5d"] = df["rel_str"].rolling(5).mean()
            df = add_rolling_zscore(df, ["rel_str","rel_str_5d"], win=zscore_win)
        else:
            df["rel_str_z"] = 0.0; df["rel_str_5d_z"] = 0.0

    # 资金流特征
    if "main_pct_z" in model.get("features", []) and code:
        flow_df = _fetch_flow_df(code)
        if not flow_df.empty:
            df = df.merge(flow_df, on="date", how="left")
            df["main_pct"] = df["main_pct"].fillna(0)
            df["big_pct"]  = df["big_pct"].fillna(0)
            df = add_rolling_zscore(df, ["main_pct","big_pct"], win=zscore_win)
        else:
            df["main_pct_z"] = 0.0; df["big_pct_z"] = 0.0

    df = df.dropna().reset_index(drop=True)
    missing = [f for f in model.get("features", []) if f not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    return df

# ── XGBoost 专属模型回测 ──────────────────────────────────
def run_xgb_backtest(df: pd.DataFrame, model: dict):
    import xgboost as xgb
    model_path = model.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"XGBoost 模型文件不存在: {model_path}")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    features    = model["features"]
    threshold   = model.get("threshold", 0.25)
    feat_mat    = df[features].values
    score_series= pd.Series(xgb_model.predict_proba(feat_mat)[:, 1], index=df.index)

    preds, actuals = [], []
    warmup = min(30, max(20, len(df) // 4))
    for i in range(warmup, len(df) - 1):
        sv = score_series.iloc[i]
        if np.isnan(sv): continue
        nc, cc = df.iloc[i+1]["close"], df.iloc[i]["close"]
        preds.append(sv)
        actuals.append({"up": nc > cc, "return": (nc-cc)/cc*100})

    if not preds:
        return [], []

    preds_arr = np.array(preds)
    thr_val   = np.percentile(preds_arr, (1-threshold)*100)
    pred_b    = preds_arr >= thr_val
    for idx, a in enumerate(actuals):
        a["pred_up"] = bool(pred_b[idx])
    return preds, actuals

# ── 核心：回测单只股票 ────────────────────────────────────
def backtest_stock(code: str, name: str, idx: int, total: int,
                   days: int, custom_models: dict):
    emit({"code": code, "msg": f"[{idx}/{total}] {name}({code}) 加载历史数据…"})
    df_raw = fetch_kline_from_db(code)
    if df_raw.empty:
        emit({"code": code, "msg": "⚠️ 无历史数据，请先下载"})
        return None

    if days > 0 and len(df_raw) > days:
        df_raw = df_raw.iloc[-days:].reset_index(drop=True)

    period_label = f"近{days}天" if days > 0 else "全部"
    has_custom = code in custom_models
    emit({"code": code, "msg": f"使用{period_label}数据（{len(df_raw)}条）"
          + ("，检测到专属模型 🎯" if has_custom else "，使用内置五因子方案")})

    result = {
        "code": code, "name": name,
        "data_days": len(df_raw),
        "date_range": f"{df_raw['date'].iloc[0].date()} ~ {df_raw['date'].iloc[-1].date()}",
        "has_custom_model": has_custom,
        "schemes": {},
        "custom": None,
        "best_scheme": "",
        "best_accuracy": 0,
        "current_accuracy": 0,
        "improvement": 0,
    }

    # ── 专属模型回测 ──────────────────────────────────────
    if has_custom:
        model     = custom_models[code]
        mtype     = model.get("model_type", "linear")
        try:
            df_custom = prepare_custom_df(df_raw, model, code=code)
            if mtype == "xgboost":
                preds, actuals = run_xgb_backtest(df_custom, model)
            else:
                preds, actuals = run_custom_backtest(df_custom, model)
            m = calc_custom_metrics(preds, actuals)
            if m:
                model_tag = "🤖 XGB" if mtype == "xgboost" else "🎯 线性"
                result["custom"] = {"model_name": model.get("name","专属模型"),
                                    "model_type": mtype, "metrics": m}
                emit({"code": code,
                      "msg": f"{model_tag} 专属模型: IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
                             f"Acc={m['accuracy']:.1f}% UpWR={m['up_win_rate']:.1f}%"})
        except Exception as e:
            emit({"code": code, "msg": f"⚠️ 专属模型计算失败: {e}"})

    # ── 内置五因子方案回测 ────────────────────────────────
    df_builtin = calc_base_indicators(df_raw)
    df_builtin = df_builtin.dropna().reset_index(drop=True)

    best_acc, best_scheme = 0, ""
    for sname, weights in BUILTIN_SCHEMES.items():
        preds, actuals = run_builtin_backtest(df_builtin, weights)
        m = calc_metrics(preds, actuals)
        if m:
            result["schemes"][sname] = {"weights": weights, "metrics": m}
            if m["accuracy"] > best_acc:
                best_acc, best_scheme = m["accuracy"], sname

    current_acc = result["schemes"].get("当前配置", {}).get("metrics", {}).get("accuracy", 0)
    improvement = best_acc - current_acc
    result.update({"best_scheme": best_scheme, "best_accuracy": best_acc,
                   "current_accuracy": current_acc, "improvement": round(improvement, 1)})

    emit({"code": code,
          "msg": f"✅ 完成：内置最优「{best_scheme}」{best_acc:.1f}%（基准{current_acc:.1f}%）"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="")
    parser.add_argument("--days", type=int, default=0)
    args = parser.parse_args()

    custom_models = load_custom_models()
    watchlist     = load_watchlist()

    if args.code:
        stocks = [(c, n) for c, n in watchlist if c == args.code]
        if not stocks:
            stocks = [(args.code, args.code)]
    else:
        stocks = watchlist

    results = []
    for idx, (code, name) in enumerate(stocks, 1):
        r = backtest_stock(code, name, idx, len(stocks), args.days, custom_models)
        if r:
            results.append(r)

    emit({"__RESULT__": True, "results": results})


if __name__ == "__main__":
    main()
