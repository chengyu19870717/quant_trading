# 股神计划 — 概要设计文档

## 项目概述

**项目名称：** 股神计划  
**类型：** A 股量化选股框架  
**语言：** Python 3  
**路径：** `~/Desktop/quant_trading/`  
**用途：** 每日对监控股票池进行多维度量化评分，生成 Markdown 复盘报告

---

## 技术栈

| 组件 | 说明 |
|---|---|
| akshare | A 股数据源（行情、基本面、筹码） |
| pandas / numpy | 数据处理 / 线性回归 |
| Python venv | `/Users/chengyu/PycharmProjects/PythonProject/.venv/` |

---

## 目录结构

```
quant_trading/
├── main.py              # 主入口
├── requirements.txt
├── reports/             # 输出报告（YYYYMMDD_report.md）
├── logs/
└── src/
    ├── data_collector.py  # 行情 & 资金数据采集
    ├── indicators.py      # 技术指标计算（MACD/KDJ/布林带等）
    ├── fundamentals.py    # 基本面分析
    ├── chip_analyzer.py   # 筹码集中度分析
    ├── ai_scorer.py       # 五维综合评分
    └── reporter.py        # Markdown 报告生成
```

---

## 核心流程

```
main.py → analyze_stock(code)
  Step 1: data_collector   → 采集日行情、资金流向
  Step 2: indicators        → 计算技术指标
  Step 3: fundamentals      → 获取基本面数据
  Step 4: chip_analyzer     → 筹码集中度分析（15日趋势）
  Step 5: ai_scorer         → 五维评分 → 综合概率
  Step 6: reporter          → 写入 reports/YYYYMMDD_report.md
```

**运行方式：**
```bash
# 使用 venv 运行（必须）
/Users/chengyu/PycharmProjects/PythonProject/.venv/bin/python main.py
/Users/chengyu/PycharmProjects/PythonProject/.venv/bin/python main.py --date 2026-04-09
/Users/chengyu/PycharmProjects/PythonProject/.venv/bin/python main.py --stock 300244
```

---

## 评分体系（ai_scorer.py）

| 维度 | 权重 | 数据来源 |
|---|---|---|
| 技术面 | 30% | MACD / KDJ / 布林带 / 均线 |
| 基本面 | 20% | PE / PB / 营收增速 |
| 资金面 | 20% | 主力净流入 / 换手率 / 量比 |
| 情绪面 | 15% | 涨跌幅 / 市场热度 |
| 筹码面 | 15% | 筹码宽度 / 获利比例 / 15日收敛趋势 |

**输出：** 0~100 综合得分 → 折算为明日上涨概率（%）

---

## 筹码分析逻辑（chip_analyzer.py）

数据源：`akshare.stock_cyq_em()`

| 信号 | 触发条件 | 含义 |
|---|---|---|
| `CHIP_CONVERGING` | 15日宽度线性回归 slope < 0 | 筹码持续收敛，看多 |
| `CHIP_TIGHT_LOW_PROFIT` | 70%筹码在股价±10%区间 + 宽度<7% + 获利<20% | 高度集中低获利，看多 |
| `CHIP_WIDE_LOW_PROFIT` | 70%筹码在股价±10%区间 + 宽度>15% + 获利<20% | 分散低获利，看多 |

---

## 报告格式（reporter.py）

输出文件：`reports/YYYYMMDD_report.md`  
每只股票包含：
- 明日上涨概率（%）
- 收盘价 / 涨跌幅
- 五维评分条（★☆ 格式）
- 筹码集中度表（70%区间 / 宽度 / 获利比例 / 15日趋势条形图）
- 技术指标表（MACD / KDJ / 布林带 / 换手率 / 量比）
- 技术信号文字列表

---

## 监控股票池（截至 2026-04-09）

| 名称 | 代码 |
|---|---|
| 迪安诊断 | 300244 |
| 宏景科技 | 待确认 |
| 中文在线 | 待确认 |
| 数据港 | 待确认 |
| 创新医疗 | 待确认 |

---

## 开发规范

- Python 必须使用 venv：`/Users/chengyu/PycharmProjects/PythonProject/.venv/`
- 新增分析维度：在 `src/` 下新建模块，在 `main.py` 的 `analyze_stock()` 中按 Step 顺序插入
- 新增评分维度：修改 `ai_scorer.py`，调整权重确保总和为 100%
- 报告变更：只改 `reporter.py`，格式以 Markdown 为主（供百宝箱 Web 解析）
