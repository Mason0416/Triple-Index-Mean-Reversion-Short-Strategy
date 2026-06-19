"""指數 region → 放空當日強勢股 回測（多漲幅門檻掃描）。

「強勢股」= 當日漲幅 (close - prev_close)/prev_close 達門檻、且 < 9.5%
（排除接近漲停的股票，因為漲停是強制停板、不代表真實買盤需求）。

對三個指數（台指期 TX、加權指數 TAIEX、櫃買指數 TPEx）× 四個 region
× 六個漲幅門檻（5%~10%）：以「指數當日 region」分桶，標的為該日強勢股，
於隔日開盤放空、收盤回補。

成本：手續費單邊 0.001425×0.6（買賣皆收）、賣出交易稅 0.003×0.2（放空在
進場賣出時收）。股數 1000。
"""

import glob
import math
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader
from data_loader import (
    calc_turnover,
    get_day_trading_suspension,
    get_tx_data,
    load_market_value_multiple,
)
from market_scanner import filter_day_trading_eligible


# 中文字型（避免圖表中文變方塊）
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang TC", "Heiti TC", "STHeiti",
    "Microsoft JhengHei", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


START_DATE = "2015-01-01"
END_DATE = "2026-06-01"

# ===================== 最終策略參數 =====================
STOP_LOSS_PCT = 0.08                      # 做空停損 8%
SLIPPAGE_PCT = 0.0015                     # 單邊滑價 0.15%
COMMISSION_RATE = 0.001425               # 券商手續費率（單邊）
COMMISSION_DISCOUNT = 0.6                # 手續費折扣（六折）
TAX_RATE = 0.003                         # 證交稅率
DAYTRADE_TAX_DISCOUNT = 0.2              # 當沖交易稅折扣（二折）
SHARES = 1000                            # 每筆股數（一張）
STRONG_STOCK_MIN = 0.08                  # 強勢股漲幅下限 8%
STRONG_STOCK_MAX = 0.095                 # 強勢股漲幅上限 9.5%（排除接近漲停）
MARKET_VALUE_THRESHOLD = 10_000_000_000  # 市值上限 100 億
TURNOVER_THRESHOLD = 0.005               # 周轉率下限 0.5%
SIGNAL_COMBO = ["TX", "TAIEX", "TPEx"]   # 三指數 AND 開低走高
# =======================================================

# 生效成本率（供回測使用）
FEE_RATE = COMMISSION_RATE * COMMISSION_DISCOUNT       # 手續費單邊（六折）
EFFECTIVE_TAX_RATE = TAX_RATE * DAYTRADE_TAX_DISCOUNT  # 當沖賣出交易稅（二折）

REGIONS = ["開高走高", "開高走低", "開低走高", "開低走低"]
THRESHOLDS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
UPPER_EXCLUDE = STRONG_STOCK_MAX   # 漲幅 > 9.5% 視為接近漲停，排除
OUTPUT_DIR = "output"
DATA_DIR = "data"

_NON_STOCK_CSV = {"TAIEX", "TPEx", "sp500_returns"}


def get_loader() -> DataLoader:
    """建立並（若有 token）登入 FinMind DataLoader。

    Returns:
        已盡可能登入的 DataLoader 實例。
    """
    load_dotenv()
    api = DataLoader()
    token = os.getenv("FINMIND_TOKEN")
    if token:
        api.login_by_token(api_token=token)
    return api


def _standardize_index(raw: pd.DataFrame) -> pd.DataFrame:
    """把 TaiwanStockPrice 指數原始資料轉為標準 OHLC（date 索引、float）。

    Args:
        raw: FinMind 原始 DataFrame（含 date, open, max, min, close）。

    Returns:
        DataFrame，index=date，欄位 open/high/low/close。
    """
    df = raw.rename(columns={"max": "high", "min": "low"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close"]]


def get_index_ohlcv(api: DataLoader, data_id: str,
                    start_date: str, end_date: str) -> pd.DataFrame:
    """抓取指數日線（TaiwanStockPrice）並帶本地 CSV 快取。

    Args:
        api: DataLoader 實例。
        data_id: 'TAIEX'（加權）或 'TPEx'（櫃買）。
        start_date: 起始日期。
        end_date: 結束日期。

    Returns:
        DataFrame，index=date，欄位 open/high/low/close。
    """
    csv_path = os.path.join(DATA_DIR, f"{data_id}.csv")
    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        for col in ["open", "high", "low", "close"]:
            cached[col] = cached[col].astype(float)
        return cached.sort_index()

    raw = api.get_data(
        dataset="TaiwanStockPrice", data_id=data_id,
        start_date=start_date, end_date=end_date,
    )
    if raw is None or raw.empty:
        raise ValueError(f"指數 {data_id} 回傳空資料")
    df = _standardize_index(raw)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path)
    return df


def load_stock_data_cache_only() -> dict:
    """從 data/*.csv 讀取所有個股日線（cache-only，不呼叫 API）。

    自動排除指數／S&P500 等非個股快取檔。

    Returns:
        dict[stock_id -> 標準 OHLCV DataFrame]。
    """
    data = {}
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        stock_id = os.path.splitext(os.path.basename(path))[0]
        if stock_id in _NON_STOCK_CSV:
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            df = data_loader._to_standard_ohlcv(df)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            continue
        if not df.empty:
            data[stock_id] = df
    return data


def build_all_gains(data: dict) -> pd.DataFrame:
    """攤平所有個股的每日漲幅 (close - prev_close)/prev_close。

    Args:
        data: dict[stock_id -> 標準 OHLCV]。

    Returns:
        long-format DataFrame（date, stock_id, gain），已去除首日 NaN。
    """
    frames = []
    for stock_id, df in data.items():
        if "close" not in df.columns or len(df) < 2:
            continue
        gain = df["close"].astype(float).pct_change()
        frames.append(pd.DataFrame({
            "date": df.index, "stock_id": str(stock_id), "gain": gain.values,
        }))
    if not frames:
        return pd.DataFrame(columns=["date", "stock_id", "gain"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["gain"])


def strong_events(all_gains: pd.DataFrame, threshold: float,
                  upper: float = UPPER_EXCLUDE) -> pd.DataFrame:
    """取得漲幅在 [threshold, upper] 區間的強勢股事件 (date, stock_id)。

    Args:
        all_gains: build_all_gains 輸出。
        threshold: 漲幅下限（如 0.05）。
        upper: 漲幅上限（預設 0.095，排除接近漲停）。

    Returns:
        DataFrame（date, stock_id）。
    """
    mask = (all_gains["gain"] >= threshold) & (all_gains["gain"] <= upper)
    ev = all_gains.loc[mask, ["date", "stock_id"]].copy()
    return ev.drop_duplicates().reset_index(drop=True)


def classify_regions(df: pd.DataFrame) -> pd.Series:
    """以「開高/開低 × 走高/走低」將每日分類為四種 region。

    Args:
        df: 標準 OHLCV DataFrame（index=date，含 open/close）。

    Returns:
        pd.Series，index 同 df，值為四種 region 字串或 NaN。
    """
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    is_open_high = open_ > prev_close
    is_open_low = open_ < prev_close
    is_up = close > open_
    is_down = close < open_

    regions = pd.Series(np.nan, index=df.index, dtype=object)
    regions[is_open_high & is_up] = "開高走高"
    regions[is_open_high & is_down] = "開高走低"
    regions[is_open_low & is_up] = "開低走高"
    regions[is_open_low & is_down] = "開低走低"
    return regions


def _short_trade_pnl(bar: pd.Series) -> float:
    """單筆「隔日開盤放空、收盤回補」損益（含滑價 + 8% 停損）。

    做空：
      進場價 = open × (1 - 滑價)        # 放空滑價往下
      停損價 = 進場價 × (1 + 停損%)      # 做空停損往上
      若當日 high >= 停損價 → 以停損價出場
      否則出場價 = close × (1 + 滑價)    # 回補滑價往上
    pnl = (進場 - 出場) × 股數 - 雙邊手續費 - 賣出交易稅。

    Args:
        bar: 個股隔日 OHLC（含 open/high/close）。

    Returns:
        該筆損益（float）；open/close 非正時回傳 NaN。
    """
    open_ = float(bar["open"])
    high = float(bar["high"])
    close = float(bar["close"])
    if open_ <= 0 or close <= 0:
        return float("nan")

    entry = open_ * (1 - SLIPPAGE_PCT)
    stop_price = entry * (1 + STOP_LOSS_PCT)
    if high >= stop_price:
        exit_price = stop_price
    else:
        exit_price = close * (1 + SLIPPAGE_PCT)

    gross = (entry - exit_price) * SHARES
    cost = ((entry + exit_price) * SHARES * FEE_RATE
            + entry * SHARES * EFFECTIVE_TAX_RATE)
    return float(gross - cost)


def backtest_short_events(index_regions: pd.Series,
                          events: pd.DataFrame,
                          data: dict) -> dict:
    """依「指數當日 region」分桶，放空事件個股，計算每筆損益。

    對每筆事件 (event_date D, stock_id X)：取指數在 D 的 region；
    若屬四種 region 之一，於 X 的下一個交易日開盤放空、收盤回補。
    放空：pnl = (entry - exit) * 1000 - 成本（雙邊手續費 + 賣出交易稅）。
    若 X 在 D 之後無資料則略過該筆。

    Args:
        index_regions: 指數每日 region 序列（index=date）。
        events: 事件 (date, stock_id)。
        data: dict[stock_id -> 標準 OHLCV]。

    Returns:
        dict[region -> 交易 DataFrame(date, stock_id, pnl, event_date)]。
    """
    buckets = {region: [] for region in REGIONS}

    for _, row in events.iterrows():
        event_date = pd.Timestamp(row["date"])
        stock_id = str(row["stock_id"])

        region = index_regions.get(event_date)
        if region not in REGIONS:
            continue

        df = data.get(stock_id)
        if df is None:
            continue

        pos = df.index.searchsorted(event_date, side="right")
        if pos >= len(df.index):
            continue
        next_day = df.index[pos]

        pnl = _short_trade_pnl(df.loc[next_day])
        if pnl != pnl:  # NaN（open/close 非正）
            continue
        buckets[region].append({
            "date": next_day, "stock_id": stock_id,
            "pnl": pnl, "event_date": event_date,
        })

    region_trades = {}
    for region in REGIONS:
        tdf = pd.DataFrame(buckets[region],
                           columns=["date", "stock_id", "pnl", "event_date"])
        if not tdf.empty:
            tdf = tdf.sort_values("date").reset_index(drop=True)
        region_trades[region] = tdf
    return region_trades


def compute_stats(trades: pd.DataFrame) -> dict:
    """計算單一 region 交易的績效指標。

    Args:
        trades: backtest_short_events 的單一 region 輸出。

    Returns:
        dict：n_trades, n_days, total_pnl, win_rate, sharpe, max_drawdown。
    """
    if trades.empty:
        return {"n_trades": 0, "n_days": 0, "total_pnl": 0.0,
                "win_rate": float("nan"), "sharpe": 0.0, "max_drawdown": 0.0}

    pnl = trades["pnl"]
    std = pnl.std()
    sharpe = float(pnl.mean() / std * math.sqrt(252)) if std and std > 0 else 0.0
    equity = pnl.cumsum()
    max_drawdown = float((equity - equity.cummax()).min())

    return {
        "n_trades": int(len(pnl)),
        "n_days": int(trades["event_date"].nunique()),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def print_threshold_summary(index_name: str, rows: list) -> None:
    """印出某指數「門檻 × region」的績效摘要表。

    Args:
        index_name: 指數名稱。
        rows: list of (threshold, region, stats dict)。
    """
    print("\n" + "=" * 80)
    print(f"{index_name} region → 放空強勢股 摘要（依門檻 × region）")
    print("=" * 80)
    print(f"{'門檻':>5} | {'region':<8}| {'交易筆數':>6} | {'觸發天數':>6} | "
          f"{'總損益':>11} | {'勝率':>7} | {'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 92)
    for threshold, region, s in rows:
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{threshold:>5.0%} | {region:<8}| {s['n_trades']:>6} | "
              f"{s['n_days']:>6} | {s['total_pnl']:>11,.0f} | {wr:>7} | "
              f"{s['sharpe']:>8.4f} | {s['max_drawdown']:>11,.0f}")


def plot_sharpe_vs_threshold(index_name: str, sharpe_by_region: dict,
                             out_path: str) -> None:
    """畫「Sharpe vs 漲幅門檻」折線圖（每個 region 一條線）。

    Args:
        index_name: 指數名稱。
        sharpe_by_region: dict[region -> 各門檻 Sharpe 清單（與 THRESHOLDS 對齊）]。
        out_path: 圖檔輸出路徑。
    """
    x_labels = [f"{t:.0%}" for t in THRESHOLDS]
    plt.figure(figsize=(10, 6))
    for region in REGIONS:
        plt.plot(x_labels, sharpe_by_region[region], marker="o", label=region)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"{index_name}：強勢股隔日放空 Sharpe vs 漲幅門檻")
    plt.xlabel("漲幅門檻")
    plt.ylabel("Sharpe（年化）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def run_index_thresholds(index_name: str, index_df: pd.DataFrame,
                         events_by_thr: dict, data: dict,
                         out_path: str) -> None:
    """對單一指數跑所有門檻 × region 回測，印摘要並畫 Sharpe 折線圖。

    Args:
        index_name: 指數名稱。
        index_df: 指數標準 OHLC（提供 region 訊號）。
        events_by_thr: dict[threshold -> 強勢股事件 DataFrame（已過濾當沖）]。
        data: 個股 OHLCV 字典。
        out_path: Sharpe 折線圖輸出路徑。
    """
    regions = classify_regions(index_df)
    rows = []
    sharpe_by_region = {region: [] for region in REGIONS}

    for threshold in THRESHOLDS:
        region_trades = backtest_short_events(
            regions, events_by_thr[threshold], data)
        for region in REGIONS:
            stats = compute_stats(region_trades[region])
            rows.append((threshold, region, stats))
            sharpe_by_region[region].append(stats["sharpe"])

    print_threshold_summary(index_name, rows)
    plot_sharpe_vs_threshold(index_name, sharpe_by_region, out_path)


def yearly_breakdown(label: str, trades: pd.DataFrame,
                     out_path: str) -> None:
    """對單一 region×門檻 的交易做逐年拆解：印表 + 畫逐年總損益長條圖。

    Args:
        label: 標籤（表頭與圖標題用），例如 "TPEx 開低走高 × 8%"。
        trades: backtest_short_events 的單一 region 輸出
            （含 date, pnl, event_date）。
        out_path: 長條圖輸出路徑。
    """
    print("\n" + "=" * 60)
    print(f"{label} 逐年拆解")
    print("=" * 60)

    if trades.empty:
        print("（無交易）")
        return

    t = trades.copy()
    t["year"] = pd.to_datetime(t["event_date"]).dt.year

    print(f"{'年份':<6}| {'交易筆數':>6} | {'觸發天數':>6} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>9}")
    print("-" * 60)
    years, totals = [], []
    for year, sub in t.groupby("year"):
        pnl = sub["pnl"]
        std = pnl.std()
        sharpe = float(pnl.mean() / std * math.sqrt(252)) if std and std > 0 else 0.0
        print(f"{year:<6}| {len(pnl):>6} | {sub['event_date'].nunique():>6} | "
              f"{pnl.sum():>11,.0f} | {(pnl > 0).mean():>6.2%} | {sharpe:>9.4f}")
        years.append(str(year))
        totals.append(float(pnl.sum()))

    colors = ["tab:green" if v >= 0 else "tab:red" for v in totals]
    plt.figure(figsize=(12, 6))
    plt.bar(years, totals, color=colors)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"{label} 逐年總損益")
    plt.xlabel("年份")
    plt.ylabel("總損益（元）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def build_event_trades(events: pd.DataFrame, data: dict) -> pd.DataFrame:
    """對事件集計算每筆「隔日放空」損益（不分 region，供組合分析共用）。

    Args:
        events: 事件 (date, stock_id)。
        data: dict[stock_id -> 標準 OHLCV]。

    Returns:
        DataFrame（event_date, date, pnl），date 為實際進出場的隔日。
    """
    records = []
    for _, row in events.iterrows():
        event_date = pd.Timestamp(row["date"])
        stock_id = str(row["stock_id"])
        df = data.get(stock_id)
        if df is None:
            continue
        pos = df.index.searchsorted(event_date, side="right")
        if pos >= len(df.index):
            continue
        next_day = df.index[pos]
        pnl = _short_trade_pnl(df.loc[next_day])
        if pnl != pnl:  # NaN（open/close 非正）
            continue
        records.append({"event_date": event_date, "date": next_day,
                        "stock_id": stock_id, "pnl": pnl})
    return pd.DataFrame(records,
                        columns=["event_date", "date", "stock_id", "pnl"])


def signal_combo_analysis(regions_map: dict, events: pd.DataFrame,
                          data: dict, out_path: str) -> None:
    """測試「開低走高」訊號的 7 種指數組合（單一 / AND 交集）並繪雙軸圖。

    對每個組合，訊號日 = 組合內所有指數當日都為「開低走高」的交集日；
    在這些日的強勢股隔日放空。印摘要表並畫 Sharpe（左軸）與總損益（右軸）
    的雙 Y 軸長條圖。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 該指數每日 region 序列]。
        events: 強勢股事件（已含門檻與當沖過濾，例如 8% 門檻）。
        data: 個股 OHLCV 字典。
        out_path: 圖檔輸出路徑。
    """
    all_trades = build_event_trades(events, data)
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}

    combos = [
        ("只用 TX", ["TX"]),
        ("只用 TAIEX", ["TAIEX"]),
        ("只用 TPEx", ["TPEx"]),
        ("TX AND TAIEX", ["TX", "TAIEX"]),
        ("TX AND TPEx", ["TX", "TPEx"]),
        ("TAIEX AND TPEx", ["TAIEX", "TPEx"]),
        ("TX AND TAIEX AND TPEx", ["TX", "TAIEX", "TPEx"]),
    ]

    print("\n" + "=" * 80)
    print("開低走高 訊號組合分析（8% 強勢股隔日放空）")
    print("=" * 80)
    print(f"{'組合':<22}| {'觸發天數':>6} | {'交易筆數':>6} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 92)

    labels, sharpes, totals = [], [], []
    for label, members in combos:
        sig_dates = set.intersection(*[kdzg[m] for m in members])
        sub = all_trades[all_trades["event_date"].isin(sig_dates)]
        sub = sub.sort_values("date").reset_index(drop=True)
        s = compute_stats(sub)
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{label:<22}| {s['n_days']:>6} | {s['n_trades']:>6} | "
              f"{s['total_pnl']:>11,.0f} | {wr:>7} | {s['sharpe']:>8.4f} | "
              f"{s['max_drawdown']:>11,.0f}")
        labels.append(label)
        sharpes.append(s["sharpe"])
        totals.append(s["total_pnl"])

    # 雙 Y 軸長條圖：左軸 Sharpe、右軸 總損益
    x = np.arange(len(labels))
    width = 0.4
    fig, ax1 = plt.subplots(figsize=(14, 7))
    bars1 = ax1.bar(x - width / 2, sharpes, width,
                    color="tab:blue", label="Sharpe")
    ax1.set_ylabel("Sharpe（年化）", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, totals, width,
                    color="tab:orange", label="總損益")
    ax2.set_ylabel("總損益（元）", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_title("開低走高 訊號組合：Sharpe（左）vs 總損益（右）")
    ax1.legend([bars1, bars2], ["Sharpe", "總損益"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] 已存 {out_path}")


def combo_yearly_breakdown(regions_map: dict, events: pd.DataFrame,
                           data: dict, out_path: str) -> None:
    """「TX AND TAIEX AND TPEx 開低走高 × 8%」逐年拆解 + 長條圖。

    訊號日 = 三指數當日都為「開低走高」的交集日；對其強勢股隔日放空，
    依年份印 年份｜觸發天數｜交易筆數｜總損益｜勝率｜Sharpe，並畫逐年總損益圖。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 每日 region 序列]。
        events: 強勢股事件（8% 門檻、已過濾當沖）。
        data: 個股 OHLCV 字典。
        out_path: 長條圖輸出路徑。
    """
    all_trades = build_event_trades(events, data)
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}
    sig_dates = set.intersection(*[kdzg[m] for m in ["TX", "TAIEX", "TPEx"]])
    sub = all_trades[all_trades["event_date"].isin(sig_dates)].copy()

    print("\n" + "=" * 60)
    print("TX AND TAIEX AND TPEx 開低走高 × 8% 逐年拆解")
    print("=" * 60)
    if sub.empty:
        print("（無交易）")
        return

    sub["year"] = pd.to_datetime(sub["event_date"]).dt.year
    print(f"{'年份':<6}| {'觸發天數':>6} | {'交易筆數':>6} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>9}")
    print("-" * 62)
    years, totals = [], []
    for year, g in sub.groupby("year"):
        pnl = g["pnl"]
        std = pnl.std()
        sharpe = float(pnl.mean() / std * math.sqrt(252)) if std and std > 0 else 0.0
        print(f"{year:<6}| {g['event_date'].nunique():>6} | {len(pnl):>6} | "
              f"{pnl.sum():>11,.0f} | {(pnl > 0).mean():>6.2%} | {sharpe:>9.4f}")
        years.append(str(year))
        totals.append(float(pnl.sum()))

    colors = ["tab:green" if v >= 0 else "tab:red" for v in totals]
    plt.figure(figsize=(12, 6))
    plt.bar(years, totals, color=colors)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("TX AND TAIEX AND TPEx 開低走高 × 8% 逐年總損益")
    plt.xlabel("年份")
    plt.ylabel("總損益（元）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def _load_short_limit_multiple(api, stock_ids: list, start_date: str,
                               end_date: str) -> dict:
    """載入各股每日融券限額 ShortSaleLimit，帶 data/margin/ 快取。

    來源 TaiwanStockMarginPurchaseShortSale；ShortSaleLimit > 0 代表該股
    當日可融券（即可資券沖放空）。實際打 API 後 sleep 0.3 秒。

    Args:
        api: DataLoader 實例。
        stock_ids: 股票代號清單。
        start_date / end_date: 日期區間。

    Returns:
        dict[stock_id -> DataFrame（index=date, 欄 ShortSaleLimit）]。
    """
    out = {}
    margin_dir = os.path.join(DATA_DIR, "margin")
    total = len(stock_ids)
    for i, sid in enumerate(stock_ids, start=1):
        path = os.path.join(margin_dir, f"{sid}.csv")
        if os.path.exists(path):
            try:
                out[sid] = pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:  # noqa: BLE001
                out[sid] = pd.DataFrame(columns=["ShortSaleLimit"])
            continue
        try:
            raw = api.get_data(
                dataset="TaiwanStockMarginPurchaseShortSale", data_id=sid,
                start_date=start_date, end_date=end_date)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] {sid} 融券資料失敗：{exc}")
            raw = None
        if raw is None or raw.empty or "ShortSaleLimit" not in raw.columns:
            df = pd.DataFrame(columns=["ShortSaleLimit"])
            df.index.name = "date"
        else:
            df = raw[["date", "ShortSaleLimit"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["ShortSaleLimit"] = df["ShortSaleLimit"].astype(float)
            df = df[["ShortSaleLimit"]]
        os.makedirs(margin_dir, exist_ok=True)
        df.to_csv(path)
        out[sid] = df
        time.sleep(0.3)
        if i % 100 == 0:
            print(f"  融券資料載入 {i}/{total}")
    return out


def daytrade_class_analysis(regions_map: dict, all_gains: pd.DataFrame,
                            suspension_df: pd.DataFrame, data: dict, api,
                            out_path: str, mv_cap: float = 100e8,
                            turn_min: float = 0.005,
                            threshold: float = 0.08) -> None:
    """把固定濾網下的強勢股分成三種當沖限制類別，比較隔日放空績效。

    固定條件：三指數AND 開低走高、漲幅>=8%且<9.5%、市值<=100億、周轉率>=0.5%。
    三類（可重疊）：
      1. 可現股當沖賣：不在 DayTradingSuspension 暫停名單內。
      2. 暫停先賣後買：在暫停名單內（date <= 事件日 <= end_date）。
      3. 可資券沖：事件日 ShortSaleLimit > 0（可融券放空）。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 每日 region 序列]。
        all_gains: build_all_gains 輸出（未過濾）。
        suspension_df: 暫停當沖名單。
        data: 個股 OHLCV 字典。
        api: DataLoader（抓融券資料用）。
        out_path: equity curve 圖路徑。
        mv_cap / turn_min / threshold: 固定濾網參數。
    """
    # 1) 原始強勢股事件（未過濾暫停）× 三指數 AND 開低走高
    ev = strong_events(all_gains, threshold)
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}
    sig_dates = set.intersection(*[kdzg[m] for m in ["TX", "TAIEX", "TPEx"]])
    ev = ev[ev["date"].isin(sig_dates)].copy()

    # 2) 市值 + 周轉率濾網
    stock_ids = sorted(ev["stock_id"].unique())
    mv_dict = load_market_value_multiple(stock_ids, START_DATE, END_DATE)
    turnover_by_stock = {
        sid: calc_turnover(data[sid], mv_dict[sid])
        for sid in stock_ids if sid in mv_dict and sid in data
    }
    mv_vals, turn_vals = [], []
    for _, row in ev.iterrows():
        sid, d = row["stock_id"], row["date"]
        mv = mv_dict.get(sid)
        mv_vals.append(float(mv.loc[d, "market_value"])
                       if (mv is not None and d in mv.index) else np.nan)
        ts = turnover_by_stock.get(sid)
        turn_vals.append(float(ts.loc[d])
                         if (ts is not None and d in ts.index) else np.nan)
    ev["market_value"] = mv_vals
    ev["turnover"] = turn_vals
    ev = ev[(ev["market_value"] <= mv_cap) & (ev["turnover"] >= turn_min)].copy()
    ev = ev[["date", "stock_id"]].reset_index(drop=True)

    # 3) 暫停分類（用 filter_day_trading_eligible 判定哪些「未暫停」）
    eligible = filter_day_trading_eligible(ev, suspension_df)
    elig_keys = set(zip(pd.to_datetime(eligible["date"]),
                        eligible["stock_id"].astype(str)))
    ev["suspended"] = [(pd.Timestamp(d), str(s)) not in elig_keys
                       for d, s in zip(ev["date"], ev["stock_id"])]

    # 4) 融券（可資券沖）分類
    print(f"\n載入 {len(stock_ids)} 檔融券限額（data/margin/ 快取）...")
    short_dict = _load_short_limit_multiple(api, stock_ids, START_DATE, END_DATE)
    margin_ok = []
    for _, row in ev.iterrows():
        sid, d = row["stock_id"], row["date"]
        sl = short_dict.get(sid)
        if sl is not None and not sl.empty and d in sl.index:
            margin_ok.append(float(sl.loc[d, "ShortSaleLimit"]) > 0)
        else:
            margin_ok.append(False)
    ev["margin_ok"] = margin_ok

    # 5) 三類事件 → 回測（共用 build_event_trades，含停損+滑價）
    classes = [
        ("1.可現股當沖賣", ev[~ev["suspended"]]),
        ("2.暫停先賣後買", ev[ev["suspended"]]),
        ("3.可資券沖", ev[ev["margin_ok"]]),
    ]

    print("\n" + "=" * 76)
    print("當沖限制三分類：三指數AND 開低走高×8% × 市值≤100億 × 周轉率≥0.5%")
    print("（隔日放空，含停損8%+滑價0.15%）")
    print("=" * 76)
    print(f"{'分類':<16}| {'交易筆數':>6} | {'觸發天數':>6} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 86)

    plt.figure(figsize=(12, 6))
    for label, sub_ev in classes:
        trades = build_event_trades(sub_ev[["date", "stock_id"]], data)
        trades = trades.sort_values("date").reset_index(drop=True)
        s = compute_stats(trades)
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{label:<16}| {s['n_trades']:>6} | {s['n_days']:>6} | "
              f"{s['total_pnl']:>11,.0f} | {wr:>7} | {s['sharpe']:>8.4f} | "
              f"{s['max_drawdown']:>11,.0f}")
        if not trades.empty:
            equity = trades.set_index("date")["pnl"].cumsum()
            plt.plot(equity.index, equity.values, label=label)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("當沖限制三分類 → 放空強勢股 累積損益")
    plt.xlabel("date")
    plt.ylabel("累積損益（元）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def combo_filtered_yearly(regions_map: dict, events: pd.DataFrame, data: dict,
                          out_path: str, mv_cap: float = 100e8,
                          turn_min: float = 0.005) -> None:
    """三指數AND 開低走高×8% × 市值<=門檻 × 周轉率>=門檻：摘要 + 逐年表 + 圖。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 每日 region 序列]。
        events: 強勢股事件（8% 門檻、已過濾當沖）。
        data: 個股 OHLCV 字典。
        out_path: 逐年長條圖輸出路徑。
        mv_cap: 市值上限（元），預設 100 億。
        turn_min: 周轉率下限，預設 0.005（0.5%）。
    """
    all_trades = build_event_trades(events, data)
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}
    sig_dates = set.intersection(*[kdzg[m] for m in ["TX", "TAIEX", "TPEx"]])
    trades = all_trades[all_trades["event_date"].isin(sig_dates)].copy()

    stock_ids = sorted(trades["stock_id"].unique())
    mv_dict = load_market_value_multiple(stock_ids, START_DATE, END_DATE)
    turnover_by_stock = {
        sid: calc_turnover(data[sid], mv_dict[sid])
        for sid in stock_ids if sid in mv_dict and sid in data
    }
    mv_vals, turn_vals = [], []
    for _, row in trades.iterrows():
        sid = row["stock_id"]
        d = row["event_date"]
        mv = mv_dict.get(sid)
        mv_vals.append(float(mv.loc[d, "market_value"])
                       if (mv is not None and d in mv.index) else np.nan)
        ts = turnover_by_stock.get(sid)
        turn_vals.append(float(ts.loc[d])
                         if (ts is not None and d in ts.index) else np.nan)
    trades["market_value"] = mv_vals
    trades["turnover"] = turn_vals

    filt = trades[(trades["market_value"] <= mv_cap)
                  & (trades["turnover"] >= turn_min)]
    filt = filt.sort_values("date").reset_index(drop=True)

    label = (f"三指數AND 開低走高×8% × 市值<={mv_cap / 1e8:.0f}億 "
             f"× 周轉率>={turn_min:.1%}")
    s = compute_stats(filt)

    print("\n" + "=" * 64)
    print(f"{label}（含停損8%+滑價0.15%）")
    print("=" * 64)
    print(f"  交易筆數 : {s['n_trades']}")
    print(f"  觸發天數 : {s['n_days']}")
    print(f"  總損益   : {s['total_pnl']:,.0f}")
    print(f"  勝率     : {s['win_rate']:.2%}" if s["n_trades"] else "  勝率     : -")
    print(f"  Sharpe   : {s['sharpe']:.4f}")
    print(f"  最大回撤 : {s['max_drawdown']:,.0f}")

    if filt.empty:
        print("（無交易，略過逐年）")
        return

    filt = filt.copy()
    filt["year"] = pd.to_datetime(filt["event_date"]).dt.year
    print("\n逐年拆解：")
    print(f"{'年份':<6}| {'觸發天數':>6} | {'交易筆數':>6} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>9}")
    print("-" * 62)
    years, totals = [], []
    for year, g in filt.groupby("year"):
        pnl = g["pnl"]
        std = pnl.std()
        sharpe = float(pnl.mean() / std * math.sqrt(252)) if std and std > 0 else 0.0
        print(f"{year:<6}| {g['event_date'].nunique():>6} | {len(pnl):>6} | "
              f"{pnl.sum():>11,.0f} | {(pnl > 0).mean():>6.2%} | {sharpe:>9.4f}")
        years.append(str(year))
        totals.append(float(pnl.sum()))

    colors = ["tab:green" if v >= 0 else "tab:red" for v in totals]
    plt.figure(figsize=(12, 6))
    plt.bar(years, totals, color=colors)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"{label} 逐年總損益")
    plt.xlabel("年份")
    plt.ylabel("總損益（元）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def plot_dual_axis(labels: list, sharpes: list, totals: list,
                   title: str, out_path: str) -> None:
    """雙 Y 軸長條圖：左軸 Sharpe、右軸 總損益。

    Args:
        labels: X 軸標籤（門檻）。
        sharpes: 各門檻 Sharpe。
        totals: 各門檻 總損益。
        title: 圖標題。
        out_path: 圖檔輸出路徑。
    """
    x = np.arange(len(labels))
    width = 0.4
    fig, ax1 = plt.subplots(figsize=(13, 7))
    b1 = ax1.bar(x - width / 2, sharpes, width, color="tab:blue")
    ax1.set_ylabel("Sharpe（年化）", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, totals, width, color="tab:orange")
    ax2.set_ylabel("總損益（元）", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title(title)
    ax1.legend([b1, b2], ["Sharpe", "總損益"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] 已存 {out_path}")


def size_turnover_grids(regions_map: dict, events: pd.DataFrame, data: dict,
                        out_size: str, out_turnover: str) -> None:
    """對「三指數 AND 開低走高 × 8%」做市值 grid 與周轉率 grid。

    Grid 1（市值）：只放空當日市值 <= 門檻的股票。
    Grid 2（周轉率）：只放空當日周轉率 >= 門檻的股票。
    市值／周轉率皆取訊號當日（event_date，放空前已知）數值。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 每日 region 序列]。
        events: 強勢股事件（8% 門檻、已過濾當沖）。
        data: 個股 OHLCV 字典。
        out_size: 市值 grid 圖路徑。
        out_turnover: 周轉率 grid 圖路徑。
    """
    all_trades = build_event_trades(events, data)
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}
    sig_dates = set.intersection(*[kdzg[m] for m in ["TX", "TAIEX", "TPEx"]])
    trades = all_trades[all_trades["event_date"].isin(sig_dates)].copy()

    # 載入涉及個股的市值（cache-only 不適用，市值另有 data/mv/ 快取）
    stock_ids = sorted(trades["stock_id"].unique())
    print(f"\n三指數 AND 組合涉及 {len(stock_ids)} 檔個股，載入市值（data/mv/ 快取）...")
    mv_dict = load_market_value_multiple(stock_ids, START_DATE, END_DATE)

    # 各股周轉率序列
    turnover_by_stock = {
        sid: calc_turnover(data[sid], mv_dict[sid])
        for sid in stock_ids if sid in mv_dict and sid in data
    }

    # 對每筆交易附上訊號當日的市值與周轉率
    mv_vals, turn_vals = [], []
    for _, row in trades.iterrows():
        sid = row["stock_id"]
        d = row["event_date"]
        mv = mv_dict.get(sid)
        mv_vals.append(float(mv.loc[d, "market_value"])
                       if (mv is not None and d in mv.index) else np.nan)
        ts = turnover_by_stock.get(sid)
        turn_vals.append(float(ts.loc[d])
                         if (ts is not None and d in ts.index) else np.nan)
    trades["market_value"] = mv_vals
    trades["turnover"] = turn_vals

    # ---- Grid 1：市值門檻（億元；放空市值 <= 門檻者）----
    size_grid = [(5, "5億"), (10, "10億"), (20, "20億"), (30, "30億"),
                 (50, "50億"), (100, "100億"), (200, "200億"), (None, "不限")]
    print("\n" + "=" * 76)
    print("市值 grid：三指數AND 開低走高×8%（只放空市值 <= 門檻；含停損8%+滑價0.15%）")
    print("=" * 76)
    print(f"{'門檻':>6} | {'交易筆數':>6} | {'總損益':>11} | {'勝率':>7} | "
          f"{'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 66)
    s_labels, s_sharpe, s_total = [], [], []
    for val, label in size_grid:
        sub = trades if val is None else trades[trades["market_value"] <= val * 1e8]
        s = compute_stats(sub.sort_values("date"))
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{label:>6} | {s['n_trades']:>6} | {s['total_pnl']:>11,.0f} | "
              f"{wr:>7} | {s['sharpe']:>8.4f} | {s['max_drawdown']:>11,.0f}")
        s_labels.append(label)
        s_sharpe.append(s["sharpe"])
        s_total.append(s["total_pnl"])
    plot_dual_axis(s_labels, s_sharpe, s_total,
                   "市值門檻 grid（三指數AND 開低走高×8%）", out_size)

    # ---- Grid 2：周轉率門檻（放空周轉率 >= 門檻者）----
    turn_grid = [(0.005, "0.5%"), (0.01, "1%"), (0.015, "1.5%"),
                 (0.02, "2%"), (0.03, "3%"), (0.05, "5%"), (None, "不限")]
    print("\n" + "=" * 76)
    print("周轉率 grid：三指數AND 開低走高×8%（只放空周轉率 >= 門檻；含停損8%+滑價0.15%）")
    print("=" * 76)
    print(f"{'門檻':>6} | {'交易筆數':>6} | {'總損益':>11} | {'勝率':>7} | "
          f"{'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 66)
    t_labels, t_sharpe, t_total = [], [], []
    for val, label in turn_grid:
        sub = trades if val is None else trades[trades["turnover"] >= val]
        s = compute_stats(sub.sort_values("date"))
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{label:>6} | {s['n_trades']:>6} | {s['total_pnl']:>11,.0f} | "
              f"{wr:>7} | {s['sharpe']:>8.4f} | {s['max_drawdown']:>11,.0f}")
        t_labels.append(label)
        t_sharpe.append(s["sharpe"])
        t_total.append(s["total_pnl"])
    plot_dual_axis(t_labels, t_sharpe, t_total,
                   "周轉率門檻 grid（三指數AND 開低走高×8%）", out_turnover)


def _short_trade(bar: pd.Series, stop_loss_pct) -> tuple:
    """做空單筆損益 + 是否觸發停損（停損值可變；None 表示不設停損）。

    Args:
        bar: 個股隔日 OHLC（含 open/high/close）。
        stop_loss_pct: 停損百分比；None 表示不設停損。

    Returns:
        (pnl, stopped) tuple；open/close 非正時回 None。
    """
    open_ = float(bar["open"])
    high = float(bar["high"])
    close = float(bar["close"])
    if open_ <= 0 or close <= 0:
        return None

    entry = open_ * (1 - SLIPPAGE_PCT)
    stopped = False
    if stop_loss_pct is not None:
        stop_price = entry * (1 + stop_loss_pct)
        if high >= stop_price:
            exit_price = stop_price
            stopped = True
        else:
            exit_price = close * (1 + SLIPPAGE_PCT)
    else:
        exit_price = close * (1 + SLIPPAGE_PCT)

    gross = (entry - exit_price) * SHARES
    cost = ((entry + exit_price) * SHARES * FEE_RATE
            + entry * SHARES * EFFECTIVE_TAX_RATE)
    return float(gross - cost), stopped


def _build_trades_stop(events: pd.DataFrame, data: dict,
                       stop_loss_pct) -> pd.DataFrame:
    """以指定停損值對事件集做隔日放空，回傳含 stopped 旗標的交易。

    Args:
        events: 事件 (date, stock_id)。
        data: 個股 OHLCV 字典。
        stop_loss_pct: 停損百分比；None 表示不設停損。

    Returns:
        DataFrame（event_date, date, pnl, stopped）。
    """
    records = []
    for _, row in events.iterrows():
        event_date = pd.Timestamp(row["date"])
        stock_id = str(row["stock_id"])
        df = data.get(stock_id)
        if df is None:
            continue
        pos = df.index.searchsorted(event_date, side="right")
        if pos >= len(df.index):
            continue
        next_day = df.index[pos]
        res = _short_trade(df.loc[next_day], stop_loss_pct)
        if res is None:
            continue
        pnl, stopped = res
        records.append({"event_date": event_date, "date": next_day,
                        "pnl": pnl, "stopped": stopped})
    return pd.DataFrame(records,
                        columns=["event_date", "date", "pnl", "stopped"])


def plot_dual_axis_line(labels: list, sharpes: list, totals: list,
                        title: str, out_path: str) -> None:
    """雙 Y 軸折線圖：左軸 Sharpe、右軸 總損益。

    Args:
        labels: X 軸標籤。
        sharpes / totals: 各點 Sharpe / 總損益。
        title: 標題。
        out_path: 圖檔路徑。
    """
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(13, 7))
    l1, = ax1.plot(x, sharpes, "o-", color="tab:blue", label="Sharpe")
    ax1.set_ylabel("Sharpe（年化）", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, totals, "s-", color="tab:orange", label="總損益")
    ax2.set_ylabel("總損益（元）", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(title)
    ax1.legend([l1, l2], ["Sharpe", "總損益"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] 已存 {out_path}")


def stoploss_grid(regions_map: dict, events: pd.DataFrame, data: dict,
                  out_path: str) -> None:
    """對「三指數AND 開低走高×8% × 市值≤100億 × 周轉率≥0.5%」做停損 grid。

    停損值 1%~10% 與「不設停損」，比較績效與停損觸發次數。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 每日 region 序列]。
        events: 強勢股事件（8% 門檻、已過濾當沖）。
        data: 個股 OHLCV 字典。
        out_path: 雙軸折線圖路徑。
    """
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}
    sig_dates = set.intersection(*[kdzg[m] for m in ["TX", "TAIEX", "TPEx"]])
    ev = events[events["date"].isin(sig_dates)].copy()

    sids = sorted(ev["stock_id"].unique())
    mv = load_market_value_multiple(sids, START_DATE, END_DATE)
    turn = {sid: calc_turnover(data[sid], mv[sid])
            for sid in sids if sid in mv and sid in data}
    mv_vals, turn_vals = [], []
    for _, row in ev.iterrows():
        sid, d = row["stock_id"], row["date"]
        m = mv.get(sid)
        mv_vals.append(float(m.loc[d, "market_value"])
                       if (m is not None and d in m.index) else np.nan)
        ts = turn.get(sid)
        turn_vals.append(float(ts.loc[d])
                         if (ts is not None and d in ts.index) else np.nan)
    ev["market_value"] = mv_vals
    ev["turnover"] = turn_vals
    ev = ev[(ev["market_value"] <= 100e8) & (ev["turnover"] >= 0.005)]
    ev = ev[["date", "stock_id"]].reset_index(drop=True)

    stops = [(v / 100, f"{v}%") for v in range(1, 11)] + [(None, "不設停損")]

    print("\n" + "=" * 84)
    print("停損 grid：三指數AND 開低走高×8% × 市值≤100億 × 周轉率≥0.5%（滑價0.15%）")
    print("=" * 84)
    print(f"{'停損':>6} | {'交易筆數':>6} | {'總損益':>11} | {'勝率':>7} | "
          f"{'Sharpe':>8} | {'最大回撤':>11} | {'停損觸發':>6}")
    print("-" * 84)

    labels, sharpes, totals = [], [], []
    for stop_val, label in stops:
        trades = _build_trades_stop(ev, data, stop_val).sort_values("date")
        s = compute_stats(trades)
        n_stop = int(trades["stopped"].sum()) if not trades.empty else 0
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{label:>6} | {s['n_trades']:>6} | {s['total_pnl']:>11,.0f} | "
              f"{wr:>7} | {s['sharpe']:>8.4f} | {s['max_drawdown']:>11,.0f} | "
              f"{n_stop:>6}")
        labels.append(label)
        sharpes.append(s["sharpe"])
        totals.append(s["total_pnl"])

    plot_dual_axis_line(labels, sharpes, totals,
                        "停損 grid（三指數AND 開低走高×8%×小型×高周轉）", out_path)


def threshold_grid_with_filters(regions_map: dict, all_gains: pd.DataFrame,
                                suspension_df: pd.DataFrame, data: dict,
                                out_path: str) -> None:
    """強勢股漲幅門檻 5%~10% grid（含市值≤100億 + 周轉率≥0.5% 濾網）。

    固定：三指數AND 開低走高、可現股當沖賣、停損 8%、滑價 0.15%。
    對每個門檻：取漲幅 [門檻, 9.5%] 的強勢股 → 套市值/周轉率濾網 → 隔日放空回測。

    Args:
        regions_map: dict['TX'/'TAIEX'/'TPEx' -> 每日 region 序列]。
        all_gains: build_all_gains 輸出（未過濾）。
        suspension_df: 暫停當沖名單。
        data: 個股 OHLCV 字典。
        out_path: 雙軸折線圖路徑。
    """
    kdzg = {name: set(r[r == "開低走高"].index)
            for name, r in regions_map.items()}
    sig_dates = set.intersection(*[kdzg[m] for m in SIGNAL_COMBO])

    thresholds = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    # 各門檻事件（三指數AND 開低走高、可當沖），並蒐集所有涉及個股
    events_by_thr = {}
    all_sids = set()
    for thr in thresholds:
        ev = filter_day_trading_eligible(strong_events(all_gains, thr),
                                         suspension_df)
        ev = ev[ev["date"].isin(sig_dates)].copy()
        events_by_thr[thr] = ev
        all_sids.update(ev["stock_id"].unique())

    # 一次載入所有涉及個股的市值，建周轉率序列
    sids = sorted(all_sids)
    mv = load_market_value_multiple(sids, START_DATE, END_DATE)
    turn = {sid: calc_turnover(data[sid], mv[sid])
            for sid in sids if sid in mv and sid in data}

    def _apply_filters(ev: pd.DataFrame) -> pd.DataFrame:
        if ev.empty:
            return ev[["date", "stock_id"]]
        mv_vals, turn_vals = [], []
        for _, row in ev.iterrows():
            sid, d = row["stock_id"], row["date"]
            m = mv.get(sid)
            mv_vals.append(float(m.loc[d, "market_value"])
                           if (m is not None and d in m.index) else np.nan)
            ts = turn.get(sid)
            turn_vals.append(float(ts.loc[d])
                             if (ts is not None and d in ts.index) else np.nan)
        ev = ev.copy()
        ev["market_value"] = mv_vals
        ev["turnover"] = turn_vals
        ev = ev[(ev["market_value"] <= MARKET_VALUE_THRESHOLD)
                & (ev["turnover"] >= TURNOVER_THRESHOLD)]
        return ev[["date", "stock_id"]]

    print("\n" + "=" * 78)
    print("強勢股門檻 grid（三指數AND 開低走高 × 市值≤100億 × 周轉率≥0.5%，停損8%+滑價0.15%）")
    print("=" * 78)
    print(f"{'門檻':>6} | {'交易筆數':>6} | {'總損益':>11} | {'勝率':>7} | "
          f"{'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 66)

    labels, sharpes, totals = [], [], []
    for thr in thresholds:
        fev = _apply_filters(events_by_thr[thr])
        trades = build_event_trades(fev, data).sort_values("date")
        s = compute_stats(trades)
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        print(f"{thr:>6.0%} | {s['n_trades']:>6} | {s['total_pnl']:>11,.0f} | "
              f"{wr:>7} | {s['sharpe']:>8.4f} | {s['max_drawdown']:>11,.0f}")
        labels.append(f"{thr:.0%}")
        sharpes.append(s["sharpe"])
        totals.append(s["total_pnl"])

    plot_dual_axis_line(labels, sharpes, totals,
                        "強勢股門檻 grid（三指數AND 開低走高×小型×高周轉）", out_path)


def main() -> None:
    """主流程：建立各門檻強勢股事件，逐一指數回測並輸出摘要與圖。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api = get_loader()

    data = load_stock_data_cache_only()
    print(f"個股 cache-only 載入：{len(data)} 檔")

    all_gains = build_all_gains(data)
    suspension_df = get_day_trading_suspension(START_DATE, END_DATE)

    # 各門檻的強勢股事件（含當沖可賣過濾）+ 樣本量
    print("\n各門檻強勢股樣本量（漲幅 >= 門檻 且 <= 9.5%，已過濾暫停當沖）：")
    events_by_thr = {}
    for threshold in THRESHOLDS:
        ev = strong_events(all_gains, threshold)
        ev = filter_day_trading_eligible(ev, suspension_df)
        events_by_thr[threshold] = ev
        per_day = ev.groupby("date").size() if not ev.empty else pd.Series(dtype=int)
        mean_per_day = float(per_day.mean()) if len(per_day) else 0.0
        print(f"  門檻 {threshold:.0%}：事件 {len(ev)} 筆，"
              f"有強勢股 {len(per_day)} 天，平均每日 {mean_per_day:.1f} 檔")

    # 指數資料
    tx_df = get_tx_data(START_DATE, END_DATE)
    print(f"\n台指期 TX：{len(tx_df)} 個交易日")
    taiex_df = get_index_ohlcv(api, "TAIEX", START_DATE, END_DATE)
    print(f"加權指數 TAIEX：{len(taiex_df)} 個交易日")
    tpex_df = get_index_ohlcv(api, "TPEx", START_DATE, END_DATE)
    print(f"櫃買指數 TPEx：{len(tpex_df)} 個交易日")

    run_index_thresholds("台指期 TX", tx_df, events_by_thr, data,
                         os.path.join(OUTPUT_DIR, "tx_strong_sharpe.png"))
    run_index_thresholds("加權指數 TAIEX", taiex_df, events_by_thr, data,
                         os.path.join(OUTPUT_DIR, "taiex_strong_sharpe.png"))
    run_index_thresholds("櫃買指數 TPEx", tpex_df, events_by_thr, data,
                         os.path.join(OUTPUT_DIR, "tpex_strong_sharpe.png"))

    # 「開低走高 × 8% 門檻」逐年拆解（三個指數各一）
    events_8 = events_by_thr[0.08]
    for index_name, index_df, png in [
        ("台指期 TX", tx_df, "tx_kdzg_8_yearly.png"),
        ("加權指數 TAIEX", taiex_df, "taiex_kdzg_8_yearly.png"),
        ("櫃買指數 TPEx", tpex_df, "tpex_kdzg_8_yearly.png"),
    ]:
        regions = classify_regions(index_df)
        trades = backtest_short_events(regions, events_8, data)["開低走高"]
        yearly_breakdown(f"{index_name} 開低走高 × 8%", trades,
                         os.path.join(OUTPUT_DIR, png))

    # 開低走高 訊號組合分析（三指數的單一/AND 交集，8% 強勢股）
    regions_map = {
        "TX": classify_regions(tx_df),
        "TAIEX": classify_regions(taiex_df),
        "TPEx": classify_regions(tpex_df),
    }
    signal_combo_analysis(regions_map, events_8, data,
                          os.path.join(OUTPUT_DIR, "signal_combo.png"))

    # 三指數 AND 組合的逐年拆解
    combo_yearly_breakdown(regions_map, events_8, data,
                           os.path.join(OUTPUT_DIR, "combo_yearly.png"))

    # 市值 grid + 周轉率 grid（三指數 AND 開低走高 × 8%）
    size_turnover_grids(regions_map, events_8, data,
                        os.path.join(OUTPUT_DIR, "size_grid.png"),
                        os.path.join(OUTPUT_DIR, "turnover_grid.png"))

    # 疊加濾網：市值<=100億 × 周轉率>=0.5% 的逐年拆解
    combo_filtered_yearly(regions_map, events_8, data,
                          os.path.join(OUTPUT_DIR, "combo_filter_yearly.png"))

    # 當沖限制三分類分析（可現股當沖賣 / 暫停先賣後買 / 可資券沖）
    daytrade_class_analysis(regions_map, all_gains, suspension_df, data, api,
                            os.path.join(OUTPUT_DIR, "daytrade_class.png"))

    # 停損 grid（1%~10% 與不設停損）
    stoploss_grid(regions_map, events_8, data,
                  os.path.join(OUTPUT_DIR, "stoploss_grid.png"))


if __name__ == "__main__":
    main()
