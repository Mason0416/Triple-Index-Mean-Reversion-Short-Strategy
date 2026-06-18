"""最終策略完整回測。

策略：三指數（TX, TAIEX, TPEx）AND「開低走高」當日 → 隔日放空當日強勢股
  - 強勢股：當日漲幅 8%~9.5%（排除接近漲停）
  - 市值 ≤ 100 億、周轉率 ≥ 0.5%、可現股當沖賣
  - 進場：隔日開盤放空；出場：隔日收盤回補
  - 停損 8%、單邊滑價 0.15%、手續費六折+當沖稅二折、每筆 1000 股

輸出：摘要表、逐年拆解表、資金曲線（output/final_equity_curve.png）、
逐年總損益（output/final_yearly.png）。所有參數見 index_region_backtest.py 最上方常數。
"""

import math
import os

import pandas as pd
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import index_region_backtest as irb
from data_loader import (
    calc_turnover,
    get_day_trading_suspension,
    get_tx_data,
    load_market_value_multiple,
)
from market_scanner import filter_day_trading_eligible


plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang TC", "Heiti TC", "STHeiti", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"


def get_index_frames(api) -> dict:
    """取得 SIGNAL_COMBO 三個指數的日線（TX 用期貨、其餘用指數）。

    Args:
        api: DataLoader 實例。

    Returns:
        dict[name -> OHLCV DataFrame]。
    """
    frames = {"TX": get_tx_data(irb.START_DATE, irb.END_DATE)}
    for name in irb.SIGNAL_COMBO:
        if name == "TX":
            continue
        frames[name] = irb.get_index_ohlcv(api, name, irb.START_DATE, irb.END_DATE)
    return frames


def get_final_events(api, data: dict) -> pd.DataFrame:
    """重建最終策略的觸發事件 (date, stock_id)。

    條件：SIGNAL_COMBO 全部「開低走高」、強勢股漲幅 [MIN, MAX]、可現股當沖賣、
    市值 <= MARKET_VALUE_THRESHOLD、周轉率 >= TURNOVER_THRESHOLD。

    Args:
        api: DataLoader 實例。
        data: 個股 OHLCV 字典。

    Returns:
        DataFrame（date, stock_id），依日期排序。
    """
    gains = irb.build_all_gains(data)
    susp = get_day_trading_suspension(irb.START_DATE, irb.END_DATE)
    ev = filter_day_trading_eligible(
        irb.strong_events(gains, irb.STRONG_STOCK_MIN, irb.STRONG_STOCK_MAX),
        susp)

    frames = get_index_frames(api)
    kdzg = []
    for name in irb.SIGNAL_COMBO:
        r = irb.classify_regions(frames[name])
        kdzg.append(set(r[r == "開低走高"].index))
    sig_dates = set.intersection(*kdzg)
    ev = ev[ev["date"].isin(sig_dates)].copy()

    sids = sorted(ev["stock_id"].unique())
    mv = load_market_value_multiple(sids, irb.START_DATE, irb.END_DATE)
    turn = {sid: calc_turnover(data[sid], mv[sid])
            for sid in sids if sid in mv and sid in data}
    mv_vals, turn_vals = [], []
    for _, row in ev.iterrows():
        sid, d = row["stock_id"], row["date"]
        m = mv.get(sid)
        mv_vals.append(float(m.loc[d, "market_value"])
                       if (m is not None and d in m.index) else float("nan"))
        ts = turn.get(sid)
        turn_vals.append(float(ts.loc[d])
                         if (ts is not None and d in ts.index) else float("nan"))
    ev["market_value"] = mv_vals
    ev["turnover"] = turn_vals
    ev = ev[(ev["market_value"] <= irb.MARKET_VALUE_THRESHOLD)
            & (ev["turnover"] >= irb.TURNOVER_THRESHOLD)]
    return ev[["date", "stock_id"]].sort_values(
        ["date", "stock_id"]).reset_index(drop=True)


def print_summary(stats: dict) -> None:
    """印出最終策略摘要表。"""
    print("\n" + "=" * 50)
    print("最終策略摘要（2015-01-01 ~ 2026-06-01）")
    print("=" * 50)
    print(f"  交易筆數 : {stats['n_trades']}")
    print(f"  觸發天數 : {stats['n_days']}")
    print(f"  總損益   : {stats['total_pnl']:,.0f}")
    print(f"  勝率     : {stats['win_rate']:.2%}")
    print(f"  Sharpe   : {stats['sharpe']:.4f}")
    print(f"  最大回撤 : {stats['max_drawdown']:,.0f}")


def print_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    """印出逐年拆解表並回傳逐年彙總（供繪圖）。

    Args:
        trades: 含 event_date / pnl 的交易明細。

    Returns:
        DataFrame（year, total）。
    """
    t = trades.copy()
    t["year"] = pd.to_datetime(t["event_date"]).dt.year
    print("\n逐年拆解：")
    print(f"{'年份':<6}| {'觸發天數':>6} | {'交易筆數':>6} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>9}")
    print("-" * 62)
    rows = []
    for year, g in t.groupby("year"):
        pnl = g["pnl"]
        std = pnl.std()
        sharpe = float(pnl.mean() / std * math.sqrt(252)) if std and std > 0 else 0.0
        print(f"{year:<6}| {g['event_date'].nunique():>6} | {len(pnl):>6} | "
              f"{pnl.sum():>11,.0f} | {(pnl > 0).mean():>6.2%} | {sharpe:>9.4f}")
        rows.append({"year": str(year), "total": float(pnl.sum())})
    return pd.DataFrame(rows)


def plot_equity(trades: pd.DataFrame, out_path: str) -> None:
    """畫最終策略資金曲線。"""
    equity = trades.set_index("date")["pnl"].cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity.values, color="tab:blue")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("最終策略 資金曲線（三指數AND 開低走高 × 強勢股隔日放空）")
    plt.xlabel("date")
    plt.ylabel("累積損益（元）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def plot_yearly(yearly: pd.DataFrame, out_path: str) -> None:
    """畫最終策略逐年總損益長條圖（正綠負紅）。"""
    colors = ["tab:green" if v >= 0 else "tab:red" for v in yearly["total"]]
    plt.figure(figsize=(12, 6))
    plt.bar(yearly["year"], yearly["total"], color=colors)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("最終策略 逐年總損益")
    plt.xlabel("年份")
    plt.ylabel("總損益（元）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out_path}")


def main() -> None:
    """執行最終策略完整流程並輸出摘要、逐年表與圖。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_dotenv()
    api = irb.get_loader()

    data = irb.load_stock_data_cache_only()
    print(f"個股 cache-only 載入：{len(data)} 檔")

    events = get_final_events(api, data)
    print(f"最終策略觸發事件：{len(events)} 筆")

    trades = irb.build_event_trades(events, data)
    trades = trades.sort_values("date").reset_index(drop=True)
    stats = irb.compute_stats(trades)

    print_summary(stats)
    yearly = print_yearly(trades)

    plot_equity(trades, os.path.join(OUTPUT_DIR, "final_equity_curve.png"))
    plot_yearly(yearly, os.path.join(OUTPUT_DIR, "final_yearly.png"))


if __name__ == "__main__":
    main()
