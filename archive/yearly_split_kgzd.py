"""開高走低 region：三種漲停事件定義的逐年績效拆解。

對「開高走低」region，分別用三種漲停事件定義建立隔日放空訊號、回測、
依年份彙總（年份｜筆數｜總損益｜勝率），三張表分開印：

  A. 全部（close >= limit_up * 0.99）
  B. 鎖死漲停（close == limit_up，容差 0.005，與原始定義一致）
  C. 逼近但未鎖死（A 但不屬於 B）

A = B ⊎ C，方便看清楚是哪一類貢獻了獲利。
股票資料一律 cache-only（不重抓 API）。
"""

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import (
    get_day_trading_suspension,
    get_price_limit_data,
    get_tx_data,
)
from market_scanner import filter_day_trading_eligible
from strategy import build_region_signals, classify_tx_regions
from portfolio_backtest import PortfolioBacktester
from diagnose_region import _cached_stock_ids, _load_cached_only, START_DATE, END_DATE


REGION = "開高走低"


def build_merged_close_limit(price_limit_df: pd.DataFrame,
                             data: dict) -> pd.DataFrame:
    """把漲停價與各股實際收盤價 merge 成同一張表。

    Returns:
        DataFrame，含 date, stock_id, close, limit_up。
    """
    pl = price_limit_df[["date", "stock_id", "limit_up"]].copy()
    pl["date"] = pd.to_datetime(pl["date"])
    pl["stock_id"] = pl["stock_id"].astype(str)
    pl["limit_up"] = pl["limit_up"].astype(float)
    pl = pl[pl["limit_up"] > 0]

    frames = []
    for stock_id, df in data.items():
        if "close" not in df.columns or df.empty:
            continue
        frames.append(pd.DataFrame({
            "date": df.index,
            "stock_id": str(stock_id),
            "close": df["close"].astype(float).values,
        }))
    close_long = pd.concat(frames, ignore_index=True)
    return pl.merge(close_long, on=["date", "stock_id"], how="inner")


def events_from_mask(merged: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """從 merged 表依布林遮罩取出 (date, stock_id) 事件。"""
    ev = merged.loc[mask, ["date", "stock_id"]].copy()
    return ev.drop_duplicates().sort_values("date").reset_index(drop=True)


def print_yearly_for_events(title: str, events: pd.DataFrame,
                            tx_regions: pd.Series, tx_df: pd.DataFrame,
                            data: dict, suspension_df: pd.DataFrame) -> None:
    """對給定事件集建立開高走低訊號、回測，並印逐年表。"""
    events = filter_day_trading_eligible(events, suspension_df)
    signals = build_region_signals(events, tx_regions, REGION,
                                   tx_df.index, list(data.keys()))
    bt = PortfolioBacktester()
    trades = bt.run(data, signals)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if trades.empty:
        print("（無交易）")
        return

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["date"]).dt.year
    grouped = trades.groupby("year")
    n_by_year = grouped["pnl"].size()
    pnl_by_year = grouped["pnl"].sum()
    win_by_year = grouped["pnl"].apply(lambda s: (s > 0).mean())

    print(f"{'年份':<6}| {'筆數':>5} | {'總損益':>14} | {'勝率':>7}")
    print("-" * 42)
    for year in range(pd.Timestamp(START_DATE).year,
                       pd.Timestamp(END_DATE).year + 1):
        if year in n_by_year.index:
            n = int(n_by_year.loc[year])
            total = float(pnl_by_year.loc[year])
            wr = f"{float(win_by_year.loc[year]):.2%}"
        else:
            n, total, wr = 0, 0.0, "-"
        print(f"{year:<6}| {n:>5} | {total:>14,.2f} | {wr:>7}")
    print("-" * 42)
    print(f"{'合計':<6}| {len(trades):>5} | {trades['pnl'].sum():>14,.2f} | "
          f"{(trades['pnl'] > 0).mean():>7.2%}")


def main() -> None:
    """重建狀態（cache-only）並印出三張逐年表。"""
    load_dotenv()

    stock_ids = _cached_stock_ids()
    print(f"[重建] cache-only 載入 {len(stock_ids)} 檔")
    data = _load_cached_only(stock_ids, START_DATE, END_DATE)
    print(f"[重建] 成功載入 {len(data)} 檔")

    tx_df = get_tx_data(START_DATE, END_DATE)
    tx_regions = classify_tx_regions(tx_df)
    price_limit_df = get_price_limit_data(START_DATE, END_DATE)
    suspension_df = get_day_trading_suspension(START_DATE, END_DATE)

    merged = build_merged_close_limit(price_limit_df, data)
    cl = merged["close"]
    lu = merged["limit_up"]

    mask_all = cl >= lu * 0.99
    mask_locked = np.isclose(cl, lu, atol=0.005)
    mask_near = mask_all & ~mask_locked  # 達 99% 但未鎖死

    ev_all = events_from_mask(merged, mask_all)
    ev_locked = events_from_mask(merged, mask_locked)
    ev_near = events_from_mask(merged, mask_near)

    print_yearly_for_events(
        "A. 開高走低｜全部（close >= limit_up * 0.99）",
        ev_all, tx_regions, tx_df, data, suspension_df)
    print_yearly_for_events(
        "B. 開高走低｜鎖死漲停（close == limit_up）",
        ev_locked, tx_regions, tx_df, data, suspension_df)
    print_yearly_for_events(
        "C. 開高走低｜逼近但未鎖死（>=99% 且未鎖死）",
        ev_near, tx_regions, tx_df, data, suspension_df)


if __name__ == "__main__":
    main()
