"""「開低走低」region 的 in-sample / out-of-sample 切分驗證。

資料區間：
  in-sample      2021-01-01 ~ 2023-12-31
  out-of-sample  2024-01-01 ~ 2026-06-01

個股日線資料直接從既有 data/*.csv 快取還原股票池並透過 load_multiple 載入；
TX、漲跌停價與暫停當沖名單沿用既有 data_loader 函式。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import data_loader
from data_loader import (
    get_day_trading_suspension,
    get_price_limit_data,
    get_tx_data,
)
from market_scanner import filter_day_trading_eligible, find_limit_up_events
from portfolio_backtest import PortfolioBacktester
from strategy import build_region_signals, classify_tx_regions


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

START_DATE = "2015-01-01"
SPLIT_DATE = "2024-01-01"
END_DATE = "2026-06-01"
TARGET_REGION = "開低走低"


def cached_stock_ids() -> list:
    """從既有 data/*.csv 快取還原股票池，不重新查詢全市場清單 API。"""
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


def load_cached_stock_data(stock_id: str,
                           start_date: str,
                           end_date: str) -> pd.DataFrame:
    """只從既有個股 CSV 快取讀資料，不呼叫 API 補抓。"""
    path = DATA_DIR / f"{stock_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 CSV 快取：{path}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cached = pd.read_csv(path, index_col=0, parse_dates=True)
    if cached.empty:
        raise ValueError(f"CSV 快取為空：{path}")

    df = data_loader._to_standard_ohlcv(cached)  # noqa: SLF001
    return df.loc[start:end]


def load_multiple_from_existing_cache(stock_ids: list,
                                      start_date: str,
                                      end_date: str) -> dict:
    """透過既有 load_multiple 流程載入，但將單檔讀取限制為 CSV cache-only。"""
    original_load_data = data_loader.load_data
    data_loader.load_data = load_cached_stock_data
    try:
        return data_loader.load_multiple(stock_ids, start_date, end_date)
    finally:
        data_loader.load_data = original_load_data


def assert_stock_cache_readable(stock_ids: list,
                                start_date: str,
                                end_date: str) -> None:
    """確認個股 CSV 快取可讀，避免後續載入時才發現檔案損壞。"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    bad_files = []
    for stock_id in stock_ids:
        path = DATA_DIR / f"{stock_id}.csv"
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception as exc:  # noqa: BLE001
            bad_files.append(f"{stock_id}（讀取失敗：{exc}）")
            continue

        if cached.empty:
            bad_files.append(f"{stock_id}（空檔案）")
            continue

        index = pd.to_datetime(cached.index)
        if index.max() < start or index.min() > end:
            bad_files.append(f"{stock_id}（{index.min().date()}~{index.max().date()}）")

    if bad_files:
        preview = "\n".join(f"  - {item}" for item in bad_files[:20])
        more = "" if len(bad_files) <= 20 else f"\n  ... 還有 {len(bad_files) - 20} 檔"
        raise RuntimeError(
            "個股 CSV 快取不可用，為避免重新下載已停止。\n"
            f"需要與範圍有交集：{start.date()}~{end.date()}\n"
            f"不可用檔案數：{len(bad_files)}\n{preview}{more}"
        )


def distinct_dates(events: pd.DataFrame) -> int:
    """計算事件表中的 distinct 日期數。"""
    if events.empty:
        return 0
    return int(pd.to_datetime(events["date"]).nunique())


def print_event_count(label: str, events: pd.DataFrame) -> None:
    """印出事件筆數與 distinct 日期數，方便逐步核對。"""
    print(f"{label}: 事件筆數={len(events)}，distinct日期數={distinct_dates(events)}")


def summarize_metrics(period: str,
                      events: pd.DataFrame,
                      metrics: dict | None) -> dict:
    """整理 IS/OOS 對比表的一列。"""
    if metrics is None:
        return {
            "期間": period,
            "事件筆數": len(events),
            "distinct日期數": distinct_dates(events),
            "總損益": np.nan,
            "勝率": np.nan,
            "Sharpe": np.nan,
            "最大回撤": np.nan,
        }

    return {
        "期間": period,
        "事件筆數": len(events),
        "distinct日期數": distinct_dates(events),
        "總損益": metrics["總損益"],
        "勝率": metrics["整體勝率"],
        "Sharpe": metrics["Sharpe Ratio"],
        "最大回撤": metrics["最大回撤"],
    }


def run_region_backtest(label: str,
                        events: pd.DataFrame,
                        tx_regions: pd.Series,
                        all_dates: pd.DatetimeIndex,
                        stock_ids: list,
                        data: dict) -> dict | None:
    """以預設 PortfolioBacktester 參數回測指定期間的目標 region 事件。"""
    signals = build_region_signals(
        events, tx_regions, TARGET_REGION, all_dates, stock_ids
    )
    n_short = int((signals == -1).sum().sum())
    print(f"{label}: 放空訊號數={n_short}")

    bt = PortfolioBacktester()
    trades = bt.run(data, signals)
    print(f"{label}: 成交筆數={len(trades)}")

    print(f"\n=== {label} ===")
    if trades.empty:
        print("（無交易，略過 daily_summary/report）")
        return None

    bt.daily_summary()
    return bt.report()


def main() -> None:
    """執行「開低走低」region 的 IS/OOS 切分驗證。"""
    os.chdir(SCRIPT_DIR)
    load_dotenv(SCRIPT_DIR / ".env")

    print("=" * 60)
    print(f"目標 region：{TARGET_REGION}")
    print(f"總資料區間：{START_DATE} ~ {END_DATE}")
    print(f"In-sample：{START_DATE} ~ 2023-12-31")
    print(f"Out-of-sample：{SPLIT_DATE} ~ {END_DATE}")
    print("=" * 60)

    stock_ids = cached_stock_ids()
    if not stock_ids:
        raise FileNotFoundError(f"找不到既有個股 CSV 快取：{DATA_DIR}")
    print(f"步驟 1：從既有 CSV 快取還原股票池 {len(stock_ids)} 檔")
    assert_stock_cache_readable(stock_ids, START_DATE, END_DATE)
    print("步驟 1：個股 CSV 快取可讀性確認完成（不補抓個股日線）")

    data = load_multiple_from_existing_cache(stock_ids, START_DATE, END_DATE)
    loaded_stock_ids = list(data.keys())
    print(f"步驟 1：load_multiple 成功載入 {len(data)} 檔日線資料")

    tx_df = get_tx_data(START_DATE, END_DATE)
    print(f"步驟 1：TX 近月交易日數 {len(tx_df)}")

    price_limit_df = get_price_limit_data(START_DATE, END_DATE)
    print(f"步驟 1：price_limit_df 筆數 {len(price_limit_df)}")

    suspension_df = get_day_trading_suspension(START_DATE, END_DATE)
    print(f"步驟 1：suspension_df 筆數 {len(suspension_df)}")

    limit_up_events = find_limit_up_events(price_limit_df, data)
    print_event_count("步驟 1：limit_up_events", limit_up_events)

    filtered_events = filter_day_trading_eligible(limit_up_events, suspension_df)
    print_event_count("步驟 1：filtered_events", filtered_events)

    tx_regions = classify_tx_regions(tx_df)
    print("\nTX 區間分布：")
    print(tx_regions.value_counts(dropna=False).to_string())

    events = filtered_events.copy()
    events["date"] = pd.to_datetime(events["date"])
    events["region"] = events["date"].map(tx_regions)
    region_events = (
        events[events["region"] == TARGET_REGION]
        [["date", "stock_id"]]
        .copy()
        .reset_index(drop=True)
    )
    print_event_count(f"\n步驟 2：{TARGET_REGION} filtered_events", region_events)

    is_events = (
        region_events[region_events["date"] < pd.Timestamp(SPLIT_DATE)]
        .copy()
        .reset_index(drop=True)
    )
    oos_events = (
        region_events[region_events["date"] >= pd.Timestamp(SPLIT_DATE)]
        .copy()
        .reset_index(drop=True)
    )
    print_event_count("步驟 2：is_events (< 2024-01-01)", is_events)
    print_event_count("步驟 2：oos_events (>= 2024-01-01)", oos_events)

    print("\n步驟 3：基本統計")
    print_event_count("in-sample", is_events)
    print_event_count("out-of-sample", oos_events)

    all_dates = pd.DatetimeIndex(tx_df.index)
    is_dates = all_dates[all_dates < pd.Timestamp(SPLIT_DATE)]
    oos_dates = all_dates[all_dates >= pd.Timestamp(SPLIT_DATE)]
    print(f"\n回測日期數：IS={len(is_dates)}，OOS={len(oos_dates)}")

    metrics_is = run_region_backtest(
        "In-Sample (2021-2023)",
        is_events,
        tx_regions,
        is_dates,
        loaded_stock_ids,
        data,
    )
    metrics_oos = run_region_backtest(
        "Out-of-Sample (2024-2026)",
        oos_events,
        tx_regions,
        oos_dates,
        loaded_stock_ids,
        data,
    )

    summary = pd.DataFrame([
        summarize_metrics("IS", is_events, metrics_is),
        summarize_metrics("OOS", oos_events, metrics_oos),
    ])
    display = summary.copy()
    display["總損益"] = display["總損益"].map(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )
    display["勝率"] = display["勝率"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2%}"
    )
    display["Sharpe"] = display["Sharpe"].map(
        lambda x: "" if pd.isna(x) else f"{x:.4f}"
    )
    display["最大回撤"] = display["最大回撤"].map(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )

    print("\n=== IS / OOS 對比摘要 ===")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
