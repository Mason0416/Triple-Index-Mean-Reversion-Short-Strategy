"""主程式：TX 區間 × 全市場漲停股放空 策略完整流程（全市場版）。

流程：
  步驟 0  資料可用性檢查（小範圍探測 PriceLimit / DayTradingSuspension）
  1       取得全市場清單（全部約 2600 檔，不抽樣）
  2~3     設定區間、依序抓全部資料、建立漲停事件並過濾
  4       印出驗證資訊（過濾前後筆數、5 筆人工核對、各 region 事件數）
  5       依四種 TX 區間各自建立隔日放空訊號並回測

資料區間：2021-01-01 ~ 2026-06-01。
"""

import sys

import pandas as pd
from dotenv import load_dotenv

from data_loader import (
    get_all_stock_ids,
    get_day_trading_suspension,
    get_price_limit_data,
    get_tx_data,
    load_multiple,
)
from market_scanner import filter_day_trading_eligible, find_limit_up_events
from strategy import build_region_signals, classify_tx_regions
from portfolio_backtest import PortfolioBacktester


START_DATE = "2015-01-01"
END_DATE = "2026-06-01"

# 驗證資訊抽 5 筆人工核對時的固定種子（可重現）
SEED = 42

REGIONS = ["大跳空上漲", "小跳空上漲", "小跳空下跌", "大跳空下跌"]

# 步驟 6：盤中觸及漲停（high）事件回測的停損百分比
INTRADAY_STOP_LOSS_PCT = 0.09


def check_data_availability() -> bool:
    """步驟 0：以小範圍探測兩個關鍵資料集是否有資料並印出欄位。

    用 2021-01-01 ~ 2021-01-31 探測 TaiwanStockPriceLimit 與
    TaiwanStockDayTradingSuspension，印出 columns 與前 5 筆。
    若任一資料集無資料則回傳 False（呼叫端應停止並等待調整 start_date）。

    Returns:
        兩個資料集都有資料時回傳 True，否則 False。
    """
    print("=" * 60)
    print("步驟 0：資料可用性檢查（2021-01-01 ~ 2021-01-31）")
    print("=" * 60)

    probe_start, probe_end = "2021-01-01", "2021-01-31"
    ok = True

    price_limit = get_price_limit_data(probe_start, probe_end)
    print(f"\n[TaiwanStockPriceLimit] 筆數：{len(price_limit)}")
    if price_limit.empty:
        print("[警告] TaiwanStockPriceLimit 在該區間無資料。")
        ok = False
    else:
        print("columns:", list(price_limit.columns))
        print(price_limit.head(5).to_string())

    suspension = get_day_trading_suspension(probe_start, probe_end)
    print(f"\n[TaiwanStockDayTradingSuspension] 筆數：{len(suspension)}")
    if suspension.empty:
        print("[警告] TaiwanStockDayTradingSuspension 在該區間無資料。")
        ok = False
    else:
        print("columns:", list(suspension.columns))
        print(suspension.head(5).to_string())

    print("=" * 60)
    return ok


def print_verification(limit_up_events: pd.DataFrame,
                       filtered_events: pd.DataFrame,
                       price_limit_df: pd.DataFrame,
                       data: dict,
                       tx_regions: pd.Series) -> None:
    """步驟 4：印出驗證資訊供人工核對。

    包含：過濾前後筆數對比、隨機 5 筆漲停事件的 close/reference_price/limit_up
    三者並列，以及 filtered_events 依 TX 區間分組的事件數量。

    Args:
        limit_up_events: 過濾前的漲停事件。
        filtered_events: 過濾後（可當沖）的漲停事件。
        price_limit_df: 漲跌停價原始資料。
        data: 各股票標準 OHLCV 字典。
        tx_regions: TX 區間分類序列。
    """
    print("\n" + "=" * 60)
    print("步驟 4：驗證資訊")
    print("=" * 60)

    # (a) 過濾前後筆數
    removed = len(limit_up_events) - len(filtered_events)
    print(f"limit_up_events 總筆數 : {len(limit_up_events)}")
    print(f"filtered_events 總筆數 : {len(filtered_events)}")
    print(f"被過濾掉（暫停當沖）   : {removed}")

    # (b) 隨機 5 筆人工核對：close vs reference_price vs limit_up
    print("\n--- 隨機抽 5 筆漲停事件人工核對 ---")
    if limit_up_events.empty:
        print("（無漲停事件可核對）")
    else:
        pl = price_limit_df.copy()
        pl["date"] = pd.to_datetime(pl["date"])
        pl["stock_id"] = pl["stock_id"].astype(str)

        n = min(5, len(limit_up_events))
        sample = limit_up_events.sample(n=n, random_state=SEED)
        print(f"{'date':<12}{'stock_id':<10}{'close':>10}{'ref_price':>12}{'limit_up':>10}  核對")
        for _, row in sample.iterrows():
            d = pd.Timestamp(row["date"])
            sid = str(row["stock_id"])

            close = float("nan")
            if sid in data and d in data[sid].index:
                close = float(data[sid].loc[d, "close"])

            pl_row = pl[(pl["date"] == d) & (pl["stock_id"] == sid)]
            ref = float(pl_row["reference_price"].iloc[0]) if not pl_row.empty else float("nan")
            lu = float(pl_row["limit_up"].iloc[0]) if not pl_row.empty else float("nan")

            match = "✓" if (lu == lu and abs(close - lu) < 0.005) else "✗"
            print(f"{d.strftime('%Y-%m-%d'):<12}{sid:<10}"
                  f"{close:>10.2f}{ref:>12.2f}{lu:>10.2f}  {match}")

    # (c) filtered_events 依 TX 區間分組
    print("\n--- filtered_events 依 TX 區間分組的事件數 ---")
    if filtered_events.empty:
        print("（無事件）")
    else:
        ev = filtered_events.copy()
        ev["region"] = pd.to_datetime(ev["date"]).map(tx_regions)
        counts = ev["region"].value_counts(dropna=False)
        for region in REGIONS:
            print(f"{region} : {int(counts.get(region, 0))}")
        na_count = int(counts.get(float('nan'), 0)) if counts.isna().any() else 0
        # 區間為 NaN（TX 走平/開平或非交易日）
        na_total = int(ev["region"].isna().sum())
        print(f"（區間 NaN/其他）: {na_total}")
    print("=" * 60)


def main() -> None:
    """執行 TX 區間 × 漲停股放空（150 檔抽樣）的完整流程與回測。"""
    load_dotenv()

    # 步驟 0：資料可用性檢查
    if not check_data_availability():
        print("\n[停止] 探測區間缺資料，請確認是否調整 start_date 後再執行。")
        sys.exit(1)

    print(f"\n資料區間：{START_DATE} ~ {END_DATE}")

    # 步驟 1：取得全市場股票池（全部，不抽樣）
    all_stock_ids = get_all_stock_ids()
    print(f"步驟 1：全市場股票池共 {len(all_stock_ids)} 檔（不抽樣）")

    # 步驟 3：依序執行資料抓取與事件建立
    data = load_multiple(all_stock_ids, START_DATE, END_DATE)
    print(f"成功載入 {len(data)} 檔日線資料")

    tx_df = get_tx_data(START_DATE, END_DATE)
    print(f"TX 近月交易日數 {len(tx_df)}")

    tx_regions = classify_tx_regions(tx_df)
    print("TX 區間分布：")
    print(tx_regions.value_counts(dropna=False).to_string())

    price_limit_df = get_price_limit_data(START_DATE, END_DATE)
    print(f"漲跌停價資料 {len(price_limit_df)} 筆")

    suspension_df = get_day_trading_suspension(START_DATE, END_DATE)
    print(f"暫停當沖名單 {len(suspension_df)} 筆")

    limit_up_events = find_limit_up_events(price_limit_df, data)
    filtered_events = filter_day_trading_eligible(limit_up_events, suspension_df)

    # 步驟 4：驗證資訊
    print_verification(limit_up_events, filtered_events,
                       price_limit_df, data, tx_regions)

    # 步驟 5：依四種 TX 區間各自建立隔日放空訊號並回測
    all_dates = tx_df.index
    stock_ids = list(data.keys())

    for region in REGIONS:
        signals = build_region_signals(
            filtered_events, tx_regions, region, all_dates, stock_ids
        )
        n_short = int((signals == -1).sum().sum())

        bt = PortfolioBacktester()
        trades = bt.run(data, signals)

        print(f"\n=== {region} ===")
        print(f"放空訊號數：{n_short}，成交筆數：{len(trades)}")
        if trades.empty:
            print("（無交易，略過 report）")
            continue
        bt.daily_summary()
        bt.report()

    # ===================================================================
    # 步驟 6：盤中觸及漲停（high >= limit_up*0.995）事件回測（停損 9%）
    # ===================================================================
    print("\n" + "=" * 60)
    print("步驟 6：盤中觸及漲停（high）事件回測（停損 9%）")
    print("=" * 60)

    # 6-1. 用 use_intraday=True 產生新的事件清單並過濾
    intraday_events = find_limit_up_events(price_limit_df, data, use_intraday=True)
    intraday_filtered = filter_day_trading_eligible(intraday_events, suspension_df)
    print(f"新事件總筆數（high 觸及）：{len(intraday_filtered)}"
          f"（對比原本 close 定義 {len(filtered_events)} 筆）")

    # 6-2. 新事件依 TX region 分布
    close_region = (
        filtered_events.assign(
            region=pd.to_datetime(filtered_events["date"]).map(tx_regions))
        ["region"].value_counts()
    )
    high_region = (
        intraday_filtered.assign(
            region=pd.to_datetime(intraday_filtered["date"]).map(tx_regions))
        ["region"].value_counts()
    )
    print("新事件依 TX region 分布：")
    for region in REGIONS:
        print(f"  {region}: {int(high_region.get(region, 0))}")

    # 6-3. 依四種 TX region 各自回測（停損 9%）
    intraday_summary = []
    for region in REGIONS:
        signals = build_region_signals(
            intraday_filtered, tx_regions, region, all_dates, stock_ids
        )
        n_short = int((signals == -1).sum().sum())

        print(f"\n=== {region}（盤中觸及, 停損 9%）===")
        print(f"放空訊號數：{n_short}")

        bt = PortfolioBacktester(stop_loss_pct=INTRADAY_STOP_LOSS_PCT)
        trades = bt.run(data, signals)

        row = {
            "region": region,
            "close_n": int(close_region.get(region, 0)),
            "high_n": int(high_region.get(region, 0)),
            "總損益": None, "勝率": None, "Sharpe": None,
        }
        if trades.empty:
            print("（無交易，略過 report）")
        else:
            bt.daily_summary()
            metrics = bt.report()
            row["總損益"] = metrics["總損益"]
            row["勝率"] = metrics["整體勝率"]
            row["Sharpe"] = metrics["Sharpe Ratio"]
        intraday_summary.append(row)

    # 6-4. close vs high 對比摘要表
    print("\n" + "=" * 60)
    print("close vs high 事件定義對比摘要（high 版停損 9%）")
    print("=" * 60)
    print(f"{'region':<8}| {'原始筆數(close)':>14} | {'新筆數(high)':>12} | "
          f"{'總損益':>13} | {'勝率':>7} | {'Sharpe':>9}")
    print("-" * 78)
    for r in intraday_summary:
        if r["總損益"] is None:
            print(f"{r['region']:<8}| {r['close_n']:>14} | {r['high_n']:>12} | "
                  f"{'(無交易)':>13} | {'-':>7} | {'-':>9}")
        else:
            print(f"{r['region']:<8}| {r['close_n']:>14} | {r['high_n']:>12} | "
                  f"{r['總損益']:>13,.2f} | {r['勝率']:>6.2%} | {r['Sharpe']:>9.4f}")


if __name__ == "__main__":
    main()
