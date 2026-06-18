"""診斷腳本：為何「開高走低」region 的漲停事件數為 0。

重建主程式的管線狀態（股票資料一律走 CSV 快取，僅 TX / 漲跌停價 /
暫停名單需重新抓取），然後執行四項檢查：

  1. tx_regions.value_counts()，確認 '開高走低' 確實存在於分類結果
  2. filtered_events 每筆 (date, stock_id) 對應的 tx_regions 值分布
  3. 用 repr() 比對迴圈字串 '開高走低' 與 tx_regions 實際值是否 byte-for-byte 相同
  4. 列出標記為 '開高走低' 的前 10 個日期，檢查原始漲停清單是否有事件

執行前提：data/ 已有主程式跑完留下的股票 CSV 快取。
"""

import os

import pandas as pd
from dotenv import load_dotenv

from data_loader import (
    _to_standard_ohlcv,
    get_day_trading_suspension,
    get_price_limit_data,
    get_tx_data,
)
from market_scanner import filter_day_trading_eligible, find_limit_up_events
from strategy import build_region_signals, classify_tx_regions
from portfolio_backtest import PortfolioBacktester


START_DATE = "2015-01-01"
END_DATE = "2026-06-01"
TARGET_REGION = "開高走低"
DATA_DIR = "data"


def _cached_stock_ids() -> list:
    """從 data/ 既有 CSV 還原股票池（即主程式實際載入成功的標的）。

    Returns:
        股票代號清單（CSV 檔名去副檔名）。
    """
    if not os.path.isdir(DATA_DIR):
        return []
    ids = [f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    return sorted(ids)


def _load_cached_only(stock_ids: list, start_date: str,
                      end_date: str) -> dict:
    """只從既有 CSV 快取讀取股票資料，不呼叫 API（讓重跑很快）。

    直接讀檔並裁切日期範圍，不做 load_data 的「快取完整性」檢查，
    因此不會因為快取少一兩天就觸發整批重抓。缺檔/空檔/壞檔一律略過。

    Args:
        stock_ids: 股票代號清單。
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        dict[str, pd.DataFrame]，標準 OHLCV，已裁切至指定範圍。
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    data = {}
    for stock_id in stock_ids:
        path = os.path.join(DATA_DIR, f"{stock_id}.csv")
        if not os.path.exists(path):
            continue
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            if cached.empty:
                continue
            df = _to_standard_ohlcv(cached).loc[start:end]
        except Exception:  # noqa: BLE001
            continue
        if not df.empty:
            data[stock_id] = df
    return data


def rebuild_pipeline_state():
    """重建管線：data / tx_df / tx_regions / limit_up_events / filtered_events。

    股票資料一律 cache-only（不呼叫 API），僅 TX / 漲跌停價 / 暫停名單
    需向 FinMind 抓取。

    Returns:
        (data, tx_df, tx_regions, limit_up_events, filtered_events) tuple。
    """
    stock_ids = _cached_stock_ids()
    print(f"[重建] 從快取還原 {len(stock_ids)} 檔股票池（cache-only，不抓 API）")
    data = _load_cached_only(stock_ids, START_DATE, END_DATE)
    print(f"[重建] 成功載入 {len(data)} 檔")

    tx_df = get_tx_data(START_DATE, END_DATE)
    tx_regions = classify_tx_regions(tx_df)

    price_limit_df = get_price_limit_data(START_DATE, END_DATE)
    suspension_df = get_day_trading_suspension(START_DATE, END_DATE)

    limit_up_events = find_limit_up_events(price_limit_df, data)
    filtered_events = filter_day_trading_eligible(limit_up_events, suspension_df)

    print(f"[重建] limit_up_events={len(limit_up_events)}，"
          f"filtered_events={len(filtered_events)}")
    return data, tx_df, tx_regions, limit_up_events, filtered_events


def check_1_region_distribution(tx_regions: pd.Series) -> None:
    """檢查 1：tx_regions 的分類分布，確認 '開高走低' 存在。"""
    print("\n" + "=" * 60)
    print("檢查 1：tx_regions.value_counts()")
    print("=" * 60)
    print(tx_regions.value_counts(dropna=False).to_string())
    exists = bool((tx_regions == TARGET_REGION).any())
    print(f"\n'{TARGET_REGION}' 是否存在於分類結果：{exists}")
    print(f"'{TARGET_REGION}' 的天數：{int((tx_regions == TARGET_REGION).sum())}")


def check_2_event_region_distribution(filtered_events: pd.DataFrame,
                                      tx_regions: pd.Series) -> None:
    """檢查 2：漲停事件發生當天 TX 屬於哪個 region 的分布。"""
    print("\n" + "=" * 60)
    print("檢查 2：filtered_events 每筆對應的 tx_regions 值分布")
    print("=" * 60)
    if filtered_events.empty:
        print("（filtered_events 為空）")
        return
    ev = filtered_events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev_regions = ev["date"].map(tx_regions)
    print(f"事件總數：{len(ev)}")
    print(ev_regions.value_counts(dropna=False).to_string())


def check_3_string_identity(tx_regions: pd.Series) -> None:
    """檢查 3：repr() 比對迴圈字串與 tx_regions 實際值是否完全相同。"""
    print("\n" + "=" * 60)
    print("檢查 3：字串 byte-for-byte 比對")
    print("=" * 60)
    loop_str = "開高走低"  # 主程式迴圈中使用的字面值
    print(f"迴圈字串      repr : {loop_str!r}")
    print(f"迴圈字串      bytes: {loop_str.encode('utf-8')}")

    mask = tx_regions == loop_str
    if not mask.any():
        print(f"\n[注意] tx_regions 中找不到等於 {loop_str!r} 的值，無法比對。")
        # 反向：列出所有唯一非空類別的 repr，看是否有近似字串
        uniques = [u for u in tx_regions.dropna().unique()]
        print("tx_regions 唯一類別（repr）：")
        for u in uniques:
            print(f"  {u!r}  bytes={u.encode('utf-8')}")
        return

    actual = tx_regions[mask].iloc[0]
    print(f"tx_regions 值 repr : {actual!r}")
    print(f"tx_regions 值 bytes: {actual.encode('utf-8')}")
    print(f"\n== 相等          : {loop_str == actual}")
    print(f"bytes 完全相同   : {loop_str.encode('utf-8') == actual.encode('utf-8')}")


def check_4_dates_vs_raw_events(tx_regions: pd.Series,
                                limit_up_events: pd.DataFrame) -> None:
    """檢查 4：'開高走低' 前 10 個日期，原始漲停清單是否有事件。"""
    print("\n" + "=" * 60)
    print("檢查 4：'開高走低' 日期 vs 原始(未篩選)漲停事件")
    print("=" * 60)
    target_dates = tx_regions[tx_regions == TARGET_REGION].index
    print(f"'{TARGET_REGION}' 總天數：{len(target_dates)}")

    lu = limit_up_events.copy()
    if not lu.empty:
        lu["date"] = pd.to_datetime(lu["date"])

    print(f"\n前 10 個 '{TARGET_REGION}' 日期，各自的原始漲停事件數：")
    for d in target_dates[:10]:
        hits = lu[lu["date"] == d] if not lu.empty else lu
        ids = list(hits["stock_id"]) if not hits.empty else []
        print(f"  {d.date()} : {len(ids)} 筆 {ids}")

    # 關鍵彙總：所有開高走低日的原始漲停事件總數
    if not lu.empty:
        total = int(lu["date"].isin(set(target_dates)).sum())
    else:
        total = 0
    print(f"\n所有 '{TARGET_REGION}' 日的原始漲停事件總數：{total}")
    if total == 0:
        print(f"=> 結論：'{TARGET_REGION}' 當天根本沒有任何個股漲停作收，"
              f"故該 region 事件數為 0 屬市場現象，非字串比對 bug。")


def check_5_distinct_dates_and_manual(filtered_events: pd.DataFrame,
                                      tx_regions: pd.Series,
                                      tx_df: pd.DataFrame) -> None:
    """檢查 5：distinct 日期、交集、與人工核對 TX OHLC 判斷邏輯。"""
    print("\n" + "=" * 60)
    print("檢查 5：distinct 日期 / 交集 / TX 判斷邏輯人工核對")
    print("=" * 60)

    if filtered_events.empty:
        print("（filtered_events 為空）")
        return

    ev = filtered_events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    tx_prev_close = tx_df["close"].shift(1)

    # 5-1) distinct 日期數 + 事件最集中的日期
    n_distinct = ev["date"].nunique()
    print(f"\n[5-1] 399 筆事件橫跨 distinct 日期數：{n_distinct}")
    print("事件最集中的前 15 天（date: 事件數）：")
    print(ev["date"].value_counts().head(15).to_string())

    # 5-2) 事件日 ∩ 開高走低日
    event_dates = set(ev["date"].unique())
    kgzd_dates = set(tx_regions[tx_regions == TARGET_REGION].index)
    inter = sorted(event_dates & kgzd_dates)
    print(f"\n[5-2] 事件日({len(event_dates)} 個) ∩ '{TARGET_REGION}'日"
          f"({len(kgzd_dates)} 個) 交集數量：{len(inter)}")
    print("交集日期：", [d.date() for d in inter] if inter else "（無交集）")

    # 5-3) 隨機 5 筆事件，人工核對 TX 判斷邏輯
    print(f"\n[5-3] 隨機抽 5 筆事件，核對當天 TX open/prev_close/close 與 region：")
    n = min(5, len(ev))
    sample = ev.sample(n=n, random_state=42)
    header = (f"{'date':<12}{'stock':<8}{'tx_open':>10}{'prev_close':>12}"
              f"{'tx_close':>10}  {'開高?':<6}{'走低?':<6}{'region':<8}")
    print(header)
    for _, row in sample.iterrows():
        d = pd.Timestamp(row["date"])
        sid = str(row["stock_id"])
        if d not in tx_df.index:
            print(f"{d.strftime('%Y-%m-%d'):<12}{sid:<8}  （該日不在 TX 交易日）")
            continue
        op = float(tx_df.loc[d, "open"])
        pc = float(tx_prev_close.loc[d])
        cl = float(tx_df.loc[d, "close"])
        region = tx_regions.loc[d]
        gao = "開高" if op > pc else ("開低" if op < pc else "開平")
        zou = "走低" if cl < op else ("走高" if cl > op else "走平")
        print(f"{d.strftime('%Y-%m-%d'):<12}{sid:<8}{op:>10.1f}{pc:>12.1f}"
              f"{cl:>10.1f}  {gao:<6}{zou:<6}{str(region):<8}")

    # 5-4) 隨機 5 個開高走低日，核對 open>prev_close 且 close<open
    print(f"\n[5-4] 隨機抽 5 個 '{TARGET_REGION}' 日，核對 open>prev_close 且 close<open：")
    kgzd_index = tx_regions[tx_regions == TARGET_REGION].index
    sample_dates = pd.Series(list(kgzd_index)).sample(
        n=min(5, len(kgzd_index)), random_state=42
    )
    print(f"{'date':<12}{'tx_open':>10}{'prev_close':>12}{'tx_close':>10}  驗證")
    for d in sample_dates:
        d = pd.Timestamp(d)
        op = float(tx_df.loc[d, "open"])
        pc = float(tx_prev_close.loc[d])
        cl = float(tx_df.loc[d, "close"])
        ok = (op > pc) and (cl < op)
        mark = "✓ 開高走低" if ok else "✗ 不符"
        print(f"{d.strftime('%Y-%m-%d'):<12}{op:>10.1f}{pc:>12.1f}{cl:>10.1f}  {mark}")


def _print_region_yearly(region: str,
                         filtered_events: pd.DataFrame,
                         tx_regions: pd.Series,
                         tx_df: pd.DataFrame,
                         data: dict) -> None:
    """印出單一 region 的逐年（年份｜筆數｜總損益｜勝率）表。

    對該 region 建立隔日放空訊號並回測，再依交易日年份彙總。
    """
    print(f"\n--- {region} ---")

    all_dates = tx_df.index
    stock_ids = list(data.keys())
    signals = build_region_signals(filtered_events, tx_regions, region,
                                   all_dates, stock_ids)
    bt = PortfolioBacktester()
    trades = bt.run(data, signals)

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
    start_year = pd.Timestamp(START_DATE).year
    end_year = pd.Timestamp(END_DATE).year
    for year in range(start_year, end_year + 1):
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


def check_6_yearly_breakdown(filtered_events: pd.DataFrame,
                             tx_regions: pd.Series,
                             tx_df: pd.DataFrame,
                             data: dict) -> None:
    """檢查 6：四個 region 各自按年份分組的筆數 / 總損益 / 勝率。

    四個 region 分開印，方便逐一比較穩定度。
    """
    print("\n" + "=" * 60)
    print("檢查 6：四個 region 逐年績效")
    print("=" * 60)

    for region in ["開高走高", "開高走低", "開低走高", "開低走低"]:
        _print_region_yearly(region, filtered_events, tx_regions, tx_df, data)


def main() -> None:
    """執行六項診斷檢查。"""
    load_dotenv()
    (data, tx_df, tx_regions,
     limit_up_events, filtered_events) = rebuild_pipeline_state()

    check_1_region_distribution(tx_regions)
    check_2_event_region_distribution(filtered_events, tx_regions)
    check_3_string_identity(tx_regions)
    check_4_dates_vs_raw_events(tx_regions, limit_up_events)
    check_5_distinct_dates_and_manual(filtered_events, tx_regions, tx_df)
    check_6_yearly_breakdown(filtered_events, tx_regions, tx_df, data)


if __name__ == "__main__":
    main()
